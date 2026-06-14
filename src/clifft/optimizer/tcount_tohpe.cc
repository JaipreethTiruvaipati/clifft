#include "clifft/optimizer/tcount_tohpe.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace clifft {

namespace {

uint32_t words_for(uint32_t n_bits) {
    return (n_bits + 63) / 64;
}

// GF(2) nullspace: given `rows` (each an m-bit vector packed in uint64 words)
// return a basis of {y in F_2^m : rows . y = 0}. Each basis vector is m bits.
std::vector<std::vector<uint64_t>> gf2_nullspace(std::vector<std::vector<uint64_t>> rows,
                                                 uint32_t m) {
    const uint32_t mw = words_for(m);
    auto bit = [](const std::vector<uint64_t>& v, uint32_t i) {
        return (v[i / 64] >> (i % 64)) & 1ULL;
    };

    // Row-reduce to find pivots over the m columns.
    std::vector<int> pivot_col;  // pivot column for each kept row
    std::vector<std::vector<uint64_t>> reduced;
    for (auto& r : rows) {
        std::vector<uint64_t> cur = r;
        for (size_t k = 0; k < reduced.size(); ++k) {
            if (bit(cur, static_cast<uint32_t>(pivot_col[k]))) {
                for (uint32_t w = 0; w < mw; ++w)
                    cur[w] ^= reduced[k][w];
            }
        }
        // find leading bit
        int lead = -1;
        for (uint32_t c = 0; c < m; ++c) {
            if (bit(cur, c)) {
                lead = static_cast<int>(c);
                break;
            }
        }
        if (lead < 0)
            continue;  // dependent row
        // eliminate this pivot from previous rows (reduced row echelon)
        for (size_t k = 0; k < reduced.size(); ++k) {
            if (bit(reduced[k], static_cast<uint32_t>(lead))) {
                for (uint32_t w = 0; w < mw; ++w)
                    reduced[k][w] ^= cur[w];
            }
        }
        reduced.push_back(cur);
        pivot_col.push_back(lead);
    }

    std::vector<bool> is_pivot(m, false);
    for (int pc : pivot_col)
        is_pivot[pc] = true;

    // Free columns parameterize the nullspace.
    std::vector<std::vector<uint64_t>> basis;
    for (uint32_t free = 0; free < m; ++free) {
        if (is_pivot[free])
            continue;
        std::vector<uint64_t> y(mw, 0);
        y[free / 64] |= (1ULL << (free % 64));
        // For each pivot row with a 1 in `free`, set the pivot var to satisfy r.y = 0.
        for (size_t k = 0; k < reduced.size(); ++k) {
            if (bit(reduced[k], free))
                y[pivot_col[k] / 64] |= (1ULL << (pivot_col[k] % 64));
        }
        basis.push_back(std::move(y));
    }
    return basis;
}

int popcount_words(const std::vector<uint64_t>& v) {
    int c = 0;
    for (uint64_t w : v)
        c += std::popcount(w);
    return c;
}

// Remove all-zero columns and cancel pairs of identical columns, pushing an
// S (coeff 2) residual for each cancelled duplicate pair. Returns true if any
// column was removed.
bool properize(std::vector<ParityColumn>& cols, std::vector<ResidualPhase>& residuals) {
    bool changed = false;
    // Drop zero columns.
    std::vector<ParityColumn> kept;
    for (auto& c : cols) {
        if (c.is_zero())
            changed = true;
        else
            kept.push_back(std::move(c));
    }
    cols.swap(kept);

    // Cancel duplicate pairs: two T gates on the same parity = S on that parity.
    bool again = true;
    while (again) {
        again = false;
        for (size_t i = 0; i < cols.size() && !again; ++i) {
            for (size_t j = i + 1; j < cols.size(); ++j) {
                if (cols[i] == cols[j]) {
                    residuals.push_back(ResidualPhase{cols[i], 2});
                    // erase j then i (j > i)
                    cols.erase(cols.begin() + static_cast<std::ptrdiff_t>(j));
                    cols.erase(cols.begin() + static_cast<std::ptrdiff_t>(i));
                    changed = true;
                    again = true;
                    break;
                }
            }
        }
    }
    return changed;
}

// f(x) = (sum over columns of [col . x]) mod 8, evaluated for all x in
// [0, 2^n_bits). Only valid for n_bits <= 14 (caller guards).
std::vector<uint8_t> phase_function(const std::vector<ParityColumn>& cols,
                                    const std::vector<ResidualPhase>& residuals, uint32_t n_bits) {
    const uint64_t N = 1ULL << n_bits;
    std::vector<uint8_t> f(N, 0);
    for (uint64_t x = 0; x < N; ++x) {
        int acc = 0;
        for (const auto& c : cols)
            acc += static_cast<int>(std::popcount(c.words[0] & x) & 1ULL);
        for (const auto& r : residuals)
            acc += r.coeff_mod8 * static_cast<int>(std::popcount(r.parity.words[0] & x) & 1ULL);
        f[x] = static_cast<uint8_t>(((acc % 8) + 8) % 8);
    }
    return f;
}

}  // namespace

TohpeResult tohpe_reduce(std::vector<ParityColumn> columns, uint32_t n_bits, size_t max_cols,
                         uint32_t max_verify_bits) {
    TohpeResult result;
    result.t_before = columns.size();

    // A reduction is accepted only after checking it preserves the exact phase
    // function f(x) over all 2^n_bits inputs. The f-table is one byte per entry,
    // so the default cap of 14 holds it to 16 KB; it bounds verification cost,
    // not the algorithm, and is exposed as a parameter. The reducer runs once
    // per block at compile time, not per shot, so this is not a runtime
    // hot-path. Blocks wider than the cap are returned unchanged rather than
    // reduced unverified. The single-word guard keeps the table indexable by a
    // uint64 mask.
    const bool verifiable = (n_bits <= max_verify_bits) && (words_for(n_bits) == 1);
    std::vector<ParityColumn> original = columns;
    std::vector<uint8_t> f_target;
    if (verifiable)
        f_target = phase_function(original, {}, n_bits);

    // `max_cols` bounds the number of odd-coefficient parities (columns) the
    // reducer will attempt. Each outer iteration computes a null-space basis of
    // an O(n_bits^2)-row system and, per null vector, scores all candidate z in
    // O(m^2) (the S(z) hash step), then verifies the chosen move with one
    // O(2^n_bits) phase-function check. That is fast in practice (wide dense
    // blocks reduce in milliseconds to a few seconds); the cap is a safety bound
    // for pathological widths, and larger blocks are returned unchanged.
    if (columns.size() > max_cols || !verifiable) {
        result.columns = std::move(columns);
        result.t_after = result.columns.size();
        return result;
    }

    std::vector<ResidualPhase> residuals;
    properize(columns, residuals);

    const uint32_t mw_n = words_for(n_bits);

    // Duplicate-and-destroy loop (Vandaele Algorithm 2 / Theorem 1). Columns only
    // decrease, so the loop terminates; the guard is a backstop.
    bool progress = true;
    int guard = 0;
    while (progress && guard++ < 4096) {
        progress = false;
        const uint32_t m = static_cast<uint32_t>(columns.size());
        if (m < 2)
            break;

        // Build constraint matrix L: rows P_a (a in [0,n_bits)) and P_a & P_b
        // (a < b). Each row is an m-bit vector over the columns.
        std::vector<std::vector<uint64_t>> L;
        const uint32_t mw = words_for(m);
        auto make_row = [&](auto pred) {
            std::vector<uint64_t> row(mw, 0);
            for (uint32_t j = 0; j < m; ++j)
                if (pred(j))
                    row[j / 64] |= (1ULL << (j % 64));
            return row;
        };
        for (uint32_t a = 0; a < n_bits; ++a)
            L.push_back(make_row([&](uint32_t j) { return columns[j].get(a); }));
        for (uint32_t a = 0; a < n_bits; ++a)
            for (uint32_t b = a + 1; b < n_bits; ++b)
                L.push_back(
                    make_row([&](uint32_t j) { return columns[j].get(a) && columns[j].get(b); }));

        auto nullspace = gf2_nullspace(L, m);
        auto vbit = [](const std::vector<uint64_t>& v, uint32_t j) {
            return (v[j / 64] >> (j % 64)) & 1ULL;
        };

        // Vandaele Algorithm 2 (TOHPE) scoring, arXiv:2407.08695. For a fixed
        // null vector y the update A -> A xor z y^T moves every column i with
        // y_i = 1 to col_i xor z and leaves the others alone, so a moved column i
        // and an unmoved column j collide -- a destroyable duplicate -- exactly
        // when z = col_i xor col_j. A hash map tallies, for each candidate z, how
        // many such collisions it makes (+2 per (moved, unmoved) pair) plus a
        // seed of 1 for every column a fresh z could duplicate on its own. That
        // scores all z for this y in O(m^2) -- the S(z) step -- instead of
        // copying and properizing a trial for every (z, y) pair. We keep each
        // y's best-scoring z; the loop below applies the best overall that also
        // preserves the exact phase function.
        struct Cand {
            int score;
            uint64_t z;
            uint32_t yi;
        };
        std::vector<Cand> cands;
        auto score_y = [&](uint32_t yi, std::unordered_map<uint64_t, int>& score,
                           std::vector<Cand>& out) {
            const std::vector<uint64_t>& y = nullspace[yi];
            int wy = popcount_words(y);
            if (wy == 0)
                return;
            const bool parity = (wy & 1);
            score.clear();
            for (uint32_t i = 0; i < m; ++i)
                if (static_cast<bool>(vbit(y, i)) != parity)
                    score.emplace(columns[i].words[0], 1);
            for (uint32_t i = 0; i < m; ++i) {
                if (!vbit(y, i))
                    continue;
                for (uint32_t j = 0; j < m; ++j)
                    if (!vbit(y, j))
                        score[columns[i].words[0] ^ columns[j].words[0]] += 2;
            }
            int best = 0;
            uint64_t bz = 0;
            bool has = false;
            for (const auto& [zv, sc] : score)
                if (sc > best || (sc == best && has && zv < bz)) {
                    best = sc;
                    bz = zv;
                    has = true;
                }
            if (has && best > 0)
                out.push_back({best, bz, yi});
        };
        const int64_t ny = static_cast<int64_t>(nullspace.size());
#ifdef _OPENMP
#pragma omp parallel
        {
            std::vector<Cand> local;
            std::unordered_map<uint64_t, int> score;
#pragma omp for schedule(dynamic, 4) nowait
            for (int64_t yi = 0; yi < ny; ++yi)
                score_y(static_cast<uint32_t>(yi), score, local);
#pragma omp critical
            cands.insert(cands.end(), local.begin(), local.end());
        }
#else
        {
            std::unordered_map<uint64_t, int> score;
            for (int64_t yi = 0; yi < ny; ++yi)
                score_y(static_cast<uint32_t>(yi), score, cands);
        }
#endif

        // Apply the best-scoring move that strictly reduces and preserves the
        // exact phase function. The signature-preserving update can still shift f
        // by a Clifford term this residual model cannot express, so such moves
        // are rejected and we fall through to the next candidate. Ties broken by
        // (z, y) index so the choice is deterministic across thread counts.
        std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
            if (a.score != b.score)
                return a.score > b.score;
            if (a.z != b.z)
                return a.z < b.z;
            return a.yi < b.yi;
        });
        for (const Cand& c : cands) {
            const std::vector<uint64_t>& y = nullspace[c.yi];
            ParityColumn z{std::vector<uint64_t>(mw_n, 0)};
            z.words[0] = c.z;
            std::vector<ParityColumn> trial = columns;
            for (uint32_t j = 0; j < m; ++j)
                if (vbit(y, j))
                    trial[j].xor_with(z);
            if (popcount_words(y) & 1)
                trial.push_back(z);  // keep |y| even (Vandaele C1)
            if (phase_function(trial, residuals, n_bits) != f_target)
                continue;
            std::vector<ResidualPhase> trial_res = residuals;
            properize(trial, trial_res);
            if (trial.size() < columns.size()) {
                columns.swap(trial);
                residuals.swap(trial_res);
                progress = true;
                break;
            }
        }
    }

    // Final verification (defensive): if anything drifted, return the original.
    auto f_final = phase_function(columns, residuals, n_bits);
    if (f_final != f_target) {
        result.columns = std::move(original);
        result.t_after = result.columns.size();
        return result;
    }

    result.columns = std::move(columns);
    result.residuals = std::move(residuals);
    result.t_after = result.columns.size();
    return result;
}

}  // namespace clifft
