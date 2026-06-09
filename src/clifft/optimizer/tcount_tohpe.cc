#include "clifft/optimizer/tcount_tohpe.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <vector>

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

TohpeResult tohpe_reduce(std::vector<ParityColumn> columns, uint32_t n_bits, size_t max_cols) {
    TohpeResult result;
    result.t_before = columns.size();

    // We accept a reduction only after checking it preserves the exact phase
    // function f(x) over all 2^n_bits inputs, where n_bits is the parity width
    // (the qubit support of the block). The cutoff 14 is a verification-cost
    // bound, not an algorithmic one: 2^14 = 16384 bytes per f-table, evaluated
    // for each candidate move, stays in L1 and costs microseconds, while 2^20+
    // would dominate runtime. Blocks whose support exceeds 14 (rare for the
    // localized blocks the front end produces) are returned unchanged rather
    // than reduced unverified. The single-word guard keeps the f-table indexable
    // by a uint64 mask.
    const bool verifiable = (n_bits <= 14) && (words_for(n_bits) == 1);
    std::vector<ParityColumn> original = columns;
    std::vector<uint8_t> f_target;
    if (verifiable)
        f_target = phase_function(original, {}, n_bits);

    if (columns.size() > max_cols || !verifiable) {
        result.columns = std::move(columns);
        result.t_after = result.columns.size();
        return result;
    }

    std::vector<ResidualPhase> residuals;
    properize(columns, residuals);

    const uint32_t mw_n = words_for(n_bits);

    // Duplicate-and-destroy loop (Vandaele Algorithm 2 / Theorem 1).
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

        // Duplicate-and-destroy (Heyfron-Campbell Lemma III.2; Vandaele Alg. 2).
        // The candidate set of update vectors z is the pairwise column XORs
        // together with the single columns: Z = {col_i xor col_j} u {col_i}
        // (Vandaele Alg. 2, line 2). For each candidate z and each null vector
        // y, the update A -> A xor z y^T (appending z when |y| is odd, to keep
        // |y| even per condition C1) preserves the order-3 signature tensor;
        // properize then destroys the duplicate / zeroed columns. We pick the
        // move that removes the most columns (the objective maximization of
        // Alg. 2, rather than the first feasible move), and verify the exact
        // phase function before committing -- the safety net the signature
        // tensor alone does not give.
        std::vector<ParityColumn> z_candidates;
        z_candidates.reserve(static_cast<size_t>(m) * (m + 1) / 2);
        for (uint32_t a = 0; a < m; ++a) {
            z_candidates.push_back(columns[a]);  // single-column z = col_a
            for (uint32_t b = a + 1; b < m; ++b) {
                ParityColumn z = columns[a];
                z.xor_with(columns[b]);
                z_candidates.push_back(std::move(z));
            }
        }

        size_t best_removed = 0;
        std::vector<ParityColumn> best_cols;
        std::vector<ResidualPhase> best_res;
        for (const auto& z : z_candidates) {
            if (z.is_zero())
                continue;
            for (const auto& y : nullspace) {
                int wy = popcount_words(y);
                if (wy == 0)
                    continue;

                std::vector<ParityColumn> trial = columns;
                for (uint32_t j = 0; j < m; ++j)
                    if (vbit(y, j))
                        trial[j].xor_with(z);
                if (wy & 1)
                    trial.push_back(z);  // keep |y| even (Vandaele C1)

                std::vector<ResidualPhase> trial_res = residuals;
                properize(trial, trial_res);
                if (trial.size() >= columns.size())
                    continue;
                size_t removed = columns.size() - trial.size();
                // Verify only when this move could beat the current best, then
                // accept it as the new best if it preserves the phase function.
                if (removed > best_removed &&
                    phase_function(trial, trial_res, n_bits) == f_target) {
                    best_removed = removed;
                    best_cols = std::move(trial);
                    best_res = std::move(trial_res);
                }
            }
        }

        if (best_removed > 0) {
            columns.swap(best_cols);
            residuals.swap(best_res);
            progress = true;
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
    (void)mw_n;
    return result;
}

}  // namespace clifft
