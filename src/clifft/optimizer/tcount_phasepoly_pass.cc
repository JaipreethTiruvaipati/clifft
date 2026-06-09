#include "clifft/optimizer/tcount_phasepoly_pass.h"

#include "clifft/optimizer/commutation.h"
#include "clifft/optimizer/tcount_tohpe.h"
#include "clifft/util/constants.h"
#include "clifft/util/stim_mask.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace clifft {

namespace {

// Normalize a T gate to a positive Pauli sign, absorbing the phase
// difference into global_weight (identical convention to PeepholeFusionPass):
//   T(-P)     = exp(+i*pi/4) * T_dag(+P)
//   T_dag(-P) = exp(-i*pi/4) * T(+P)
// After this, the op has sign=false and its rotation direction depends only
// on the dagger flag.
void normalize_t_sign(HirModule& hir, HeisenbergOp& op) {
    if (op.op_type() == OpType::T_GATE && hir.sign(op)) {
        hir.global_weight *= op.is_dagger() ? kExpMinusIPiOver4 : kExpIPiOver4;
        op.set_dagger(!op.is_dagger());
        hir.set_sign(op, false);
    }
}

// Do ops i and j (both already known to be T_GATE) commute?
bool ops_commute(const HirModule& hir, const HeisenbergOp& a, const HeisenbergOp& b) {
    return !anti_commute(hir.destab_mask(a), hir.stab_mask(a), hir.destab_mask(b),
                         hir.stab_mask(b));
}

// Same virtual Pauli axis (X and Z masks equal; sign already normalized away).
bool same_axis(const HirModule& hir, const HeisenbergOp& a, const HeisenbergOp& b) {
    return hir.destab_mask(a) == hir.destab_mask(b) && hir.stab_mask(a) == hir.stab_mask(b);
}

// One re-emitted operation: either a T_GATE (with dagger) or a Clifford-angle
// PHASE_ROTATION (alpha in half-turns).
struct OutOp {
    bool is_t;
    bool dagger;   // for is_t
    double alpha;  // for !is_t
};

// Minimal HIR representation of T^c about a single axis, c in [0, 8).
// Always emits at most one T_GATE; any even (Clifford) remainder rides along
// as a PHASE_ROTATION (0.5 = S, 1.5 = S_dag, 1.0 = Z) that downstream passes
// and the back-end absorb. Exact: T^c = diag(1, e^{i*pi*c/4}).
std::vector<OutOp> synthesize(int c) {
    switch (((c % 8) + 8) % 8) {
        case 0:
            return {};
        case 1:
            return {{true, false, 0.0}};
        case 2:
            return {{false, false, 0.5}};  // S
        case 3:
            return {{true, false, 0.0}, {false, false, 0.5}};  // T * S
        case 4:
            return {{false, false, 1.0}};  // Z
        case 5:
            return {{true, true, 0.0}, {false, false, 1.5}};  // T_dag * S_dag
        case 6:
            return {{false, false, 1.5}};  // S_dag
        case 7:
            return {{true, true, 0.0}};  // T_dag
        default:
            return {};
    }
}

// A folded axis group: the op slots that share one virtual Pauli axis, and the
// Z_8 coefficient they fold to.
struct FoldedGroup {
    std::vector<size_t> slots;
    int c;
};

// Overwrite an op's arena mask with a single-type Pauli: the parity bits go in
// the X plane when `is_x` (X-type axis), otherwise the Z plane (Z-type axis).
void write_pauli_mask(HirModule& hir, const HeisenbergOp& op, const std::vector<uint64_t>& parity,
                      bool is_x) {
    auto m = hir.mask_at(op);
    auto x = m.x();
    auto z = m.z();
    for (uint32_t w = 0; w < x.num_words(); ++w) {
        uint64_t p = (w < parity.size()) ? parity[w] : 0;
        x.words[w] = is_x ? p : 0;
        z.words[w] = is_x ? 0 : p;
    }
    m.set_sign(false);
}

void emit_t(HirModule& hir, HeisenbergOp& op, const std::vector<uint64_t>& parity, bool is_x) {
    op.reset_to_tgate(false);
    write_pauli_mask(hir, op, parity, is_x);
}

void emit_pr(HirModule& hir, HeisenbergOp& op, double alpha, const std::vector<uint64_t>& parity,
             bool is_x) {
    op.reset_to_phase_rotation(alpha);
    write_pauli_mask(hir, op, parity, is_x);
}

// Map a local (support-indexed) parity column back to a full-width Z mask.
std::vector<uint64_t> col_to_zmask(const ParityColumn& c, const std::vector<uint32_t>& support,
                                   uint32_t nwords) {
    std::vector<uint64_t> z(nwords, 0);
    for (uint32_t k = 0; k < support.size(); ++k) {
        if (c.get(k)) {
            uint32_t q = support[k];
            z[q / 64] |= (1ULL << (q % 64));
        }
    }
    return z;
}

// ---------------------------------------------------------------------------
// Mixed-Pauli-type blocks: lift the single-type restriction by reducing in a
// symplectic basis (Phase B for blocks whose axes mix X and Z planes).
//
// A commuting set of Pauli rotations is simultaneously diagonal in the joint
// eigenbasis of a generating set {g_1..g_r} of independent, pairwise-commuting
// Paulis. Each axis P_k is, on (x, z) masks, a GF(2) combination of the
// generators, so its parity coordinate is that combination -- a Z-type parity
// in the generator basis. TOHPE runs unchanged on those coordinates; reduced
// coordinates map back to product Paulis whose mask is the generator XOR and
// whose sign is the exact Pauli-product sign from Stim. This stays entirely in
// the HIR (no diagonalizing Clifford is materialized, no qubits added).
// ---------------------------------------------------------------------------

// A full (x, z, sign) Pauli over the arena width.
struct FullPauli {
    std::vector<uint64_t> x;
    std::vector<uint64_t> z;
    bool sign = false;
};

// Overwrite an op's arena mask with a full (x, z, sign) Pauli.
void write_full_mask(HirModule& hir, const HeisenbergOp& op, const FullPauli& p) {
    auto m = hir.mask_at(op);
    auto x = m.x();
    auto z = m.z();
    for (uint32_t w = 0; w < x.num_words(); ++w) {
        x.words[w] = (w < p.x.size()) ? p.x[w] : 0;
        z.words[w] = (w < p.z.size()) ? p.z[w] : 0;
    }
    m.set_sign(p.sign);
}

// Emit a +1-coefficient T about the (signed) Pauli `p`. The sign is kept on the
// axis -- a signed Pauli generator-product has the eigenvalue (-1)^{coord.y}
// that the coordinate-space phase function certified, so stripping it would
// change the unitary.
void emit_t_full(HirModule& hir, HeisenbergOp& op, const FullPauli& p) {
    op.reset_to_tgate(false);
    write_full_mask(hir, op, p);
}

void emit_pr_full(HirModule& hir, HeisenbergOp& op, double alpha, const FullPauli& p) {
    op.reset_to_phase_rotation(alpha);
    write_full_mask(hir, op, p);
}

// Build a Stim PauliString from full-width (x, z) masks (sign +).
stim::PauliString<kStimWidth> make_ps(const std::vector<uint64_t>& x,
                                      const std::vector<uint64_t>& z, uint32_t n) {
    stim::PauliString<kStimWidth> p(n);
    const uint32_t words = (n + 63) / 64;
    for (uint32_t w = 0; w < words; ++w) {
        if (w < x.size())
            p.xs.u64[w] = x[w];
        if (w < z.size())
            p.zs.u64[w] = z[w];
    }
    return p;
}

// Product (as signed Hermitian Paulis) of the generators selected by the set
// bits of `coords`, returning the resulting full (x, z, sign). The generators
// pairwise commute, so the running product commutes with each next generator
// and the accumulated i-power is even; sign carries the leftover (-1).
FullPauli genprod(const std::vector<stim::PauliString<kStimWidth>>& gens,
                  const ParityColumn& coords, uint32_t n, uint32_t nwords) {
    stim::PauliString<kStimWidth> prod(n);
    // Stim's operator*= folds the (-1) part of the i-power into the sign and
    // asserts the i-power is even -- which holds because the generators (and
    // hence every partial product) pairwise commute.
    for (uint32_t i = 0; i < gens.size(); ++i)
        if (coords.get(i))
            prod.ref() *= gens[i].ref();
    FullPauli out;
    out.x.assign(nwords, 0);
    out.z.assign(nwords, 0);
    const uint32_t words = (n + 63) / 64;
    for (uint32_t w = 0; w < nwords && w < words; ++w) {
        out.x[w] = prod.xs.u64[w];
        out.z[w] = prod.zs.u64[w];
    }
    out.sign = prod.sign;
    return out;
}

// Phase B for mixed-type blocks. Same contract as apply_tohpe: returns true and
// rewrites the block's slots on a verified reduction, false to fall back.
bool apply_tohpe_mixed(HirModule& hir, const std::vector<FoldedGroup>& folded,
                       const std::vector<size_t>& block_slots, std::vector<uint8_t>& deleted,
                       size_t& tohpe_removed, size_t& t_after_counter) {
    const uint32_t nwords = hir.pauli_masks.num_words();
    const uint32_t n = nwords * 64;

    // Collect every nonzero-coefficient axis as a full (sign-normalized) Pauli.
    struct Term {
        FullPauli p;
        int c;
    };
    std::vector<Term> terms;
    for (const auto& fg : folded) {
        if (fg.c == 0)
            continue;
        const HeisenbergOp& rep = hir.ops[fg.slots.front()];
        FullPauli p;
        p.x.assign(nwords, 0);
        p.z.assign(nwords, 0);
        auto xm = hir.destab_mask(rep);
        auto zm = hir.stab_mask(rep);
        for (uint32_t w = 0; w < nwords; ++w) {
            p.x[w] = xm.words[w];
            p.z[w] = zm.words[w];
        }
        terms.push_back({p, fg.c});
    }
    if (terms.size() < 2)
        return false;

    // Build a GF(2) basis of the axes' symplectic (x || z) vectors. The
    // generators are the axes selected as pivots, in order. Crucially we track,
    // for each reduced echelon row, the coordinate vector that rebuilds it from
    // the ORIGINAL generators (not the eliminated rows), so genprod(coord) below
    // reproduces the actual Pauli mask. The rank is capped at 14, so the (at
    // most 14) generator indices fit in a single coordinate word.
    constexpr uint32_t kMaxRank = 14;
    const uint32_t two_n = 2 * n;
    auto sbit = [&](const FullPauli& p, uint32_t i) -> uint64_t {
        if (i < n)
            return (p.x[i / 64] >> (i % 64)) & 1ULL;
        uint32_t j = i - n;
        return (p.z[j / 64] >> (j % 64)) & 1ULL;
    };
    auto to_bits = [&](const FullPauli& p) {
        std::vector<uint8_t> b(two_n, 0);
        for (uint32_t i = 0; i < two_n; ++i)
            b[i] = static_cast<uint8_t>(sbit(p, i));
        return b;
    };
    std::vector<std::vector<uint8_t>> basis_red;  // reduced echelon rows
    std::vector<ParityColumn> basis_coord;        // each row in generator space
    std::vector<uint32_t> basis_pivot;
    std::vector<FullPauli> gen_full;  // generators (= pivot axes)

    // Reduce `v` against the current basis, accumulating its generator-space
    // coordinates into `coord` (which must start zeroed).
    auto reduce = [&](std::vector<uint8_t>& v, ParityColumn& coord) {
        for (size_t b = 0; b < basis_red.size(); ++b) {
            if (v[basis_pivot[b]]) {
                for (uint32_t i = 0; i < two_n; ++i)
                    v[i] ^= basis_red[b][i];
                coord.xor_with(basis_coord[b]);
            }
        }
    };

    for (const auto& t : terms) {
        std::vector<uint8_t> v = to_bits(t.p);
        ParityColumn coord{std::vector<uint64_t>(1, 0)};
        reduce(v, coord);
        uint32_t piv = two_n;
        for (uint32_t i = 0; i < two_n; ++i)
            if (v[i]) {
                piv = i;
                break;
            }
        if (piv != two_n) {
            const uint32_t g = static_cast<uint32_t>(gen_full.size());
            if (g >= kMaxRank)
                return false;    // rank too high; 2^rank verify table too large
            coord.set(g, true);  // v = generator_g (xor the earlier subtractions)
            basis_red.push_back(v);
            basis_coord.push_back(coord);
            basis_pivot.push_back(piv);
            gen_full.push_back(t.p);
        }
    }
    const uint32_t r = static_cast<uint32_t>(gen_full.size());
    if (r < 2)
        return false;

    std::vector<stim::PauliString<kStimWidth>> gens;
    for (const auto& g : gen_full)
        gens.push_back(make_ps(g.x, g.z, n));

    // Coordinates of every axis in generator space, with the rotation
    // coefficient re-expressed about the *signed* generator product
    // genprod(coord): if that product carries a (-1), the coefficient is negated
    // so a column always means "+1 T about genprod(coord)". Odd coefficients
    // become TOHPE columns; even ones are Clifford phases emitted directly.
    std::vector<ParityColumn> cols;
    std::vector<int> col_coeff;
    std::vector<std::pair<FullPauli, int>> evens;  // (signed Pauli, even coeff)
    for (const auto& t : terms) {
        std::vector<uint8_t> v = to_bits(t.p);
        ParityColumn coord{std::vector<uint64_t>(1, 0)};
        reduce(v, coord);
        FullPauli gp = genprod(gens, coord, n, nwords);
        int c = t.c % 8;
        if (gp.sign)
            c = (8 - c) % 8;  // P_k = -genprod(coord): negate the coefficient
        if (c % 2 == 1) {
            cols.push_back(coord);
            col_coeff.push_back(c);
        } else if (c != 0) {
            evens.push_back({gp, c});
        }
    }
    if (cols.size() < 2)
        return false;

    TohpeResult res = tohpe_reduce(cols, r);
    if (res.columns.size() >= cols.size())
        return false;

    // Odd-coeff remainders (c != 1) become Clifford phases on their own axis.
    std::vector<std::pair<ParityColumn, int>> col_rem;
    for (size_t i = 0; i < cols.size(); ++i)
        if (col_coeff[i] != 1)
            col_rem.push_back({cols[i], col_coeff[i] - 1});

    const size_t out_count =
        res.columns.size() + res.residuals.size() + evens.size() + col_rem.size();
    if (out_count > block_slots.size())
        return false;

    // Re-emit. Each reduced column is a +1 T about its signed generator product;
    // residuals and remainders are Clifford PHASE_ROTATIONs about theirs. The
    // Pauli sign is kept on the axis (it carries the eigenvalue the verified
    // coordinate-space phase function relies on).
    size_t si = 0;
    for (const auto& c : res.columns) {
        emit_t_full(hir, hir.ops[block_slots[si++]], genprod(gens, c, n, nwords));
        ++t_after_counter;
    }
    for (const auto& rphase : res.residuals)
        emit_pr_full(hir, hir.ops[block_slots[si++]], 0.25 * rphase.coeff_mod8,
                     genprod(gens, rphase.parity, n, nwords));
    for (const auto& [coord, coeff] : col_rem)
        emit_pr_full(hir, hir.ops[block_slots[si++]], 0.25 * coeff,
                     genprod(gens, coord, n, nwords));
    for (const auto& [pauli, coeff] : evens)
        emit_pr_full(hir, hir.ops[block_slots[si++]], 0.25 * coeff, pauli);
    while (si < block_slots.size())
        deleted[block_slots[si++]] = 1;

    tohpe_removed += cols.size() - res.columns.size();
    return true;
}

// Phase B: TOHPE multi-axis reduction on a single Z-only commuting block.
// `folded` carries the per-axis folded coefficients (Phase A). Returns true and
// rewrites the block's op slots if a genuine multi-axis reduction was found and
// verified; returns false to fall back to plain folding.
bool apply_tohpe(HirModule& hir, const std::vector<FoldedGroup>& folded,
                 const std::vector<size_t>& block_slots, std::vector<uint8_t>& deleted,
                 size_t& tohpe_removed, size_t& t_after_counter) {
    const uint32_t nwords = hir.pauli_masks.num_words();

    // The block must be a single Pauli type so its axes are simultaneously
    // diagonal as parities: all Z-type (X plane empty) or all X-type (Z plane
    // empty). Mixed/Y blocks need a diagonalizing Clifford and are skipped.
    bool all_z = true;
    bool all_x = true;
    for (const auto& fg : folded) {
        const HeisenbergOp& rep = hir.ops[fg.slots.front()];
        if (!hir.destab_mask(rep).is_zero())
            all_z = false;
        if (!hir.stab_mask(rep).is_zero())
            all_x = false;
    }
    if (!all_z && !all_x)
        return apply_tohpe_mixed(hir, folded, block_slots, deleted, tohpe_removed, t_after_counter);
    const bool is_x = all_x && !all_z;
    auto parity_mask = [&](const HeisenbergOp& op) {
        return is_x ? hir.destab_mask(op) : hir.stab_mask(op);
    };

    // Support (union of parity bits over all axes).
    std::vector<uint64_t> support_mask(nwords, 0);
    size_t odd_count = 0;
    for (const auto& fg : folded) {
        const HeisenbergOp& rep = hir.ops[fg.slots.front()];
        if (fg.c % 2 == 1)
            ++odd_count;
        auto pm = parity_mask(rep);
        for (uint32_t w = 0; w < nwords; ++w)
            support_mask[w] |= pm.words[w];
    }
    if (odd_count < 2)
        return false;  // nothing for the multi-axis reducer to do

    std::vector<uint32_t> support;
    for (uint32_t q = 0; q < nwords * 64; ++q)
        if ((support_mask[q / 64] >> (q % 64)) & 1ULL)
            support.push_back(q);
    if (support.size() > 14)
        return false;  // verification table 2^s would be too large
    const uint32_t s = static_cast<uint32_t>(support.size());

    // Build TOHPE columns (one per odd axis) and collect Clifford phases.
    struct CliffPhase {
        std::vector<uint64_t> zmask;
        int coeff;  // even
    };
    std::vector<ParityColumn> cols;
    std::vector<CliffPhase> cliff;
    for (const auto& fg : folded) {
        if (fg.c == 0)
            continue;
        const HeisenbergOp& rep = hir.ops[fg.slots.front()];
        std::vector<uint64_t> zfull(nwords, 0);
        auto z = parity_mask(rep);
        for (uint32_t w = 0; w < nwords; ++w)
            zfull[w] = z.words[w];

        if (fg.c % 2 == 1) {
            ParityColumn col{std::vector<uint64_t>(1, 0)};
            for (uint32_t k = 0; k < s; ++k) {
                uint32_t q = support[k];
                if ((zfull[q / 64] >> (q % 64)) & 1ULL)
                    col.set(k, true);
            }
            cols.push_back(std::move(col));
            if (fg.c != 1)
                cliff.push_back({zfull, fg.c - 1});  // T^{c} = T * (even Clifford)
        } else {
            cliff.push_back({zfull, fg.c});
        }
    }

    TohpeResult res = tohpe_reduce(cols, s);
    if (res.columns.size() >= cols.size())
        return false;  // TOHPE found no reduction; plain folding is enough

    const size_t out_count = res.columns.size() + res.residuals.size() + cliff.size();
    if (out_count > block_slots.size())
        return false;  // would need more ops than the block holds

    size_t si = 0;
    for (const auto& c : res.columns) {
        emit_t(hir, hir.ops[block_slots[si++]], col_to_zmask(c, support, nwords), is_x);
        ++t_after_counter;
    }
    for (const auto& r : res.residuals)
        emit_pr(hir, hir.ops[block_slots[si++]], 0.25 * r.coeff_mod8,
                col_to_zmask(r.parity, support, nwords), is_x);
    for (const auto& ph : cliff)
        emit_pr(hir, hir.ops[block_slots[si++]], 0.25 * ph.coeff, ph.zmask, is_x);
    while (si < block_slots.size())
        deleted[block_slots[si++]] = 1;

    tohpe_removed += cols.size() - res.columns.size();
    return true;
}

}  // namespace

void TCountPhasePolyPass::run(HirModule& hir) {
    blocks_ = 0;
    t_before_ = 0;
    t_after_ = 0;
    max_block_axes_ = 0;
    tohpe_blocks_ = 0;
    tohpe_removed_ = 0;

    const size_t n = hir.ops.size();
    const bool has_source_map = hir.source_map.size() == n;
    std::vector<uint8_t> deleted(n, 0);

    size_t i = 0;
    while (i < n) {
        if (hir.ops[i].op_type() != OpType::T_GATE) {
            ++i;
            continue;
        }

        // Grow a maximal commuting block of consecutive T_GATE ops: each new
        // op must commute with every op already in the block. Because the
        // members pairwise commute they are simultaneously diagonal, so any
        // reordering (and hence per-axis folding) is exact.
        size_t block_end = i + 1;
        while (block_end < n && hir.ops[block_end].op_type() == OpType::T_GATE) {
            bool commutes_with_all = true;
            for (size_t k = i; k < block_end; ++k) {
                if (!ops_commute(hir, hir.ops[block_end], hir.ops[k])) {
                    commutes_with_all = false;
                    break;
                }
            }
            if (!commutes_with_all)
                break;
            ++block_end;
        }

        const size_t block_size = block_end - i;
        if (block_size < 2) {
            i = block_end;
            continue;
        }

        // Normalize signs so coefficients depend only on the dagger flag.
        for (size_t k = i; k < block_end; ++k)
            normalize_t_sign(hir, hir.ops[k]);

        // Group block indices by virtual Pauli axis.
        std::vector<std::vector<size_t>> groups;
        for (size_t k = i; k < block_end; ++k) {
            bool placed = false;
            for (auto& g : groups) {
                if (same_axis(hir, hir.ops[k], hir.ops[g.front()])) {
                    g.push_back(k);
                    placed = true;
                    break;
                }
            }
            if (!placed)
                groups.push_back({k});
        }

        ++blocks_;
        t_before_ += block_size;
        if (groups.size() > max_block_axes_)
            max_block_axes_ = groups.size();

        // Phase A: fold each axis's coefficient in Z_8.
        std::vector<FoldedGroup> folded;
        folded.reserve(groups.size());
        for (auto& g : groups) {
            int c = 0;
            for (size_t idx : g)
                c += hir.ops[idx].is_dagger() ? -1 : 1;
            folded.push_back({g, ((c % 8) + 8) % 8});

            // Merge source lines onto the group's first slot for provenance.
            if (has_source_map) {
                auto& dst = hir.source_map[g[0]];
                for (size_t gi = 1; gi < g.size(); ++gi) {
                    auto& src = hir.source_map[g[gi]];
                    dst.insert(dst.end(), src.begin(), src.end());
                }
            }
        }

        std::vector<size_t> block_slots;
        block_slots.reserve(block_size);
        for (size_t k = i; k < block_end; ++k)
            block_slots.push_back(k);

        // Phase B: TOHPE multi-axis reduction on Z-only blocks; otherwise emit
        // the Phase A folding result (at most one T per surviving odd axis).
        if (enable_tohpe_ &&
            apply_tohpe(hir, folded, block_slots, deleted, tohpe_removed_, t_after_)) {
            ++tohpe_blocks_;
        } else {
            for (const auto& fg : folded) {
                std::vector<OutOp> out = synthesize(fg.c);
                for (size_t k = 0; k < fg.slots.size(); ++k) {
                    if (k < out.size()) {
                        if (out[k].is_t) {
                            hir.demote_to_tgate(hir.ops[fg.slots[k]], out[k].dagger);
                            ++t_after_;
                        } else {
                            hir.demote_to_phase_rotation(hir.ops[fg.slots[k]], out[k].alpha);
                        }
                    } else {
                        deleted[fg.slots[k]] = 1;
                    }
                }
            }
        }

        i = block_end;
    }

    // Compact deleted ops (erase-shift), mirroring PeepholeFusionPass.
    size_t write = 0;
    for (size_t read = 0; read < n; ++read) {
        if (!deleted[read]) {
            if (write != read) {
                hir.ops[write] = hir.ops[read];
                if (has_source_map)
                    hir.source_map[write] = std::move(hir.source_map[read]);
            }
            ++write;
        }
    }
    hir.ops.erase(hir.ops.begin() + static_cast<std::ptrdiff_t>(write), hir.ops.end());
    if (has_source_map)
        hir.source_map.resize(write);
}

}  // namespace clifft
