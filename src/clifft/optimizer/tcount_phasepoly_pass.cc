#include "clifft/optimizer/tcount_phasepoly_pass.h"

#include "clifft/optimizer/commutation.h"
#include "clifft/optimizer/tcount_tohpe.h"
#include "clifft/util/constants.h"

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

// Phase B: TOHPE multi-axis reduction on a single Z-only commuting block.
// `folded` carries the per-axis folded coefficients (Phase A). Returns true and
// rewrites the block's op slots if a genuine multi-axis reduction was found and
// verified; returns false to fall back to plain folding.
bool apply_tohpe(HirModule& hir, const std::vector<FoldedGroup>& folded,
                 const std::vector<size_t>& block_slots, std::vector<uint8_t>& deleted,
                 size_t& tohpe_removed, size_t& t_after_counter, uint32_t max_verify_bits) {
    const uint32_t nwords = hir.pauli_masks.num_words();

    // Reduce only single-Pauli-type blocks: all Z-type (X plane empty) or all
    // X-type (Z plane empty), so the axes are simultaneously diagonal as parities
    // and f(x) mod 8 is the exact unitary. Blocks that mix X and Z are left to
    // plain folding (see the evaluation doc on the mixed-type scope).
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
        return false;
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
    if (support.size() > max_verify_bits)
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

    TohpeResult res = tohpe_reduce(cols, s, 256, max_verify_bits);
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

        // Capture the block's source lines before re-synthesis. Phase B
        // reshuffles which slot carries which Pauli, so a reduced op derives
        // from the whole block rather than from its slot's original line; the
        // union is reattached to a surviving slot below.
        std::vector<uint32_t> block_src;
        if (has_source_map)
            for (size_t k = i; k < block_end; ++k)
                block_src.insert(block_src.end(), hir.source_map[k].begin(),
                                 hir.source_map[k].end());

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

        // Phase B: TOHPE multi-axis reduction (single- or mixed-type blocks);
        // otherwise emit the Phase A folding result (at most one T per surviving
        // odd axis).
        if (enable_tohpe_ && apply_tohpe(hir, folded, block_slots, deleted, tohpe_removed_,
                                         t_after_, max_verify_bits_)) {
            ++tohpe_blocks_;
            // Reattach the whole block's provenance to its first surviving slot.
            if (has_source_map)
                for (size_t slot : block_slots)
                    if (!deleted[slot]) {
                        hir.source_map[slot] = block_src;
                        break;
                    }
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
