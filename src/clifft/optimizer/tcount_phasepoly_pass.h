#pragma once

#include "clifft/optimizer/hir_pass.h"

#include <cstddef>

namespace clifft {

/// Experimental, opt-in HIR pass exploring global T-count reduction via a
/// phase-polynomial view of the Heisenberg IR.
///
/// A T gate in the HIR is a pi/8 rotation about a virtual Pauli axis (the
/// Cliffords are already absorbed into the offline frame U_C). A maximal run of
/// pairwise-commuting T_GATE ops is a phase polynomial whose gate-synthesis
/// matrix is read directly off the T axes' (X, Z) masks. Within such a block the
/// pass runs two phases (theory in docs/theory/tcount.md, evaluation in
/// tcount_evaluation.md):
///
///   - Phase A (folding): adds the per-axis coefficients in Z_8 and keeps at
///     most one T per surviving odd-coefficient axis. PeepholeFusionPass already
///     reaches this same per-block optimum.
///   - Phase B (TOHPE, Vandaele arXiv:2407.08695): the multi-axis reducer that
///     can drop below the distinct-axis count using the block's cubic structure.
///     It runs only on single-Pauli-type blocks (all-Z or all-X), where the
///     diagonal phase function f(x) mod 8 is the exact unitary, and accepts a
///     duplicate-and-destroy move only if it preserves that f exactly. That is a
///     subset of the moves Algorithm 2's Theorem 1 permits (Theorem 1 allows a
///     Clifford correction, including quadratic terms, that this check rejects),
///     traded for an exactly-verified result. Mixed-X/Z blocks are left to
///     Phase A.
///
/// Clifford remainders are re-emitted as single-axis PHASE_ROTATION ops that a
/// following PeepholeFusionPass absorbs into the frame.
///
/// Registered with default_enabled = false; opt-in, and exactly
/// semantics-preserving on the blocks it reduces.
class TCountPhasePolyPass : public HirPass {
  public:
    /// `enable_tohpe` toggles Phase B (the multi-axis TOHPE reducer). With it
    /// off the pass performs only Phase A folding, which is how the evaluation
    /// isolates each phase's contribution. `max_verify_bits` caps the width of a
    /// block Phase B will reduce: each accepted move is checked against the exact
    /// 2^width phase function, so this bounds that check's cost (default 14 ->
    /// 16 KB table). Wider blocks are left to Phase A folding.
    explicit TCountPhasePolyPass(bool enable_tohpe = true, uint32_t max_verify_bits = 14)
        : enable_tohpe_(enable_tohpe), max_verify_bits_(max_verify_bits) {}

    void run(HirModule& hir) override;

    /// Statistics from the last run (reset on each run()).

    /// Number of maximal commuting T_GATE blocks of size >= 2 processed.
    size_t blocks() const { return blocks_; }

    /// Total T_GATE ops inside processed blocks, before folding.
    size_t t_before() const { return t_before_; }

    /// Total T_GATE ops emitted from processed blocks, after folding.
    size_t t_after() const { return t_after_; }

    /// Net T_GATE ops removed: t_before() - t_after().
    size_t t_removed() const { return t_before_ - t_after_; }

    /// Largest number of distinct Pauli axes (phase-polynomial columns) seen
    /// in any single block. A block with more distinct axes than its
    /// post-fold T-count is where a multi-axis (TODD) technique could, in
    /// principle, reduce further -- the analyzer signal for productionization.
    size_t max_block_axes() const { return max_block_axes_; }

    /// Number of commuting blocks where the Phase B (TOHPE) multi-axis reducer
    /// found and applied a genuine reduction beyond same-axis folding.
    size_t tohpe_blocks() const { return tohpe_blocks_; }

    /// T gates removed by Phase B (TOHPE) *beyond* what Phase A folding
    /// achieves -- the isolated marginal contribution of the multi-axis pass.
    size_t tohpe_removed() const { return tohpe_removed_; }

  private:
    bool enable_tohpe_ = true;
    uint32_t max_verify_bits_ = 14;
    size_t blocks_ = 0;
    size_t t_before_ = 0;
    size_t t_after_ = 0;
    size_t max_block_axes_ = 0;
    size_t tohpe_blocks_ = 0;
    size_t tohpe_removed_ = 0;
};

}  // namespace clifft
