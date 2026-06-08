#pragma once

#include "clifft/optimizer/hir_pass.h"

#include <cstddef>

namespace clifft {

/// Experimental, opt-in HIR pass exploring global T-count reduction via a
/// phase-polynomial view of the Heisenberg IR.
///
/// Background: a T gate in the HIR is a pi/8 rotation about a virtual Pauli
/// axis (the Cliffords are already absorbed into the offline frame U_C). A
/// maximal run of consecutive, pairwise-commuting T_GATE ops is therefore a
/// set of simultaneously-diagonalizable rotations -- i.e. a phase polynomial
/// whose gate-synthesis matrix is read directly off the T axes' (X, Z) masks,
/// with no need to track CNOTs.
///
/// Within such a "commuting block" this pass folds the per-axis rotation
/// coefficients in Z_8 (T = +1, T_dag = -1, T^8 = I, T^2 = S, T^4 = Z) and
/// re-emits the minimal representation: at most one T_GATE per surviving
/// odd-coefficient axis, with any even (Clifford) remainder re-emitted as a
/// PHASE_ROTATION that the existing peephole/back-end then absorbs.
///
/// Scope and findings (see docs/reference/passes.md):
///   - This is provably T-count-optimal *within a commuting block* for the
///     class of "products of Pauli rotations" that the HIR can express: the
///     Pauli-rotation characters are orthogonal, so distinct axes cannot
///     cancel. It reproduces, through an explicit phase-polynomial matrix,
///     the same per-block reduction that PeepholeFusionPass already achieves.
///   - Genuine *multi-axis* T-count reduction (TODD / Reed-Muller / ZX) needs
///     ancilla qubits or Hadamard gadgets, i.e. structure outside the
///     commuting-Pauli-rotation class. Emitting that structure would require
///     changing the HIR/VM, which this issue places out of scope. The
///     analyzer metrics below quantify how much multi-axis opportunity exists.
///
/// The pass is registered with default_enabled = false and is intended for
/// evaluation, not the default pipeline. It is exactly semantics-preserving.
class TCountPhasePolyPass : public HirPass {
  public:
    /// `enable_tohpe` toggles Phase B (the multi-axis TOHPE reducer). With it
    /// off the pass performs only Phase A folding, which is how the evaluation
    /// isolates each phase's contribution.
    explicit TCountPhasePolyPass(bool enable_tohpe = true) : enable_tohpe_(enable_tohpe) {}

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
    size_t blocks_ = 0;
    size_t t_before_ = 0;
    size_t t_after_ = 0;
    size_t max_block_axes_ = 0;
    size_t tohpe_blocks_ = 0;
    size_t tohpe_removed_ = 0;
};

}  // namespace clifft
