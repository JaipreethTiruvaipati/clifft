# Evaluation: HIR-level global T-count reduction

This is the scientific evaluation of the experimental `TCountPhasePolyPass`
(issue #40). The theory and citations are in [tcount.md](tcount.md). All numbers
below are produced by `tools/bench/tcount/bench_tcount.cc` (a standalone harness
linked against `clifft_core`) and are reproducible with:

```
cmake --build build --target bench_tcount && ./build/tests/bench_tcount
```

## Method

For each circuit we trace to HIR and measure the T-count under four
configurations, isolating each optimization phase:

* **no-opt** -- front-end trace only.
* **peephole** -- `PeepholeFusionPass` (Clifft's existing local optimizer).
* **+foldA** -- peephole, then `TCountPhasePolyPass(enable_tohpe=false)`
  (Phase A: phase folding, Amy-Maslov-Mosca arXiv:1303.2042).
* **+TOHPE** -- peephole, then the full pass (Phase B: TOHPE multi-axis
  reduction, Vandaele arXiv:2407.08695).

Circuits are Clifford+T text. The parser has no CCX/CCZ, so Toffoli and CCZ are
hand-decomposed; CCZ stays Z-diagonal (its 7-term phase polynomial), while a
Toffoli's internal Hadamards make its block mixed-type after Clifford
absorption. `cultivation_d5` is the real distance-5 magic-state cultivation
fixture shipped with the repo. Equivalence is checked exactly by dense
statevector (n <= 10); diagonal blocks are additionally verified by an exhaustive
`f(x) mod 8` check in the TOHPE core and the end-to-end tests.

## Results: per-phase T-count (ancilla-free)

| circuit | n | no-opt | peephole | +foldA | +TOHPE | TOHPE removed | equiv |
|---|--:|--:|--:|--:|--:|--:|:-:|
| ccz_single | 3 | 7 | 7 | 7 | 7 | 0 | OK |
| ccz_ladder_2 | 4 | 14 | 8 | 8 | 8 | 0 | OK |
| ccz_ladder_3 | 5 | 21 | 11 | 11 | 11 | 0 | OK |
| ccz_ladder_4 | 6 | 28 | 14 | 14 | 14 | 0 | OK |
| ccz_ladder_6 | 8 | 42 | 20 | 20 | 20 | 0 | OK |
| **s_empty_4** | 4 | 15 | 15 | 15 | **0** | **15** | OK |
| **s_empty_4_minus_full** | 4 | 14 | 14 | 14 | **1** | **13** | OK |
| toffoli_single | 3 | 7 | 7 | 7 | 7 | 0 | OK |
| toffoli_chain_3 | 5 | 21 | 17 | 17 | 17 | 0 | OK |
| random_6q_d120 | 6 | 34 | 14 | 14 | 14 | 0 | OK |
| random_8q_d200 | 8 | 50 | 22 | 22 | 22 | 0 | OK |
| **cultivation_d5** (real) | 26 | 72 | 72 | 72 | 72 | 0 | n/a |

## Results: commuting-block structure after peephole

This explains *where the multi-axis reducer can fire*. A block must be a single
Pauli type (Z-only or X-only) for Phase B to act, and must carry residual cubic
redundancy for it to find anything.

| circuit | blocks (>=2) | single-type | mixed-type | largest |
|---|--:|--:|--:|--:|
| ccz_ladder_6 | 1 | 1 | 0 | 20 |
| s_empty_4 | 1 | 1 | 0 | 15 |
| toffoli_single | 1 | 0 | 1 | 7 |
| toffoli_chain_3 | 3 | 0 | 3 | 7 |
| random_8q_d200 | 5 | 0 | 5 | 5 |
| cultivation_d5 (real) | 5 | 5 | 0 | 19 |

## Analysis

**Phase A folding contributes nothing beyond peephole.** In every row
`+foldA == peephole`. This empirically confirms the theory: `PeepholeFusionPass`
already reaches the per-block same-axis folding optimum (see also
`test_tcount_phasepoly.py::TestRedundantWithPeephole`). Phase A exists only to
build the gate-synthesis matrix and to make the split measurable; it is not a
source of reduction.

**Phase B (TOHPE) is correct and genuinely multi-axis, but only fires on
redundant phase polynomials.** It is exact (every reduction is verified against
the full `f(x) mod 8`, and rejected otherwise) and it removes T gates that
folding provably cannot: `s_empty_4` collapses 15 -> 0 and `s_empty_4_minus_full`
collapses 14 -> 1 (the optimal single `T_dag` on Z0^Z1^Z2^Z3). These are the
Amy-Maslov-Mosca trivial-polynomial constructions -- maximally redundant by
design.

**On every realistic circuit TOHPE removes zero T gates beyond peephole**, for
two distinct, measured reasons:

1. *Hadamard-bearing structure becomes mixed-type.* A Toffoli is
   `H; CCZ; H`; after the front end absorbs the Hadamards into `U_C`, the block's
   T axes are no longer a single Pauli type (`toffoli_*`, `random_*` rows show
   0 single-type blocks). The single-type reducer skips them. This is Clifft's
   form of the well-known phase-polynomial Achilles heel (Hadamards); in the
   op-T-mize literature it is handled by *Hadamard gadgetization with ancillas*,
   which Clifft's fixed `2^{k_max}` allocation and fixed Pauli arena forbid.
2. *Single-type blocks are already at their folding optimum.* The CCZ ladders and
   the real `cultivation_d5` circuit do form large single-type blocks (up to
   size 19-20), but once same-parity folding is applied there is no residual
   cubic redundancy for TOHPE to exploit, so it returns them unchanged.

**Calibration against the literature.** This is consistent with the ancilla-free
benchmark numbers of Vandaele 2024 (Table 2), where even a full FastTODD leaves
the T-count of many standard circuits unchanged, precisely because the large
op-T-mize reductions come from Hadamard gadgetization (ancillas). Clifft sits
firmly in the ancilla-free regime.

## Conclusion: is this worth productionizing?

**Not as-is.** On Clifft's realistic near-Clifford workloads -- including the
real magic-state cultivation circuit and random Clifford+T circuits -- HIR-level
phase-polynomial T-count reduction yields **no improvement over the existing
`PeepholeFusionPass`**. The multi-axis TOHPE reducer is correct and provably
removes T gates that folding cannot, but the circuits where it helps
(`S_empty`-style redundant phase polynomials) are not representative of what
Clifft simulates.

The evaluation does, however, sharpen *what would have to change* for the
direction to pay off, and these are the recommended follow-ups:

1. **Per-block diagonalization.** Synthesize a Clifford that maps a mixed-type
   commuting block to single-type before reduction, then re-absorb it through the
   frame. This would unlock the Toffoli/arithmetic regime (`toffoli_*`) where the
   literature shows the real reductions live. It is the natural next prototype
   and stays within the HIR.
2. **FastTODD (Vandaele Theorem 6)** in place of the simpler TOHPE (Theorem 1)
   used here, which can find marginally more reductions on single-type blocks.
3. **The hard ceiling is ancillas.** The headline op-T-mize gains (GF(2^m)
   multipliers, adders) require Hadamard gadgetization with ancilla qubits.
   Delivering those in Clifft would mean relaxing the single-allocation /
   fixed-arena invariants -- a VM-level change explicitly out of scope for this
   issue, and a separate design discussion.

In short: the substrate is real and the reducer is verified, but the ancilla-free
HIR regime that Clifft is constrained to does not, on representative circuits,
offer T-count reduction beyond what Clifft already does. We recommend keeping the
pass opt-in and experimental rather than adding it to the default pipeline.
