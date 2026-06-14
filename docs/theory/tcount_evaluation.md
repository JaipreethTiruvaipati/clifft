# Evaluation: HIR-level global T-count reduction

This page evaluates the experimental `TCountPhasePolyPass` (issue #40). The
theory and citations are in [tcount.md](tcount.md). All numbers below come from
`tools/bench/tcount/bench_tcount.cc`, a standalone harness linked against
`clifft_core`, and are reproducible with:

```text
cmake --build build --target bench_tcount && ./build/tests/bench_tcount
```

## Method

For each circuit we trace to HIR and measure the T-count under four
configurations, isolating each optimization phase:

* `no-opt`: front-end trace only.
* `peephole`: `PeepholeFusionPass` (Clifft's existing local optimizer).
* `+foldA`: peephole, then `TCountPhasePolyPass(enable_tohpe=false)` (Phase A
  folding, Amy-Maslov-Mosca arXiv:1303.2042).
* `+TOHPE`: peephole, then the full pass (Phase B TOHPE multi-axis reduction,
  Vandaele arXiv:2407.08695).

Circuits are Clifford+T text. The parser has no CCX/CCZ, so Toffoli and CCZ are
hand-decomposed; CCZ stays Z-diagonal (its 7-term phase polynomial), while a
Toffoli's internal Hadamards make its block mixed-type after Clifford absorption.
`cultivation_d5` is the real distance-5 magic-state cultivation fixture shipped
with the repo. The bench checks equivalence by comparing the peephole'd circuit
against the peephole + pass circuit amplitude by amplitude, including global
phase, for n <= 10 (it compares against peephole rather than the raw trace so it
isolates this pass and not peephole's own up-to-global-phase behaviour). Diagonal
blocks are additionally verified by an exhaustive `f(x) mod 8` check in the TOHPE
core and the end-to-end tests.

## Results: per-phase T-count (ancilla-free)

Bold rows are where Phase B (TOHPE) removes T gates beyond peephole. The table
is `bench_tcount` output with the per-block-count column dropped for width; the
T-counts are deterministic, the `time_ms` column is a single-run snapshot of the
Phase B call and is machine-dependent (8-core box; the first Phase B call also
pays one-time OpenMP thread-pool startup).

| circuit | n | no-opt | peephole | +foldA | +TOHPE | removed | time_ms | equiv |
| --- | --: | --: | --: | --: | --: | --: | --: | :-: |
| ccz_single | 3 | 7 | 7 | 7 | 7 | 0 | 0.3 | OK |
| ccz_ladder_2 | 4 | 14 | 8 | 8 | 8 | 0 | 0.3 | OK |
| ccz_ladder_3 | 5 | 21 | 11 | 11 | 11 | 0 | 0.5 | OK |
| ccz_ladder_4 | 6 | 28 | 14 | 14 | 14 | 0 | 2.0 | OK |
| ccz_ladder_6 | 8 | 42 | 20 | 20 | 20 | 0 | 1.5 | OK |
| ccz_ladder_10 | 12 | 70 | 32 | 32 | 32 | 0 | 11.9 | n/a |
| **ccz_complete_4** | 4 | 28 | 8 | 8 | **7** | **1** | 8.2 | OK |
| ccz_complete_5 | 5 | 70 | 20 | 20 | 20 | 0 | 10.3 | OK |
| **ccz_complete_6** | 6 | 140 | 20 | 20 | **12** | **8** | 15.0 | OK |
| **ccz_complete_7** | 7 | 245 | 63 | 63 | **21** | **42** | 34.8 | OK |
| **ccz_complete_8** | 8 | 392 | 64 | 64 | **22** | **42** | 54.5 | OK |
| **ccz_complete_9** | 9 | 588 | 120 | 120 | **44** | **76** | 216.8 | OK |
| **ccz_complete_10** | 10 | 840 | 120 | 120 | **36** | **84** | 541.7 | OK |
| **ccz_complete_11** | 11 | 1155 | 231 | 231 | **85** | **146** | 3705.4 | n/a |
| **ccz_complete_12** | 12 | 1540 | 232 | 232 | **80** | **152** | 7861.0 | n/a |
| **ccz_complete_13** | 13 | 2002 | 364 | 364 | **150** | **214** | 78948.0 | n/a |
| **ccz_complete_14** | 14 | 2548 | 364 | 364 | **216** | **148** | 81014.7 | n/a |
| ccz_star_5 | 7 | 35 | 23 | 23 | 23 | 0 | 1.6 | OK |
| ccz_star_8 | 10 | 56 | 32 | 32 | 32 | 0 | 36.5 | OK |
| s_empty_4_bellmixed | 4 | 15 | 15 | 15 | 15 | 0 | 0.3 | OK |
| **s_empty_4** | 4 | 15 | 15 | 15 | **0** | **15** | 1.5 | OK |
| **s_empty_5** | 5 | 31 | 31 | 31 | **0** | **31** | 1.4 | OK |
| **s_empty_4_minus_full** | 4 | 14 | 14 | 14 | **1** | **13** | 0.4 | OK |
| toffoli_single | 3 | 7 | 7 | 7 | 7 | 0 | 0.1 | OK |
| toffoli_chain_3 | 5 | 21 | 17 | 17 | 17 | 0 | 0.4 | OK |
| tof_ladder_3 | 5 | 14 | 14 | 14 | 14 | 0 | 0.2 | OK |
| tof_ladder_4 | 7 | 21 | 21 | 21 | 21 | 0 | 0.3 | OK |
| tof_ladder_5 | 9 | 28 | 28 | 28 | 28 | 0 | 0.4 | OK |
| mcx_3 | 5 | 28 | 8 | 8 | 8 | 0 | 0.5 | OK |
| mcx_4 | 7 | 42 | 16 | 16 | 16 | 0 | 0.8 | OK |
| random_6q_d120 | 6 | 34 | 14 | 14 | 14 | 0 | 5.7 | OK |
| random_8q_d200 | 8 | 50 | 22 | 22 | 22 | 0 | 1.8 | OK |
| cultivation_d5 (real) | 26 | 72 | 72 | 72 | 72 | 0 | 32.7 | n/a |

`ccz_complete_k` is all $\binom{k}{3}$ CCZ gates on $k$ qubits, a dense diagonal
phase polynomial, and the family that stresses the reducer hardest. The whole
range reduces: 6 drops 20 -> 12, 7 drops 63 -> 21, 8 drops 64 -> 22, 9 drops
120 -> 44, 10 drops 120 -> 36, 11 drops 231 -> 85, 12 drops 232 -> 80, 13 drops
364 -> 150, and 14 drops 364 -> 216. ($k = 14$, $n = 14$, is the top of the
exact-$f$ verifiable range; wider $n$ is returned unchanged.) Each is verified
exact against the full diagonal $f(x)\bmod 8$ (and against the dense statevector
for $n \le 10$, the `OK` column).

### Performance

The search is Algorithm 2's $S(z)$ scoring (theory doc): per null vector, a hash
map scores every candidate $z$ in $O(m^2)$ by how many columns it duplicates, the
scoring is split across cores, and the exact $f$ check fires once on the chosen
move. This holds up as blocks widen. In `time_ms`, `ccz_complete_7` (63 columns)
reduces 63 -> 21 in ~35 ms (it was ~20 s under the earlier per-trial-properize
scan), and the previously infeasible wide blocks now finish: `ccz_complete_9/10`
(120 columns) in ~0.2-0.5 s, `ccz_complete_11` (231 columns) in ~3.7 s,
`ccz_complete_12` (232 columns) in ~8 s, and the widest verifiable blocks
`ccz_complete_13/14` (364 columns) in ~80 s, reducing 364 -> 150 and 364 -> 216.
The localized blocks the front end actually emits are sub-millisecond to a few
milliseconds. The `max_cols` width cap is 384, raised from the slow scan's
conservative 256 now that the search is fast enough to cover the whole exact-$f$
verifiable range ($n \le 14$, i.e. up to the 364-wide `ccz_complete_14` in
~80 s); blocks past the cap are returned unchanged.

`s_empty_4_bellmixed` is the same inner `s_empty(4)` wrapped in an entangling
Clifford (`CX 0 1; H 0; ... ; H 0; CX 0 1`). That makes the commuting block
mixed-type, so Phase B leaves it untouched (`removed = 0`) and the unitary is
preserved exactly. It is the regression for the mixed-type scope: an earlier
version reduced such a block but dropped a global phase, which the exact-amplitude
`equiv` check (not fidelity) catches.

## Results: commuting-block structure after peephole

This shows where the multi-axis reducer can fire. A block is a maximal run of
consecutive, pairwise-commuting `T_GATE` ops. It is single-type when every axis
lies in one Pauli plane (all-Z or all-X, that is, all `x` masks zero or all `z`
masks zero), so it is simultaneously diagonal as binary parities; it is
mixed-type when axes mix X and Z (for example some `Y`), so it is not diagonal in
the computational basis without a further Clifford. Phase B reduces single-type
blocks only; mixed-type blocks are left to Phase A folding (see the theory doc on
why the exact check needs the single-type case). The column shows how often each
arises.

| circuit | blocks (>=2) | single-type | mixed-type | largest |
| --- | --: | --: | --: | --: |
| ccz_complete_6 | 1 | 1 | 0 | 20 |
| ccz_complete_7 | 1 | 1 | 0 | 63 |
| ccz_complete_8 | 1 | 1 | 0 | 64 |
| ccz_complete_10 | 1 | 1 | 0 | 120 |
| ccz_complete_12 | 1 | 1 | 0 | 232 |
| ccz_complete_14 | 1 | 1 | 0 | 364 |
| s_empty_4_bellmixed | 1 | 0 | 1 | 15 |
| ccz_ladder_6 | 1 | 1 | 0 | 20 |
| toffoli_single | 1 | 0 | 1 | 7 |
| toffoli_chain_3 | 3 | 0 | 3 | 7 |
| random_8q_d200 | 5 | 0 | 5 | 5 |
| cultivation_d5 (real) | 5 | 5 | 0 | 19 |

## Analysis

Phase A folding contributes nothing beyond peephole. In every row `+foldA ==
peephole`, which confirms the theory: `PeepholeFusionPass` already reaches the
per-block same-axis folding optimum (see
`test_tcount_phasepoly.py::TestRedundantWithPeephole`). Phase A is there only to
build the gate-synthesis matrix and to make the per-phase split measurable; it is
not itself a source of reduction.

Phase B (TOHPE) reduces ancilla-free T-count where the block carries cubic
redundancy. It removes T gates folding cannot: `s_empty_4/5` collapse 15/31 -> 0
(Amy-Maslov-Mosca trivial polynomials), `s_empty_4_minus_full` 14 -> 1 (one
`T_dag`), and the dense diagonal `ccz_complete_k` family drops all the way from
`k = 6` (20 -> 12) up to `k = 14` (364 -> 216). Every accepted move is checked
against the full `f(x) mod 8`, so these are exact.

TOHPE is selective, and the block-structure table says why. It fires on dense
shared cubic structure (`ccz_complete_*`) but not on sparse structure:
`ccz_ladder_*` and `ccz_star_*` are single-type but, once same-parity folding is
applied, carry no residual cubic redundancy, so TOHPE returns them unchanged
(`removed = 0`). The reducer follows Vandaele Algorithm 2: for each null-space
vector it scores every candidate update `z` by how many columns the move would
turn into duplicates (the `S(z)` hash step) and keeps that vector's best-scoring
`z`, then applies the best move overall -- but only if it preserves the exact
diagonal `f(x) mod 8`. That f-check is stricter than Theorem 1, which guarantees
the signature tensor only up to a Clifford correction that may include quadratic
terms; the check rejects any move that would need such a correction, trading some
reductions for a result that is exact with no extra bookkeeping. The `0` on the
sparse circuits is a property of those polynomials -- no residual cubic redundancy
once same-parity folding is applied -- not a search artifact.

Hadamard-bearing circuits become mixed-type, and Phase B leaves them to folding.
A Toffoli is `H; CCZ; H`; once the front end absorbs the Hadamards into `U_C`, the
block's axes mix Pauli planes (`toffoli_*`, `random_*`, `s_empty_4_bellmixed` show
0 single-type blocks), so Phase B does not act on them (the single-type scope
above). Even if it did, a single Toffoli is one CCZ, already T-optimal (7 T on
three qubits, Amy-Maslov-Mosca), and `toffoli_chain_3` is three such independent
blocks with no shared cubic structure, so `removed = 0` either way.

Real Toffoli-based circuits behave the same way. `tof_ladder_k` (the tof_n
compute ladder) and `mcx_k` (compute, Z, uncompute) are Clifford+T members of the
op-T-mize / Amy arithmetic family, and the pass removes nothing beyond peephole on
any of them. The reason is structural: each Toffoli is an independent, already
T-optimal CCZ once its Hadamards are absorbed, and on `mcx` the compute/uncompute
T gates that do cancel are cancelled by peephole, not by Phase B (`mcx_3`:
28 -> 8 at peephole, then 8 with TOHPE). This is the same boundary the rest of the
field reports: ancilla-free, arithmetic and reversible-logic circuits give no
phase-polynomial T reduction, because their reductions come from gadgetization
with ancillas. The pass earns its keep on diagonal phase-polynomial structure
(`ccz_complete`), not on arithmetic.

This matches Vandaele 2024, Table 2 (ancilla-free): structured diagonal circuits
reduce, while many standard benchmarks see little ancilla-free reduction because
their headline op-T-mize gains come from Hadamard gadgetization with ancillas.
Clifft sits in the ancilla-free regime, so the dense-diagonal cases
(`ccz_complete`) are where ancilla-free TOHPE is expected to help.

For prior implementations: the published TOHPE/FastTODD reference is Vandaele's
Rust tool (`VivienVandaele/quantum-circuit-optimization`); TODD has
Heyfron-Campbell's C++ `TOpt`, and phase folding has Amy's `feynman`/`t-par`.
Those operate on `{CNOT, T}` QASM circuits and pay an explicit Hadamard/ancilla
overhead. This PR is a from-scratch in-HIR implementation that consumes the
already-Clifford-absorbed virtual Pauli axes directly (no CNOT tracking, no
ancillas) and adds the exact-`f` verification gate, which those tools do not need
because they re-synthesise a full circuit.

## Conclusion: is this worth productionizing?

Conditionally yes, for single-type diagonal phase-polynomial structure. Phase B
reduces ancilla-free T-count on dense diagonal blocks (`ccz_complete_6`: 20 -> 12,
40% beyond peephole; `s_empty`: 15 -> 0), the family that shows up in IQP
sampling, Hamming-weight phasing, and diagonal phase oracles. On sparse structure,
on random circuits, on Toffoli-based arithmetic, and on the real `cultivation_d5`
circuit it matches peephole, so it should stay opt-in rather than default: a user
targeting diagonal-heavy circuits can enable it for a reduction, while a user on
generic near-Clifford workloads pays nothing by leaving it off.

Scope is single-Pauli-type blocks, where `f(x) mod 8` is the exact unitary and a
move can be accepted with an exact check. Mixed-X/Z blocks (Hadamard-absorbed
Toffolis) are left to folding; reducing them needs the quadratic-Clifford and
global-phase tracking discussed in the theory doc, which is future work. The
remaining follow-ups are:

1. FastTODD (Vandaele Theorem 6) in place of TOHPE (Theorem 1), which can find
   marginally more reductions per block.
2. Ancilla-based gadgetization. The largest op-T-mize gains (GF(2^m) multipliers,
   adders) come from Hadamard gadgetization, which adds qubits and mid-circuit
   measurement. This is a circuit/HIR-level transformation that runs before
   lowering, so it does not require a VM change: a circuit with more qubits
   lowers to a different bytecode trace with a larger `k_max`, and the VM
   allocates accordingly. The cost is in the resulting trace (more active
   amplitudes), not in the VM. It is the highest-impact follow-up but the largest
   in scope, so it is left out of this PR.

A note on the FTCircuitBench suite suggested in review: its inputs are
arbitrary-angle (continuous `rz`/`cu1`) circuits intended to be fed through
Gridsynth, and its pre-synthesised Clifford+T outputs are Gridsynth-exploded (for
example `qft_4q` reaches about 115k T gates from the approximation, which is
precision, not algorithmic structure), so they are not a useful target for a
phase-polynomial T-reducer without first standing up the full synthesis pipeline.

On scope: the table above already includes hand-built members of the op-T-mize /
Amy arithmetic family (`tof_ladder_k`, `mcx_k`), and the result there is a clean
zero beyond peephole, as discussed in the analysis. The remaining members
(GF(2^m)-mult, the larger ripple adders, `barenco_tof`) are bigger and tedious to
transcribe by hand; reading them directly needs a `.qc`/QASM to Stim importer,
which is the natural next benchmarking step. The expectation from both the
literature and the `tof`/`mcx` rows here is more of the same: ancilla-free, these
arithmetic circuits do not yield phase-polynomial T reductions.
