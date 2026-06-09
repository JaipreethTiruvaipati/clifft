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

Bold rows are where Phase B (TOHPE) removes T gates **beyond** peephole.

| circuit | n | no-opt | peephole | +foldA | +TOHPE | TOHPE removed | equiv |
|---|--:|--:|--:|--:|--:|--:|:-:|
| ccz_single | 3 | 7 | 7 | 7 | 7 | 0 | OK |
| ccz_ladder_2 | 4 | 14 | 8 | 8 | 8 | 0 | OK |
| ccz_ladder_6 | 8 | 42 | 20 | 20 | 20 | 0 | OK |
| ccz_ladder_10 | 12 | 70 | 32 | 32 | 32 | 0 | n/a |
| **ccz_complete_4** | 4 | 28 | 8 | 8 | **7** | **1** | OK |
| ccz_complete_5 | 5 | 70 | 20 | 20 | 20 | 0 | OK |
| **ccz_complete_6** | 6 | 140 | 20 | 20 | **12** | **8** | OK |
| ccz_star_5 | 7 | 35 | 23 | 23 | 23 | 0 | OK |
| ccz_star_8 | 10 | 56 | 32 | 32 | 32 | 0 | OK |
| **s_empty_4** | 4 | 15 | 15 | 15 | **0** | **15** | OK |
| **s_empty_5** | 5 | 31 | 31 | 31 | **0** | **31** | OK |
| **s_empty_4_minus_full** | 4 | 14 | 14 | 14 | **1** | **13** | OK |
| **ccz_complete_6_hmixed** | 6 | 140 | 20 | 20 | **12** | **8** | OK |
| toffoli_single | 3 | 7 | 7 | 7 | 7 | 0 | OK |
| toffoli_chain_3 | 5 | 21 | 17 | 17 | 17 | 0 | OK |
| random_6q_d120 | 6 | 34 | 14 | 14 | 14 | 0 | OK |
| random_8q_d200 | 8 | 50 | 22 | 22 | 22 | 0 | OK |
| **cultivation_d5** (real) | 26 | 72 | 72 | 72 | 72 | 0 | n/a |

`ccz_complete_k` is all $\binom{k}{3}$ CCZ gates on $k$ qubits -- a dense
diagonal phase polynomial. On `ccz_complete_6` TOHPE removes **8 of 20** T gates
(40%) that peephole could not, and the reduction is verified exact (the test
`PhasePoly TOHPE: dense CCZ-complete block reduces beyond peephole, exact`
checks the full diagonal $f(x)\bmod 8$ is preserved).

`ccz_complete_6_hmixed` is the same circuit conjugated by Hadamards on three
qubits: this rotates parities into the X plane so the commuting block is
**mixed-type** (not all-Z), yet the mixed-type path still removes the same 8 T
gates. Because the conjugated unitary is genuinely non-diagonal, `equiv = OK`
here is a full statevector check (amplitudes and global phase), not just a
diagonal $f(x)$ check -- the strongest correctness evidence in the table.

## Results: commuting-block structure after peephole

This explains *where the multi-axis reducer can fire*. A "block" is a maximal run
of consecutive, pairwise-commuting `T_GATE` ops. A block is **single-type** when
every axis lies in one Pauli plane (all-Z or all-X, i.e. all `x` masks zero or
all `z` masks zero) and is therefore simultaneously diagonal as binary parities;
it is **mixed-type** when axes mix X and Z (e.g. some `Y`), so it is not diagonal
in any computational-basis sense without a further Clifford. Phase B reduces
single-type blocks directly and mixed-type blocks via a symplectic basis change
(see the theory doc); the column below is kept because the block *type* still
governs which code path runs and how much work the reduction takes.

| circuit | blocks (>=2) | single-type | mixed-type | largest |
|---|--:|--:|--:|--:|
| ccz_complete_6 | 1 | 1 | 0 | 20 |
| ccz_complete_6_hmixed | 1 | 0 | 1 | 20 |
| ccz_ladder_6 | 1 | 1 | 0 | 20 |
| toffoli_single | 1 | 0 | 1 | 7 |
| toffoli_chain_3 | 3 | 0 | 3 | 7 |
| random_8q_d200 | 5 | 0 | 5 | 5 |
| cultivation_d5 (real) | 5 | 5 | 0 | 19 |

## Analysis

**Phase A folding contributes nothing beyond peephole.** In every row
`+foldA == peephole`. This empirically confirms the theory: `PeepholeFusionPass`
already reaches the per-block same-axis folding optimum (see also
`test_tcount_phasepoly.py::TestRedundantWithPeephole`). Phase A exists only to
build the gate-synthesis matrix and to make the per-phase split measurable; it is
not itself a source of reduction.

**Phase B (TOHPE) genuinely reduces ancilla-free T-count, on the circuits that
carry cubic redundancy.** It removes T gates that folding provably cannot:
`s_empty_4/5` collapse 15/31 -> 0 (Amy-Maslov-Mosca trivial polynomials),
`s_empty_4_minus_full` collapses 14 -> 1 (the optimal single `T_dag`), and -- the
non-degenerate case -- the dense diagonal `ccz_complete_6` drops **20 -> 12**
ancilla-free. Every accepted reduction is verified against the full
`f(x) mod 8`, so these are exact. This is direct evidence that the
implementation reproduces real TOHPE reductions, not just folding.

**TOHPE is selective, and the block-structure table says why.** It fires on
*dense* shared cubic structure (`ccz_complete_*`) but not on *sparse* structure:
`ccz_ladder_*` and `ccz_star_*` are single-type but, once same-parity folding is
applied, carry no residual cubic redundancy, so TOHPE returns them unchanged
(`removed = 0`). The faithful Vandaele Algorithm 2 search (single-column
candidates plus pairwise, objective-maximised) confirms this is a genuine
property of those polynomials, not a search artifact -- it was the addition of
the single-column candidates that turned `ccz_complete` from 0 into the 8-gate
reduction above.

**Hadamard-bearing circuits become mixed-type, and Phase B now handles them.** A
Toffoli is `H; CCZ; H`; once the front end absorbs the Hadamards into `U_C`, the
block's axes mix Pauli planes (`toffoli_*`, `random_*`, and `*_hmixed` show 0
single-type blocks). The mixed-type path diagonalizes such a block in a
symplectic generator basis, reduces it there, and maps the result back to product
Paulis with exact signs (theory doc, "Mixed-type blocks"). `ccz_complete_6_hmixed`
demonstrates this: a fully mixed-type block reduced 20 -> 12 with the statevector
preserved exactly. The Toffoli rows still show `removed = 0` -- not because they
are skipped, but because a single Toffoli is one CCZ, which is already T-optimal
(7 T on three qubits, Amy-Maslov-Mosca), and `toffoli_chain_3` is three such
independent blocks with no shared cubic structure to exploit.

**Calibration against the literature.** The pattern matches Vandaele 2024,
Table 2 (ancilla-free): structured diagonal circuits reduce, while many standard
benchmarks see little ancilla-free reduction because their headline op-T-mize
gains come from Hadamard gadgetization with ancillas. Clifft sits in the
ancilla-free regime, so the dense-diagonal wins (`ccz_complete`) are exactly the
regime where ancilla-free TOHPE is expected to help.

**Relation to existing implementations.** The published TOHPE/FastTODD reference
is Vandaele's Rust tool (`VivienVandaele/quantum_circuit_optimization`); TODD has
Heyfron-Campbell's C++ `TOpt`, and phase folding has Amy's `feynman`/`t-par`.
Those operate on `{CNOT, T}` QASM circuits and pay an explicit Hadamard/ancilla
overhead. This PR is a from-scratch in-HIR implementation that consumes the
already-Clifford-absorbed virtual Pauli axes directly (no CNOT tracking, no
ancillas) and adds the exact-`f` verification gate, which those tools do not
need because they re-synthesise a full circuit.

## Conclusion: is this worth productionizing?

**Conditionally yes, for circuits with cubic phase-polynomial redundancy.**
Phase B demonstrably and exactly reduces ancilla-free T-count on dense diagonal
structure (`ccz_complete_6`: 20 -> 12, 40% beyond peephole) and on its mixed-type
Hadamard-conjugate (`ccz_complete_6_hmixed`, same 8 gates) -- the family that
shows up in IQP sampling, Hamming-weight phasing, and diagonal phase oracles. On
sparse structure, on random circuits, and on the real `cultivation_d5` circuit it
matches peephole, so it should remain **opt-in** rather than default; a user
targeting structured circuits can enable it for a real win, while a user on
generic near-Clifford workloads pays nothing by leaving it off.

Both block types are handled: single-Pauli-type blocks are reduced directly, and
mixed-type (Hadamard-absorbed) blocks are reduced via a symplectic basis change
entirely within the HIR -- so the single-type restriction noted in earlier
revisions is gone. The remaining follow-ups are:

1. **FastTODD (Vandaele Theorem 6)** in place of TOHPE (Theorem 1), which can
   find marginally more reductions per block.
2. **Ancillas are the hard ceiling, and that one *is* structural.** The largest
   op-T-mize gains (GF(2^m) multipliers, adders) come from Hadamard gadgetization,
   which adds qubits and mid-circuit measurement. That is a circuit-level
   transformation, but Clifft's VM allocates exactly `2^{k_max}` amplitudes once
   and the Pauli arena is fixed at trace time, so introducing new qubits
   mid-circuit is not expressible without relaxing those invariants -- a separate
   VM-level design discussion, out of scope here.

A note on the FTCircuitBench suite suggested in review: its inputs are
arbitrary-angle (continuous `rz`/`cu1`) circuits intended to be fed through
Gridsynth, and its pre-synthesised Clifford+T outputs are Gridsynth-exploded
(e.g. `qft_4q` -> ~115k T gates from approximation, not algorithmic structure),
so they are not a meaningful target for a phase-polynomial T-reducer without
first standing up the full synthesis pipeline. The op-T-mize / Amy benchmark set
(already Clifford+T: GF(2^m)-mult, adders, `tof_n`, `barenco_tof`) is the right
real-world corpus; those circuits are Toffoli-based and therefore mixed-type in
Clifft, which the mixed-type path now handles -- ingesting them only needs a
`.qc`/QASM -> Stim front end, which is the natural next benchmarking step.
