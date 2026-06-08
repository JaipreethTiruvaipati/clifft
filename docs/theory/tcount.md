# Global T-count Reduction in the Heisenberg IR

This page is the theory companion for the experimental, opt-in
`PhasePolyTCountPass` (issue #40). It states the optimization problem, the
prior work it builds on, and *why the technique is exactly valid on Clifft's
HIR without changing the parser, HIR layout, bytecode, or VM*.

## Why T-count

In the standard fault-tolerant gate set Clifford + T, Clifford gates are cheap
(often transversal) while each T gate is implemented by magic-state distillation
and injection at a cost one to several orders of magnitude higher
(Bravyi and Kitaev, *Universal quantum computation with ideal Clifford gates and
noisy ancillas*, Phys. Rev. A 71, 022316, 2005). The **T-count** -- the number of
T gates -- is therefore the dominant non-Clifford cost metric and the target of a
large optimization literature.

## Clifft's HIR is a phase-polynomial substrate

Clifft's front end absorbs every physical Clifford -- including every Hadamard --
into the offline Clifford frame `U_C` (Heisenberg picture). What survives in the
HIR is a sequence of non-Clifford operations carried as **virtual Pauli
strings**: each `T_GATE` op is a pi/4 rotation `exp(-i (pi/8) P)` about a virtual
Pauli axis `P`, recorded as an (x, z, sign) mask. (Writing a discrete or
continuous non-Clifford gate as a Pauli rotation conjugated by the accumulated
Clifford frame is standard; see Litinski, *A Game of Surface Codes*, Quantum 3,
128, 2019, eq. (1), and Vandaele 2024 below, eq. (2).)

A set of Pauli rotations whose axes **pairwise commute** is simultaneously
diagonalizable: a single Clifford `W` maps the axes to Z-type strings, i.e. to
binary parities. In that basis the block is exactly a **phase polynomial**
(Amy, Maslov, Mosca, *Polynomial-time T-depth Optimization of Clifford+T circuits
via Matroid Partitioning*, arXiv:1303.2042, 2013, Lemma 3):

```
U |x> = omega^{p(x)} |g(x)|,   omega = e^{i pi / 4},
p(x) = sum_k c_k (a_k . x)  (mod 8),   c_k in Z_8,  a_k in F_2^n.
```

Each parity `a_k` with an **odd** coefficient `c_k` costs one T gate; even
coefficients are Clifford (`c=2` is S, `c=4` is Z). Two phase polynomials that
agree after taking every coefficient mod 2 are Clifford-equivalent.

## The two reductions, and why each is valid here

### Phase A -- phase folding (Amy-Maslov-Mosca)

Collect the per-axis coefficients of a commuting block and add them in `Z_8`.
T gates on the *same* parity, however far apart, merge:
`T + T = S` (T-count 2 -> 0), `T^4 = Z`, `T^8 = I`. This is the
phase-folding step of the Tpar algorithm (arXiv:1303.2042, Example 1). On
Clifft's HIR the parities are read straight off the (x, z) masks, so no parity
tracking through CNOTs is needed -- the Cliffords are already in `U_C`.

**Folding is provably T-optimal within a commuting block** for any representation
as a product of Pauli rotations: the rotation characters are orthogonal, so
distinct axes cannot cancel. `PeepholeFusionPass` already reaches this per-block
optimum along commuting paths; the unit test
`test_tcount_phasepoly.py::TestRedundantWithPeephole` confirms Phase A removes
**zero** further T gates after peephole. Phase A is included only to drive the
phase-polynomial matrix used by Phase B and to make the per-phase contribution
measurable in isolation.

### Phase B -- multi-axis reduction (TODD / TOHPE)

Going below the distinct-parity count requires the cubic structure of the phase
polynomial. Stack the odd-coefficient parities of a commuting block as the
columns of a **gate-synthesis matrix** `A in F_2^{r x m}` (Heyfron and Campbell,
*An Efficient Quantum Compiler that reduces T count*, arXiv:1712.01557, 2018,
Lemma II.1). The unitary depends on `A` only through its order-3 **signature
tensor**

```
S_{a,b,g} = sum_j A_{a,j} A_{b,j} A_{g,j}  (mod 2),
```

so minimizing T-count is the *3rd-order symmetric tensor rank* problem
(Amy and Mosca, *T-count optimization and Reed-Muller codes*, arXiv:1601.07363,
2019 -- equivalent to decoding a punctured Reed-Muller code; NP-hard, van de
Wetering and Amy, arXiv:2310.05958).

We use the **TOHPE** reducer (Vandaele, *Lower T-count with faster algorithms*,
arXiv:2407.08695, 2024/2025 -- the current state of the art, faster than and at
least as good as TODD). TOHPE performs a *duplicate-and-destroy* column update
`A -> A xor z y^T` that **preserves the signature tensor exactly** when (Vandaele
Theorem 1)

```
C1:  |y|              = 0 (mod 2)
C2:  |A_a  & y|       = 0 (mod 2)   for all rows a
C3:  |A_a & A_b & y|  = 0 (mod 2)   for all row pairs a < b
```

with `z = col_a xor col_b`. A `y` satisfying C1-C3 makes two columns duplicates;
destroying the duplicate pair removes two T gates while leaving the unitary
unchanged up to a Clifford (Vandaele Theorem 2, subadditivity).

### Why this fits Clifft with no structural change

* **Hadamards for free.** Phase-polynomial methods normally split a Clifford + T
  circuit into Hadamard-free `{CNOT, T}` blocks, paying a Clifford/ancilla
  overhead at the cuts (Vandaele section 2.1; Heyfron-Campbell Hadamard
  gadgetization). Clifft's frame absorption *is* that partition, performed once
  by the front end. A maximal commuting `T_GATE` run is precisely a Hadamard-free
  phase-polynomial block, with the gate-synthesis matrix already in hand.

* **No new ops, no new axes.** The reduced columns are products of the block's
  own generators, hence Paulis expressible as XORs of existing masks -- re-emitted
  as `T_GATE` ops reusing freed arena slots. The fixed-capacity `PauliMaskArena`
  is never grown (see `claim_empty_pauli_mask`), so the pass respects the
  "no HIR data-structure change" constraint.

* **Clifford residual re-absorbed by existing machinery.** The Clifford that
  TOHPE leaves behind is diagonal (S/Z on single axes, CZ on axis pairs). The
  single-axis part is absorbed downstream by the *same* symplectic S-conjugation
  the peephole already uses (`apply_virtual_s_downstream` /
  `conjugate_pauli_by_S` in `peephole.cc`), which also updates the final tableau
  and `global_weight`.

## The ancilla-free ceiling (the scientific question)

Clifft cannot add qubits -- the VM allocates exactly `2^{k_max}` amplitudes and
the Pauli arena is fixed at trace time. So this pass operates strictly in the
**ancilla-free** regime. The op-T-mize literature shows that the large T-count
reductions (e.g. GF(2^m) multipliers, adders) come disproportionately from
*Hadamard gadgetization with ancillas*; ancilla-free, many benchmark circuits see
little or no reduction (Vandaele 2024, Table 2). Moreover, because Clifft has
already absorbed the Hadamards into `U_C`, consecutive T axes frequently
anti-commute, so commuting blocks can be small.

Whether HIR-level phase-polynomial reduction is worth productionizing in Clifft
is therefore an **empirical** question: it depends on the commuting-block-size
distribution of the target workloads. The companion benchmark
(`tools/bench/tcount/`) measures, per circuit and per phase: T-count under
no-opt / peephole / +folding / +TOHPE, and the commuting-block-size histogram
that explains where TOHPE can and cannot fire. See the evaluation summary in the
pull request for the conclusion.

## References

1. M. Amy, D. Maslov, M. Mosca. *Polynomial-time T-depth Optimization of
   Clifford+T circuits via Matroid Partitioning.* arXiv:1303.2042 (2013).
2. M. Amy, M. Mosca. *T-count optimization and Reed-Muller codes.*
   arXiv:1601.07363 (2019).
3. L. E. Heyfron, E. T. Campbell. *An Efficient Quantum Compiler that reduces
   T count.* arXiv:1712.01557 (2018).
4. V. Vandaele. *Lower T-count with faster algorithms* (TOHPE / FastTODD).
   arXiv:2407.08695 (2024/2025).
5. A. Kissinger, J. van de Wetering. *Reducing T-count with the ZX-calculus.*
   arXiv:1903.10477 (2019). (Complementary; ZX tooling is out of scope here.)
6. S. Bravyi, A. Kitaev. *Universal quantum computation with ideal Clifford gates
   and noisy ancillas.* Phys. Rev. A 71, 022316 (2005).
7. D. Litinski. *A Game of Surface Codes.* Quantum 3, 128 (2019).
