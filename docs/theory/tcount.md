# Global T-count Reduction in the Heisenberg IR

This page is the theory companion for the experimental, opt-in
`PhasePolyTCountPass` (issue #40). It states the optimization problem, the prior
work it builds on, and why the technique is valid on Clifft's HIR without
changing the parser, HIR layout, bytecode, or VM.

## Why T-count

In the standard fault-tolerant gate set Clifford + T, Clifford gates are cheap
(often transversal), while each T gate is implemented by magic-state distillation
and injection at a cost one to several orders of magnitude higher
(Bravyi and Kitaev, *Universal quantum computation with ideal Clifford gates and
noisy ancillas*, Phys. Rev. A 71, 022316, 2005). The T-count, the number of T
gates, is therefore the dominant non-Clifford cost metric and the target of a
large optimization literature.

## Clifft's HIR is a phase-polynomial substrate

Clifft's front end absorbs every physical Clifford, including every Hadamard,
into the offline Clifford frame `U_C` (Heisenberg picture). What survives in the
HIR is a sequence of non-Clifford operations carried as virtual Pauli strings:
each `T_GATE` op is a pi/4 rotation `exp(-i (pi/8) P)` about a virtual Pauli axis
`P`, recorded as an (x, z, sign) mask. Writing a discrete or continuous
non-Clifford gate as a Pauli rotation conjugated by the accumulated Clifford
frame is standard; see Litinski, *A Game of Surface Codes*, Quantum 3, 128, 2019,
eq. (1), and Vandaele 2024 below, eq. (2).

A set of Pauli rotations whose axes pairwise commute is simultaneously
diagonalizable: a single Clifford `W` maps the axes to Z-type strings, that is,
to binary parities. In that basis the block is a phase polynomial (Amy, Maslov,
Mosca, *Polynomial-time T-depth Optimization of Clifford+T circuits via Matroid
Partitioning*, arXiv:1303.2042, 2013, Lemma 3):

$$U\,|x\rangle = \omega^{p(x)}\,|g(x)\rangle, \qquad \omega = e^{i\pi/4},$$

$$p(x) = \sum_k c_k\,(a_k \cdot x) \pmod 8, \qquad c_k \in \mathbb{Z}_8,\quad a_k \in \mathbb{F}_2^{n}.$$

Each parity `a_k` with an odd coefficient `c_k` costs one T gate; even
coefficients are Clifford (`c=2` is S, `c=4` is Z). Two phase polynomials that
agree after taking every coefficient mod 2 are Clifford-equivalent.

## The two reductions, and why each is valid here

### Phase A: phase folding (Amy-Maslov-Mosca)

Collect the per-axis coefficients of a commuting block and add them in `Z_8`.
T gates on the same parity, however far apart, merge: `T + T = S` (T-count 2 ->
0), `T^4 = Z`, `T^8 = I`. This is the phase-folding step of the Tpar algorithm
(arXiv:1303.2042, Example 1). On Clifft's HIR the parities are read straight off
the (x, z) masks, so no parity tracking through CNOTs is needed; the Cliffords
are already in `U_C`.

Folding reaches the optimum of the restricted representation in which the block
is written as a product of single-axis Pauli rotations, and only that restricted
optimum, not the global T-count optimum of the block. In that representation the
rotation characters $(-1)^{a_k\cdot x}$ are orthogonal, so two distinct parities
cannot cancel and folding cannot do better. This is an elementary
character-orthogonality argument, not a result quoted from a reference. It is a
local claim about one representation class, and it is what Phase B is built to
beat: TODD/TOHPE leave the product-of-rotations class and exploit the cubic
gate-synthesis structure to go below the distinct-parity count (for example
$S_\emptyset$: 15 T $\to$ 0, which folding cannot touch). `PeepholeFusionPass`
already reaches this folding optimum along commuting paths; the unit test
`test_tcount_phasepoly.py::TestRedundantWithPeephole` confirms Phase A removes
zero further T gates after peephole. Phase A is included only to build the
gate-synthesis matrix Phase B consumes and to make the per-phase contribution
measurable in isolation.

### Phase B: multi-axis reduction (TODD / TOHPE)

Going below the distinct-parity count requires the cubic structure of the phase
polynomial. Stack the odd-coefficient parities of a commuting block as the
columns of a gate-synthesis matrix $A \in \mathbb{F}_2^{r \times m}$ (Heyfron and
Campbell, *An Efficient Quantum Compiler that reduces T count*, arXiv:1712.01557,
2018, Lemma II.1). The unitary depends on $A$ only through its order-3 signature
tensor

$$S_{\alpha\beta\gamma} = \sum_j A_{\alpha,j}\,A_{\beta,j}\,A_{\gamma,j} \pmod 2,$$

so minimizing T-count is the 3rd-order symmetric tensor rank problem (Amy and
Mosca, *T-count optimization and Reed-Muller codes*, arXiv:1601.07363, 2019,
equivalent to decoding a punctured Reed-Muller code; NP-hard, van de Wetering and
Amy, arXiv:2310.05958).

The reducer is TOHPE (Vandaele, *Lower T-count with faster algorithms*,
arXiv:2407.08695, 2024/2025, the current state of the art, faster than and at
least as good as TODD). TOHPE performs a duplicate-and-destroy column update
$A \to A \oplus z\,y^{T}$ that preserves the signature tensor exactly when
(Vandaele Theorem 1):

- $C1:\quad |y| \equiv 0 \pmod 2$
- $C2:\quad |A_\alpha \wedge y| \equiv 0 \pmod 2 \quad$ for all rows $\alpha$
- $C3:\quad |A_\alpha \wedge A_\beta \wedge y| \equiv 0 \pmod 2 \quad$ for all row pairs $\alpha < \beta$

where $|\cdot|$ is Hamming weight and $\wedge$ is bitwise AND. The candidate
update vectors are $z \in \{\,c_i \oplus c_j\,\} \cup \{\,c_i\,\}$: the pairwise
column XORs and the single columns (Vandaele Algorithm 2, line 2). This PR
implements both and picks the move that removes the most columns rather than the
first feasible one. A valid $(z, y)$ makes two columns duplicates (or zeroes
one); destroying them removes T gates while leaving the unitary unchanged up to a
Clifford (Vandaele Theorem 2, subadditivity).

### Why this fits Clifft with no structural change

Hadamards are absorbed, not partitioned. Phase-polynomial methods normally split
a Clifford + T circuit into Hadamard-free `{CNOT, T}` blocks and pay a
Clifford/ancilla overhead at the cuts (Vandaele section 2.1; Heyfron-Campbell
Hadamard gadgetization), because the gate-synthesis matrix only describes a block
with no interior Hadamards. Clifft does not cut the circuit this way. The
Hadamards are still present, but the front end conjugates them into the offline
Clifford frame `U_C`, so their effect is carried in the `(x, z)` masks of the
surviving non-Clifford ops. A maximal commuting `T_GATE` run is then already a
Hadamard-free phase polynomial in those masks, with the gate-synthesis matrix in
hand: the property the partition exists to produce, obtained without
partitioning.

No new ops or axes are introduced. The reduced columns are products of the
block's own generators, hence Paulis expressible as XORs of existing masks, and
they are re-emitted as `T_GATE` ops reusing freed arena slots. The fixed-capacity
`PauliMaskArena` is never grown (see `claim_empty_pauli_mask`), so the pass
respects the "no HIR data-structure change" constraint.

The Clifford residual is emitted as a PHASE_ROTATION and absorbed by the trailing
peephole. Each duplicate-and-destroy step leaves a single-axis Clifford phase:
cancelling a duplicate column pair on parity $w$ leaves $T^2 = S$ on that one
axis (never a two-axis CZ, because the step only ever removes a duplicate pair).
The pass re-emits this as a `PHASE_ROTATION` op on $w$ (0.5 = S, 1.0 = Z,
1.5 = S_dag), reusing a freed arena slot, so the pass itself does not call into
the tableau machinery. The intended pipeline runs `PeepholeFusionPass` after this
pass; peephole's Clifford-angle demotion then absorbs those `PHASE_ROTATION`s
into the frame via the same symplectic S-conjugation it already uses
(`apply_virtual_s_downstream` / `conjugate_pauli_by_S` in `peephole.cc`, updating
the final tableau and `global_weight`). The exactness check in `tcount_tohpe.cc`
validates the emitted `T_GATE` + `PHASE_ROTATION` set against the original phase
function, so the re-emission is verified regardless of when the residual is
absorbed.

### Scope: single-Pauli-type blocks only

Phase B runs only on blocks whose axes are all Z-type (or all X-type). There the
diagonal phase function $f(x) \bmod 8$ over the computational basis *is* the
unitary, so checking it before accepting a move pins the result down exactly,
global phase included.

A block whose axes mix the X and Z planes (for example a Hadamard-absorbed
Toffoli, which leaves $Y$-bearing axes once the frame absorbs the Hadamards) is
diagonal only in some other basis. Reducing it correctly would mean tracking the
Clifford correction Theorem 1 leaves behind under that change of basis, which can
include quadratic ($CZ$) terms and a global phase. A coordinate-space $f$ check
is not enough on its own: two operators can agree on the coordinate-space phase
function yet differ by a global phase on the physical qubits, so a move can pass
that check and still change the unitary. Until Phase B carries an exact
physical-space check for this case, it leaves mixed-type blocks to Phase A
folding. Extending it is future work.

## The ancilla-free ceiling

Clifft cannot add qubits: the VM allocates exactly `2^{k_max}` amplitudes and the
Pauli arena is fixed at trace time, so this pass operates in the ancilla-free
regime. The op-T-mize literature shows that the large T-count reductions (for
example GF(2^m) multipliers, adders) come disproportionately from Hadamard
gadgetization with ancillas; ancilla-free, many benchmark circuits see little or
no reduction (Vandaele 2024, Table 2). Because Clifft has already absorbed the
Hadamards into `U_C`, consecutive T axes frequently anti-commute, so commuting
blocks can be small.

Whether HIR-level phase-polynomial reduction is worth productionizing in Clifft
is therefore an empirical question: it depends on the commuting-block-size
distribution of the target workloads. The companion benchmark
(`tools/bench/tcount/`) measures, per circuit and per phase, the T-count under
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
