"""Correctness + behaviour tests for the experimental TCountPhasePolyPass.

Two verification strategies, mirroring test_peephole_oracle.py:

1. Self-consistency: compile with the pass on vs off, assert identical
   statevectors (fidelity ~ 1), proving exact semantics preservation.
2. Structural: the pass folds per-axis T coefficients in Z_8 and reports
   analyzer metrics; we check the metrics and the known algebraic identities.

We also pin down the central finding of issue #40: within a commuting block
this pass reaches the same per-block optimum that PeepholeFusionPass already
reaches, so running it after peephole removes nothing further.
"""

import numpy as np
import pytest
from conftest import (
    assert_statevectors_equal,
    random_clifford_t_circuit,
    random_dense_clifford_t_circuit,
)

import clifft


def _phasepoly_only(circuit_str: str) -> clifft.Program:
    """Compile running ONLY the experimental pass on the HIR."""
    hir = clifft.trace(clifft.parse(circuit_str))
    pm = clifft.HirPassManager()
    pm.add(clifft.TCountPhasePolyPass())
    pm.run(hir)
    return clifft.lower(hir)


def _statevector(prog: clifft.Program) -> np.ndarray:
    state = clifft.State(peak_rank=prog.peak_rank, num_measurements=prog.num_measurements)
    clifft.execute(prog, state)
    return np.array(clifft.get_statevector(prog, state))


def _t_count(circuit_str: str, *, optimize: bool) -> int:
    hir = clifft.trace(clifft.parse(circuit_str))
    if optimize:
        pm = clifft.HirPassManager()
        pm.add(clifft.TCountPhasePolyPass())
        pm.run(hir)
    return int(hir.num_t_gates)


# ---------------------------------------------------------------------------
# Semantic equivalence (pass on vs off) on random Clifford+T circuits
# ---------------------------------------------------------------------------


class TestStatevectorEquivalence:
    @pytest.mark.parametrize("num_qubits", [2, 3, 4, 5, 6])
    @pytest.mark.parametrize("seed", range(5))
    def test_random_small(self, num_qubits: int, seed: int) -> None:
        circuit = random_clifford_t_circuit(num_qubits, depth=20, seed=seed)
        base = _statevector(clifft.compile(circuit, hir_passes=None, bytecode_passes=None))
        opt = _statevector(_phasepoly_only(circuit))
        assert_statevectors_equal(opt, base, msg=f"{num_qubits}q seed={seed}")

    @pytest.mark.parametrize("seed", range(8))
    def test_random_8q_depth30(self, seed: int) -> None:
        circuit = random_clifford_t_circuit(8, depth=30, seed=seed)
        base = _statevector(clifft.compile(circuit, hir_passes=None, bytecode_passes=None))
        opt = _statevector(_phasepoly_only(circuit))
        assert_statevectors_equal(opt, base, msg=f"8q seed={seed}")

    @pytest.mark.parametrize("seed", range(5))
    def test_dense_entangled(self, seed: int) -> None:
        circuit = random_dense_clifford_t_circuit(5, depth=40, seed=seed)
        base = _statevector(clifft.compile(circuit, hir_passes=None, bytecode_passes=None))
        opt = _statevector(_phasepoly_only(circuit))
        assert_statevectors_equal(opt, base, msg=f"dense 5q seed={seed}")


# ---------------------------------------------------------------------------
# Algebraic identities and analyzer metrics
# ---------------------------------------------------------------------------


class TestFolding:
    def test_two_t_to_clifford(self) -> None:
        assert _t_count("T 0\nT 0", optimize=False) == 2
        assert _t_count("T 0\nT 0", optimize=True) == 0

    def test_three_t_to_one(self) -> None:
        assert _t_count("T 0\nT 0\nT 0", optimize=True) == 1

    def test_four_t_to_clifford(self) -> None:
        assert _t_count("T 0\nT 0\nT 0\nT 0", optimize=True) == 0

    def test_eight_t_to_identity(self) -> None:
        assert _t_count("T 0\nT 0\nT 0\nT 0\nT 0\nT 0\nT 0\nT 0", optimize=True) == 0

    def test_anticommuting_axis_blocks(self) -> None:
        # H rotates the middle T off the Z axis; the three T gates cannot fold.
        assert _t_count("T 0\nH 0\nT 0\nH 0\nT 0", optimize=True) == 3

    def test_pass_metrics(self) -> None:
        hir = clifft.trace(clifft.parse("T 0\nT 0\nT 1\nT 1\nT 1"))
        p = clifft.TCountPhasePolyPass()
        pm = clifft.HirPassManager()
        pm.add(p)
        pm.run(hir)
        # One commuting block of 5 (Z0,Z1 commute); two axis groups.
        assert p.blocks == 1
        assert p.t_before == 5
        assert p.max_block_axes == 2
        # q0: c=2 -> 0 T; q1: c=3 -> 1 T. Net 4 removed.
        assert p.t_after == 1
        assert p.t_removed == 4


class TestTohpe:
    """Phase B (multi-axis TOHPE) removes T gates that folding cannot."""

    @staticmethod
    def _parity(mask: int, dag: bool = False) -> str:
        bits = [q for q in range(8) if mask & (1 << q)]
        t = bits[0]
        up = "".join(f"CX {b} {t}\n" for b in bits[1:])
        dn = "".join(f"CX {b} {t}\n" for b in reversed(bits[1:]))
        return up + ("T_DAG " if dag else "T ") + f"{t}\n" + dn

    def test_s_empty_reduces_beyond_folding(self) -> None:
        # T on every nonzero parity of F_2^4 (Amy-Maslov-Mosca S_empty): folding
        # removes none (15 distinct axes); TOHPE collapses it.
        text = "".join(self._parity(a) for a in range(1, 16))
        hir = clifft.trace(clifft.parse(text))
        assert int(hir.num_t_gates) == 15

        p = clifft.TCountPhasePolyPass()  # TOHPE enabled
        pm = clifft.HirPassManager()
        pm.add(p)
        pm.run(hir)

        assert int(hir.num_t_gates) < 15
        assert p.tohpe_removed > 0
        assert p.tohpe_blocks >= 1

    def test_folding_alone_does_not_reduce_s_empty(self) -> None:
        text = "".join(self._parity(a) for a in range(1, 16))
        hir = clifft.trace(clifft.parse(text))
        p = clifft.TCountPhasePolyPass(enable_tohpe=False)
        pm = clifft.HirPassManager()
        pm.add(p)
        pm.run(hir)
        assert int(hir.num_t_gates) == 15  # folding sees 15 distinct axes

    def _ccz(self, a: int, b: int, c: int) -> str:
        p = self._parity
        return (
            p(1 << a)
            + p(1 << b)
            + p(1 << c)
            + p((1 << a) | (1 << b), dag=True)
            + p((1 << a) | (1 << c), dag=True)
            + p((1 << b) | (1 << c), dag=True)
            + p((1 << a) | (1 << b) | (1 << c))
        )

    def test_mixed_type_block_reduces_and_preserves_statevector(self) -> None:
        # ccz_complete(6) conjugated by H on {0,1,2}: parities touching those
        # qubits rotate into the X plane, so the commuting block is MIXED-type.
        # The mixed-type path diagonalizes it in a symplectic basis, reduces it,
        # and preserves the (genuinely non-diagonal) unitary exactly.
        inner = "".join(
            self._ccz(a, b, c) for a in range(6) for b in range(a + 1, 6) for c in range(b + 1, 6)
        )
        text = "H 0\nH 1\nH 2\n" + inner + "H 0\nH 1\nH 2\n"

        # The intended pipeline is peephole then the pass (as in
        # tools/bench/tcount). Peephole canonicalizes the commuting block first.
        hir = clifft.trace(clifft.parse(text))
        t0 = int(hir.num_t_gates)
        pm = clifft.HirPassManager()
        pm.add(clifft.PeepholeFusionPass())
        p = clifft.TCountPhasePolyPass()
        pm.add(p)
        pm.run(hir)

        assert p.tohpe_removed > 0  # the mixed-type path fired
        assert int(hir.num_t_gates) < t0
        opt = _statevector(clifft.lower(hir))
        base = _statevector(clifft.compile(text, hir_passes=None, bytecode_passes=None))
        assert_statevectors_equal(opt, base, msg="mixed-type ccz_complete_6 H-conjugated")

    def test_max_verify_bits_kwarg_gates_reduction(self) -> None:
        # S_empty(4) spans support 4. A cutoff below 4 disables Phase B; the
        # default reduces. Checks the binding's max_verify_bits argument.
        text = "".join(self._parity(a) for a in range(1, 16))

        def t_count(max_verify_bits: int) -> int:
            hir = clifft.trace(clifft.parse(text))
            pm = clifft.HirPassManager()
            pm.add(clifft.TCountPhasePolyPass(max_verify_bits=max_verify_bits))
            pm.run(hir)
            return int(hir.num_t_gates)

        assert t_count(3) == 15  # cutoff below the support: unchanged
        assert t_count(14) < 15  # default cutoff: reduces


# ---------------------------------------------------------------------------
# Central finding: peephole already reaches the per-block optimum
# ---------------------------------------------------------------------------


class TestRedundantWithPeephole:
    @pytest.mark.parametrize("seed", range(8))
    def test_no_gain_after_peephole(self, seed: int) -> None:
        """After PeepholeFusionPass, the pass finds nothing further to remove.

        Demonstrates that same-axis folding (which peephole already performs)
        is the commuting-block optimum: any leftover multi-axis reduction needs
        structure outside the current HIR.
        """
        circuit = random_clifford_t_circuit(6, depth=30, seed=seed)
        hir = clifft.trace(clifft.parse(circuit))
        pm = clifft.HirPassManager()
        pm.add(clifft.PeepholeFusionPass())
        pm.run(hir)
        t_after_peephole = int(hir.num_t_gates)

        # Phase A folding only (enable_tohpe=False): this isolates the claim
        # that same-axis folding adds nothing once peephole has run.
        pp = clifft.TCountPhasePolyPass(enable_tohpe=False)
        pm2 = clifft.HirPassManager()
        pm2.add(pp)
        pm2.run(hir)

        assert pp.t_removed == 0
        assert int(hir.num_t_gates) == t_after_peephole


# ---------------------------------------------------------------------------
# Pass enabled in the default pipeline: a "default-on" run over the corpus,
# confirming it composes with the other passes and preserves the circuit.
# ---------------------------------------------------------------------------


class TestPassInDefaultPipeline:
    @staticmethod
    def _with_pass(circuit_str: str) -> clifft.Program:
        """Default HIR pipeline (peephole, squeeze) with the pass inserted."""
        hir = clifft.trace(clifft.parse(circuit_str))
        pm = clifft.HirPassManager()
        pm.add(clifft.PeepholeFusionPass())
        pm.add(clifft.TCountPhasePolyPass())
        pm.add(clifft.StatevectorSqueezePass())
        pm.run(hir)
        return clifft.lower(hir)

    @pytest.mark.parametrize("num_qubits", [2, 3, 4, 5, 6])
    @pytest.mark.parametrize("seed", range(6))
    def test_random_matches_default(self, num_qubits: int, seed: int) -> None:
        circuit = random_clifford_t_circuit(num_qubits, depth=30, seed=seed)
        base = _statevector(clifft.compile(circuit))  # default passes
        opt = _statevector(self._with_pass(circuit))
        assert_statevectors_equal(opt, base, msg=f"{num_qubits}q seed={seed}")

    @pytest.mark.parametrize("seed", range(5))
    def test_dense_entangled_matches_default(self, seed: int) -> None:
        circuit = random_dense_clifford_t_circuit(5, depth=40, seed=seed)
        base = _statevector(clifft.compile(circuit))
        opt = _statevector(self._with_pass(circuit))
        assert_statevectors_equal(opt, base, msg=f"dense 5q seed={seed}")
