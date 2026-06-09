"""Per-phase T-count evaluation for the experimental TCountPhasePolyPass (issue #40).

Reproduces the table in docs/theory/tcount_evaluation.md. The authoritative
numbers are produced by the C++ harness (tools/bench/tcount/bench_tcount.cc),
which can also read the real cultivation_d5.stim fixture and compute the
commuting-block histogram; this Python script mirrors the per-phase table using
the public API.

Run:  uv run python docs/guide/scripts/tcount_evaluation.py
"""

from __future__ import annotations

import clifft

# --- Circuit generators (Clifford+T text; parser has no CCX/CCZ) ----------------


def parity_phase(mask: int, dag: bool) -> str:
    """T (or T_dag) on the Z-parity given by the set bits of mask."""
    bits = [q for q in range(64) if mask & (1 << q)]
    tgt = bits[0]
    up = "".join(f"CX {b} {tgt}\n" for b in bits[1:])
    down = "".join(f"CX {b} {tgt}\n" for b in reversed(bits[1:]))
    return up + ("T_DAG " if dag else "T ") + f"{tgt}\n" + down


def ccz(a: int, b: int, c: int) -> str:
    """CCZ phase polynomial: +singles, -pairs, +triple (7 T gates, diagonal)."""
    return (
        parity_phase(1 << a, False)
        + parity_phase(1 << b, False)
        + parity_phase(1 << c, False)
        + parity_phase((1 << a) | (1 << b), True)
        + parity_phase((1 << a) | (1 << c), True)
        + parity_phase((1 << b) | (1 << c), True)
        + parity_phase((1 << a) | (1 << b) | (1 << c), False)
    )


def ccz_ladder(k: int) -> str:
    return "".join(ccz(i, i + 1, i + 2) for i in range(k))


def toffoli(a: int, b: int, c: int) -> str:
    return f"H {c}\n" + ccz(a, b, c) + f"H {c}\n"


def toffoli_chain(k: int) -> str:
    return "".join(toffoli(i, i + 1, i + 2) for i in range(k))


def ccz_complete(nq: int) -> str:
    """All C(nq, 3) CCZ triples: a dense diagonal phase polynomial."""
    return "".join(
        ccz(a, b, c) for a in range(nq) for b in range(a + 1, nq) for c in range(b + 1, nq)
    )


def s_empty(nq: int, skip_full: bool = False) -> str:
    out = []
    for a in range(1, 1 << nq):
        if skip_full and a == (1 << nq) - 1:
            continue
        out.append(parity_phase(a, False))
    return "".join(out)


# --- Per-phase measurement ------------------------------------------------------


def t_count(text: str, peephole: bool, fold: bool, tohpe: bool) -> tuple[int, int, int]:
    hir = clifft.trace(clifft.parse(text))
    if peephole:
        pm = clifft.HirPassManager()
        pm.add(clifft.PeepholeFusionPass())
        pm.run(hir)
    removed = blocks = 0
    if fold:
        p = clifft.TCountPhasePolyPass(tohpe)
        pm = clifft.HirPassManager()
        pm.add(p)
        pm.run(hir)
        removed, blocks = p.tohpe_removed, p.tohpe_blocks
    return int(hir.num_t_gates), removed, blocks


# Mirrors a subset of the C++ harness tools/bench/tcount/bench_tcount.cc, which
# is the authoritative source (it also reads the real cultivation_d5 fixture and
# computes the commuting-block histogram).
CIRCUITS = [
    ("ccz_single", ccz(0, 1, 2)),
    ("ccz_ladder_6", ccz_ladder(6)),
    ("ccz_complete_6", ccz_complete(6)),  # TOHPE: 20 -> 12 (beyond peephole)
    ("s_empty_4", s_empty(4)),  # TOHPE: 15 -> 0
    ("s_empty_4_minus_full", s_empty(4, skip_full=True)),  # TOHPE: 14 -> 1
    ("toffoli_single", toffoli(0, 1, 2)),  # mixed-type, skipped
    ("toffoli_chain_3", toffoli_chain(3)),  # mixed-type, skipped
]


def main() -> None:
    print("| circuit | no-opt | peephole | +foldA | +TOHPE | TOHPE removed |")
    print("|---|--:|--:|--:|--:|--:|")
    for name, text in CIRCUITS:
        t0, _, _ = t_count(text, False, False, False)
        tp, _, _ = t_count(text, True, False, False)
        tf, _, _ = t_count(text, True, True, False)
        tt, removed, _ = t_count(text, True, True, True)
        print(f"| {name} | {t0} | {tp} | {tf} | {tt} | {removed} |")


if __name__ == "__main__":
    main()
