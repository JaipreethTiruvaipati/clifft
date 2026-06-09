// Scientific evaluation harness for the experimental global T-count pass.
//
// Reports, per circuit and per optimization phase, the T-count under:
//   no-opt | peephole | +Phase A folding | +Phase B TOHPE
// plus the commuting-block-size distribution (which explains where the
// multi-axis reducer can fire) and an exactness check.
//
// Circuits are Clifford+T text in Clifft's Stim superset. Toffoli/CCZ are
// hand-decomposed (the parser has no CCX/CCZ): CCZ stays Z-diagonal, while a
// Toffoli's internal Hadamards make its block mixed-type after Clifford
// absorption -- the central limitation this evaluation quantifies.

#include "clifft/backend/backend.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/frontend/hir.h"
#include "clifft/optimizer/commutation.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/optimizer/tcount_phasepoly_pass.h"
#include "clifft/svm/svm.h"

#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace clifft;

// ---------------------------------------------------------------------------
// Circuit generators (Stim-superset text)
// ---------------------------------------------------------------------------

// T (or T_dag) on the Z-parity given by the set bits of `mask`, computed onto
// the lowest support qubit via a CNOT ladder so the front end records one
// virtual Z-parity rotation.
static std::string parity_phase(uint64_t mask, bool dag) {
    std::vector<int> bits;
    for (int q = 0; q < 64; ++q)
        if (mask & (1ULL << q))
            bits.push_back(q);
    int tgt = bits.front();
    std::string s;
    for (size_t k = 1; k < bits.size(); ++k)
        s += "CX " + std::to_string(bits[k]) + " " + std::to_string(tgt) + "\n";
    s += (dag ? "T_DAG " : "T ") + std::to_string(tgt) + "\n";
    for (size_t k = bits.size(); k-- > 1;)
        s += "CX " + std::to_string(bits[k]) + " " + std::to_string(tgt) + "\n";
    return s;
}

// CCZ(a,b,c) phase polynomial: +singles, -pairs, +triple (7 T gates), diagonal.
static std::string ccz(int a, int b, int c) {
    auto B = [](int q) { return uint64_t{1} << q; };
    std::string s;
    s += parity_phase(B(a), false);
    s += parity_phase(B(b), false);
    s += parity_phase(B(c), false);
    s += parity_phase(B(a) | B(b), true);
    s += parity_phase(B(a) | B(c), true);
    s += parity_phase(B(b) | B(c), true);
    s += parity_phase(B(a) | B(b) | B(c), false);
    return s;
}

// k overlapping CCZ gates CCZ(i, i+1, i+2): a real diagonal phase-polynomial
// network (Hamming-weight phasing / IQP style) with a large commuting Z-block.
static std::string ccz_ladder(int k) {
    std::string s;
    for (int i = 0; i < k; ++i)
        s += ccz(i, i + 1, i + 2);
    return s;
}

// Toffoli(a,b,c) = H c; CCZ(a,b,c); H c. The Hadamards make the traced block
// mixed-type, so the single-type reducer skips it.
static std::string toffoli(int a, int b, int c) {
    return "H " + std::to_string(c) + "\n" + ccz(a, b, c) + "H " + std::to_string(c) + "\n";
}

static std::string toffoli_chain(int k) {
    std::string s;
    for (int i = 0; i < k; ++i)
        s += toffoli(i, i + 1, i + 2);
    return s;
}

// All nonzero parities of F_2^nq (Amy-Maslov-Mosca S_empty when nq=4: identity).
static std::string s_empty(int nq, bool skip_full = false) {
    std::string s;
    for (uint64_t a = 1; a < (1ULL << nq); ++a) {
        if (skip_full && a == (1ULL << nq) - 1)
            continue;
        s += parity_phase(a, false);
    }
    return s;
}

// All C(nq,3) CCZ triples on nq qubits: a dense diagonal phase polynomial with
// heavy shared cubic structure, the regime where TODD/TOHPE is expected to win.
static std::string ccz_complete(int nq) {
    std::string s;
    for (int a = 0; a < nq; ++a)
        for (int b = a + 1; b < nq; ++b)
            for (int c = b + 1; c < nq; ++c)
                s += ccz(a, b, c);
    return s;
}

// CCZ(0, 1, j) for j = 2..k+1: a fan of CCZ gates sharing the control pair
// {0, 1}, so the parities Z_0, Z_1, Z_0^Z_1 and the pairwise terms recur.
static std::string ccz_star(int k) {
    std::string s;
    for (int j = 2; j < 2 + k; ++j)
        s += ccz(0, 1, j);
    return s;
}

// Conjugate a circuit by a layer of Hadamards on `qs`. Conjugating a diagonal
// (Z-type) phase polynomial this way rotates the parities that touch those
// qubits into the X plane, making the commuting block MIXED-type while
// preserving its reduction structure -- the test bed for the mixed-type path.
static std::string h_conjugate(const std::string& inner, const std::vector<int>& qs) {
    std::string h;
    for (int q : qs)
        h += "H " + std::to_string(q) + "\n";
    return h + inner + h;
}

// Deterministic pseudo-random Clifford+T circuit (typical workload).
static std::string random_clifford_t(int nq, int depth, uint64_t seed) {
    const char* g1[] = {"H", "S", "S_DAG", "X", "Z"};
    std::string s;
    auto next = [&]() {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        return seed >> 33;
    };
    for (int d = 0; d < depth; ++d) {
        uint64_t r = next() % 100;
        if (r < 25 && nq >= 2) {
            int a = static_cast<int>(next() % nq), b = static_cast<int>(next() % nq);
            if (a == b)
                b = (b + 1) % nq;
            s += "CX " + std::to_string(a) + " " + std::to_string(b) + "\n";
        } else if (r < 50) {
            s += "T " + std::to_string(next() % nq) + "\n";
        } else {
            s += std::string(g1[next() % 5]) + " " + std::to_string(next() % nq) + "\n";
        }
    }
    return s;
}

// ---------------------------------------------------------------------------
// Measurement helpers
// ---------------------------------------------------------------------------

static size_t t_after(const std::string& text, bool peephole, bool fold, bool tohpe,
                      size_t* removed = nullptr, size_t* tblocks = nullptr) {
    HirModule hir = trace(parse(text));
    if (peephole)
        PeepholeFusionPass().run(hir);
    if (fold) {
        TCountPhasePolyPass pass(tohpe);
        pass.run(hir);
        if (removed)
            *removed = pass.tohpe_removed();
        if (tblocks)
            *tblocks = pass.tohpe_blocks();
    }
    return hir.num_t_gates();
}

// Commuting-block-size histogram after peephole: sizes and how many blocks are
// single Pauli type (Z-only or X-only) -- the ones the reducer can act on.
struct BlockStats {
    std::map<size_t, size_t> size_hist;
    size_t single_type_blocks = 0;
    size_t multi_type_blocks = 0;
    size_t largest = 0;
};

static BlockStats block_stats(const std::string& text) {
    HirModule hir = trace(parse(text));
    PeepholeFusionPass().run(hir);
    BlockStats bs;
    const size_t n = hir.ops.size();
    size_t i = 0;
    while (i < n) {
        if (hir.ops[i].op_type() != OpType::T_GATE) {
            ++i;
            continue;
        }
        size_t end = i + 1;
        while (end < n && hir.ops[end].op_type() == OpType::T_GATE) {
            bool ok = true;
            for (size_t k = i; k < end; ++k)
                if (anti_commute(hir.destab_mask(hir.ops[end]), hir.stab_mask(hir.ops[end]),
                                 hir.destab_mask(hir.ops[k]), hir.stab_mask(hir.ops[k]))) {
                    ok = false;
                    break;
                }
            if (!ok)
                break;
            ++end;
        }
        size_t sz = end - i;
        if (sz >= 2) {
            bs.size_hist[sz]++;
            bs.largest = std::max(bs.largest, sz);
            bool all_z = true, all_x = true;
            for (size_t k = i; k < end; ++k) {
                if (!hir.destab_mask(hir.ops[k]).is_zero())
                    all_z = false;
                if (!hir.stab_mask(hir.ops[k]).is_zero())
                    all_x = false;
            }
            if (all_z || all_x)
                bs.single_type_blocks++;
            else
                bs.multi_type_blocks++;
        }
        i = end;
    }
    return bs;
}

// Exact equivalence for unitary circuits with n <= 10 (dense statevector).
static bool statevector_equiv(const std::string& text, int nq) {
    if (nq > 10)
        return true;  // not checked here
    auto sv = [&](bool opt) {
        HirModule hir = trace(parse(text));
        if (opt) {
            PeepholeFusionPass().run(hir);
            TCountPhasePolyPass(true).run(hir);
        }
        CompiledModule prog = lower(hir);
        SchrodingerState st(prog.peak_rank, prog.total_meas_slots, prog.num_qubits);
        execute(prog, st);
        return get_statevector(prog, st);
    };
    auto a = sv(false), b = sv(true);
    if (a.size() != b.size())
        return false;
    std::complex<double> ip = 0;
    double na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        ip += std::conj(a[i]) * b[i];
        na += std::norm(a[i]);
        nb += std::norm(b[i]);
    }
    double fid = (na > 0 && nb > 0) ? std::norm(ip) / (na * nb) : 1.0;
    return std::abs(fid - 1.0) < 1e-9;
}

// ---------------------------------------------------------------------------

struct BenchCircuit {
    std::string name;
    std::string text;
    int nq;
};

int main() {
    std::vector<BenchCircuit> circuits;
    circuits.push_back({"ccz_single", ccz(0, 1, 2), 3});
    circuits.push_back({"ccz_ladder_2", ccz_ladder(2), 4});
    circuits.push_back({"ccz_ladder_3", ccz_ladder(3), 5});
    circuits.push_back({"ccz_ladder_4", ccz_ladder(4), 6});
    circuits.push_back({"ccz_ladder_6", ccz_ladder(6), 8});
    circuits.push_back({"ccz_ladder_10", ccz_ladder(10), 12});
    circuits.push_back({"ccz_complete_4", ccz_complete(4), 4});
    circuits.push_back({"ccz_complete_5", ccz_complete(5), 5});
    circuits.push_back({"ccz_complete_6", ccz_complete(6), 6});
    circuits.push_back({"ccz_star_5", ccz_star(5), 7});
    circuits.push_back({"ccz_star_8", ccz_star(8), 10});
    circuits.push_back({"ccz_complete_6_hmixed", h_conjugate(ccz_complete(6), {0, 1, 2}), 6});
    circuits.push_back({"s_empty_4", s_empty(4), 4});
    circuits.push_back({"s_empty_5", s_empty(5), 5});
    circuits.push_back({"s_empty_4_minus_full", s_empty(4, true), 4});
    circuits.push_back({"toffoli_single", toffoli(0, 1, 2), 3});
    circuits.push_back({"toffoli_chain_3", toffoli_chain(3), 5});
    circuits.push_back({"random_6q_d120", random_clifford_t(6, 120, 42), 6});
    circuits.push_back({"random_8q_d200", random_clifford_t(8, 200, 7), 8});

    // Real fixture (magic-state cultivation, distance 5).
    {
        std::ifstream f(std::string(CLIFFT_FIXTURES_DIR) + "/cultivation_d5.stim");
        if (f) {
            std::stringstream ss;
            ss << f.rdbuf();
            circuits.push_back({"cultivation_d5", ss.str(), 26});
        }
    }

    std::cout << "## Per-phase T-count (ancilla-free)\n\n";
    std::cout << "| circuit | n | no-opt | peephole | +foldA | +TOHPE | TOHPE-removed | "
                 "tohpe-blocks | equiv |\n";
    std::cout << "|---|--:|--:|--:|--:|--:|--:|--:|:-:|\n";
    for (auto& c : circuits) {
        size_t removed = 0, tblk = 0;
        size_t t0 = t_after(c.text, false, false, false);
        size_t tp = t_after(c.text, true, false, false);
        size_t tf = t_after(c.text, true, true, false);
        size_t tt = t_after(c.text, true, true, true, &removed, &tblk);
        bool eq = statevector_equiv(c.text, c.nq);
        std::cout << "| " << c.name << " | " << c.nq << " | " << t0 << " | " << tp << " | " << tf
                  << " | " << tt << " | " << removed << " | " << tblk << " | "
                  << (c.nq <= 10 ? (eq ? "OK" : "FAIL") : "n/a") << " |\n";
    }

    std::cout << "\n## Commuting-block structure after peephole (why TOHPE fires or not)\n\n";
    std::cout << "| circuit | #blocks>=2 | single-type | mixed-type | largest |\n";
    std::cout << "|---|--:|--:|--:|--:|\n";
    for (auto& c : circuits) {
        BlockStats bs = block_stats(c.text);
        size_t nb = bs.single_type_blocks + bs.multi_type_blocks;
        std::cout << "| " << c.name << " | " << nb << " | " << bs.single_type_blocks << " | "
                  << bs.multi_type_blocks << " | " << bs.largest << " |\n";
    }
    return 0;
}
