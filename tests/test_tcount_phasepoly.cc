// Tests for the experimental phase-polynomial T-count pass.

#include "clifft/backend/backend.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/frontend/hir.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/optimizer/tcount_phasepoly_pass.h"
#include "clifft/svm/svm.h"
#include "clifft/util/constants.h"

#include "test_helpers.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <complex>
#include <string>
#include <vector>

using namespace clifft;
using clifft::test::X;
using clifft::test::Z;

static HirModule hir_from(const char* text) {
    return clifft::trace(clifft::parse(text));
}

// Dense statevector of a unitary circuit, optionally running the pass first.
static std::vector<std::complex<double>> statevector(const char* text, bool with_pass) {
    auto hir = hir_from(text);
    if (with_pass) {
        TCountPhasePolyPass pass;
        pass.run(hir);
    }
    auto prog = lower(hir);
    SchrodingerState state(prog.peak_rank, prog.total_meas_slots, prog.num_qubits);
    execute(prog, state);
    return get_statevector(prog, state);
}

// Assert the pass preserves the exact statevector (amplitudes AND global phase).
static void require_equiv(const char* text) {
    auto a = statevector(text, /*with_pass=*/false);
    auto b = statevector(text, /*with_pass=*/true);
    REQUIRE(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        CHECK_THAT(b[i].real(), Catch::Matchers::WithinAbs(a[i].real(), 1e-9));
        CHECK_THAT(b[i].imag(), Catch::Matchers::WithinAbs(a[i].imag(), 1e-9));
    }
}

// =============================================================================
// Structural folding
// =============================================================================

TEST_CASE("PhasePoly: two T fold to a Clifford S (zero T)", "[tcount]") {
    auto hir = hir_from("T 0\nT 0");
    TCountPhasePolyPass pass;
    pass.run(hir);

    REQUIRE(hir.num_t_gates() == 0);
    REQUIRE(hir.ops.size() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::PHASE_ROTATION);
    REQUIRE(pass.blocks() == 1);
    REQUIRE(pass.t_before() == 2);
    REQUIRE(pass.t_after() == 0);
    REQUIRE(pass.t_removed() == 2);
}

TEST_CASE("PhasePoly: three T fold to one T plus Clifford", "[tcount]") {
    auto hir = hir_from("T 0\nT 0\nT 0");
    TCountPhasePolyPass pass;
    pass.run(hir);

    REQUIRE(hir.num_t_gates() == 1);
    REQUIRE(pass.t_before() == 3);
    REQUIRE(pass.t_after() == 1);
}

TEST_CASE("PhasePoly: four T fold to a Clifford Z (zero T)", "[tcount]") {
    auto hir = hir_from("T 0\nT 0\nT 0\nT 0");
    TCountPhasePolyPass pass;
    pass.run(hir);

    REQUIRE(hir.num_t_gates() == 0);
}

TEST_CASE("PhasePoly: eight T vanish to identity", "[tcount]") {
    auto hir = hir_from("T 0\nT 0\nT 0\nT 0\nT 0\nT 0\nT 0\nT 0");
    TCountPhasePolyPass pass;
    pass.run(hir);

    REQUIRE(hir.num_t_gates() == 0);
    REQUIRE(hir.ops.empty());
    REQUIRE(pass.t_removed() == 8);
}

TEST_CASE("PhasePoly: T and T_dag on same axis cancel", "[tcount]") {
    auto hir = hir_from("T 0\nT_DAG 0");
    TCountPhasePolyPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.empty());
}

TEST_CASE("PhasePoly: two commuting axes fold independently", "[tcount]") {
    // Z(0) and Z(1) commute -> one block of 4, two axis groups, each c=2.
    auto hir = hir_from("T 0\nT 1\nT 0\nT 1");
    TCountPhasePolyPass pass;
    pass.run(hir);

    REQUIRE(hir.num_t_gates() == 0);
    REQUIRE(pass.blocks() == 1);
    REQUIRE(pass.max_block_axes() == 2);
    REQUIRE(pass.t_removed() == 4);
}

TEST_CASE("PhasePoly: anti-commuting axis blocks folding", "[tcount]") {
    // H rotates the middle T to the X axis, which anti-commutes with Z.
    // The three T gates split into singleton blocks and cannot fold.
    auto hir = hir_from("T 0\nH 0\nT 0\nH 0\nT 0");
    TCountPhasePolyPass pass;
    pass.run(hir);

    REQUIRE(hir.num_t_gates() == 3);
    REQUIRE(pass.blocks() == 0);
}

TEST_CASE("PhasePoly: single T is untouched", "[tcount]") {
    auto hir = hir_from("T 0");
    TCountPhasePolyPass pass;
    pass.run(hir);

    REQUIRE(hir.num_t_gates() == 1);
    REQUIRE(pass.blocks() == 0);
}

TEST_CASE("PhasePoly: multi-qubit ZZ axis folds", "[tcount]") {
    // CX entangles; T on qubit 1 then acts on a ZZ Pauli axis. Two such fold.
    auto hir = hir_from("CX 0 1\nT 1\nT 1");
    TCountPhasePolyPass pass;
    pass.run(hir);

    REQUIRE(hir.num_t_gates() == 0);
    REQUIRE(pass.blocks() == 1);
}

// =============================================================================
// Global-phase correctness (manual HIR, negative-sign axes)
// =============================================================================

TEST_CASE("PhasePoly: negative-sign T pair preserves global phase", "[tcount]") {
    // Two T on -Z(0): T(-Z) = exp(+i*pi/4) T_dag(+Z); the pair contributes a
    // net global phase of i and folds to S_dag.
    HirModule hir(1, /*pauli_capacity=*/16);
    clifft::test::append_tgate(hir, 0, Z(0), /*sign=*/true);
    clifft::test::append_tgate(hir, 0, Z(0), /*sign=*/true);
    auto initial_weight = hir.global_weight;

    TCountPhasePolyPass pass;
    pass.run(hir);

    REQUIRE(hir.num_t_gates() == 0);
    REQUIRE(hir.ops.size() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::PHASE_ROTATION);

    auto ratio = hir.global_weight / initial_weight;
    CHECK_THAT(ratio.real(), Catch::Matchers::WithinAbs(0.0, 1e-12));
    CHECK_THAT(ratio.imag(), Catch::Matchers::WithinAbs(1.0, 1e-12));
}

// =============================================================================
// Semantic equivalence via dense statevector (unitary circuits)
// =============================================================================

TEST_CASE("PhasePoly: statevector preserved on single-qubit T powers", "[tcount]") {
    require_equiv("H 0\nT 0\nT 0");
    require_equiv("H 0\nT 0\nT 0\nT 0");
    require_equiv("H 0\nT 0\nT 0\nT 0\nT 0\nT 0");
}

TEST_CASE("PhasePoly: statevector preserved on entangled axes", "[tcount]") {
    require_equiv("H 0\nH 1\nCX 0 1\nT 1\nT 1\nT 1");
    require_equiv("H 0\nT 0\nT 1\nT 0\nT 1\nCX 0 1");
}

TEST_CASE("PhasePoly: statevector preserved with interleaved different axes", "[tcount]") {
    require_equiv("H 0\nH 1\nH 2\nT 0\nT 1\nT 2\nT 0\nT 1\nT 2");
}

// =============================================================================
// Phase B (TOHPE) end-to-end: a real multi-axis reduction with the diagonal
// unitary verified exactly. This is the evidence the issue asks for -- TOHPE
// removing T gates that phase folding / peephole cannot, semantics preserved.
// =============================================================================

// "T on Z-parity a" (a's set bits), computed onto the lowest support qubit by a
// CNOT ladder so the front end records a single virtual Z-parity T.
static std::string t_on_parity(uint32_t a, int nq) {
    std::vector<int> bits;
    for (int q = 0; q < nq; ++q)
        if (a & (1u << q))
            bits.push_back(q);
    int tgt = bits[0];
    std::string s;
    for (size_t k = 1; k < bits.size(); ++k)
        s += "CX " + std::to_string(bits[k]) + " " + std::to_string(tgt) + "\n";
    s += "T " + std::to_string(tgt) + "\n";
    for (size_t k = bits.size(); k-- > 1;)
        s += "CX " + std::to_string(bits[k]) + " " + std::to_string(tgt) + "\n";
    return s;
}

// Diagonal phase function f(x) mod 8 of a purely Z-type HIR (T_GATE and
// PHASE_ROTATION ops only), evaluated over all 2^nq inputs. This is the exact
// unitary the block implements, independent of how it is factored.
static std::vector<int> hir_phase_fn(const HirModule& hir, int nq) {
    std::vector<int> f(1u << nq, 0);
    for (const auto& op : hir.ops) {
        int coeff;
        if (op.op_type() == OpType::T_GATE) {
            coeff = op.is_dagger() ? -1 : 1;
        } else if (op.op_type() == OpType::PHASE_ROTATION) {
            coeff = static_cast<int>(std::llround(op.alpha() * 4.0));
        } else {
            continue;
        }
        REQUIRE(hir.destab_mask(op).is_zero());  // Z-only
        REQUIRE(!hir.sign(op));
        uint64_t parity = hir.stab_mask(op).words[0];
        for (int x = 0; x < (1 << nq); ++x) {
            int bit = static_cast<int>(__builtin_popcountll(parity & static_cast<uint64_t>(x)) & 1);
            f[x] = (((f[x] + coeff * bit) % 8) + 8) % 8;
        }
    }
    return f;
}

TEST_CASE("PhasePoly TOHPE: S_empty (15 parities) reduces, diagonal preserved", "[tcount][tohpe]") {
    // Amy-Maslov-Mosca arXiv:1303.2042: T on every nonzero parity of F_2^4 is
    // the identity (f = 0). Phase folding removes none (15 distinct axes); TOHPE
    // removes T gates while preserving the exact diagonal unitary.
    std::string text;
    for (uint32_t a = 1; a < 16; ++a)
        text += t_on_parity(a, 4);

    auto hir = hir_from(text.c_str());
    REQUIRE(hir.num_t_gates() == 15);
    auto f_before = hir_phase_fn(hir, 4);

    TCountPhasePolyPass pass;
    pass.run(hir);

    INFO("t_after=" << hir.num_t_gates() << " tohpe_removed=" << pass.tohpe_removed());
    REQUIRE(hir.num_t_gates() < 15);  // genuine multi-axis reduction
    REQUIRE(pass.tohpe_removed() > 0);
    REQUIRE(pass.tohpe_blocks() >= 1);
    REQUIRE(hir_phase_fn(hir, 4) == f_before);  // exact semantics
}

TEST_CASE("PhasePoly TOHPE: 14-parity block collapses to one T, diagonal preserved",
          "[tcount][tohpe]") {
    // The 15 parities minus Z_0123 implement a single T_dag on Z_0^Z_1^Z_2^Z_3:
    // a nontrivial diagonal unitary that TOHPE should reduce from 14 T to ~1 T.
    std::string text;
    for (uint32_t a = 1; a < 15; ++a)  // skip a = 15 (Z_0123)
        text += t_on_parity(a, 4);

    auto hir = hir_from(text.c_str());
    REQUIRE(hir.num_t_gates() == 14);
    auto f_before = hir_phase_fn(hir, 4);
    // Confirm the target is the nontrivial single-T_dag phase.
    for (int x = 0; x < 16; ++x) {
        int p = __builtin_popcount(x & 0xF) & 1;
        REQUIRE(f_before[x] == (p ? 7 : 0));
    }

    TCountPhasePolyPass pass;
    pass.run(hir);

    INFO("t_after=" << hir.num_t_gates() << " tohpe_removed=" << pass.tohpe_removed());
    REQUIRE(hir.num_t_gates() < 14);
    REQUIRE(hir.num_t_gates() >= 1);
    REQUIRE(pass.tohpe_removed() > 0);
    REQUIRE(hir_phase_fn(hir, 4) == f_before);  // exact semantics
}

// CCZ(a,b,c) as its 7-term phase polynomial (+singles, -pairs, +triple).
static std::string ccz_str(int a, int b, int c) {
    auto par = [](uint32_t mask, bool dag) {
        std::vector<int> bits;
        for (int q = 0; q < 12; ++q)
            if (mask & (1u << q))
                bits.push_back(q);
        int t = bits[0];
        std::string s;
        for (size_t k = 1; k < bits.size(); ++k)
            s += "CX " + std::to_string(bits[k]) + " " + std::to_string(t) + "\n";
        s += (dag ? "T_DAG " : "T ") + std::to_string(t) + "\n";
        for (size_t k = bits.size(); k-- > 1;)
            s += "CX " + std::to_string(bits[k]) + " " + std::to_string(t) + "\n";
        return s;
    };
    uint32_t A = 1u << a, B = 1u << b, C = 1u << c;
    return par(A, false) + par(B, false) + par(C, false) + par(A | B, true) + par(A | C, true) +
           par(B | C, true) + par(A | B | C, false);
}

TEST_CASE("PhasePoly TOHPE: dense CCZ-complete block reduces beyond peephole, exact",
          "[tcount][tohpe]") {
    // All C(6,3)=20 CCZ triples on 6 qubits: a dense diagonal phase polynomial.
    // Peephole/folding reach 20 T; faithful TOHPE removes more (a real
    // ancilla-free multi-axis reduction on structured input), exactly.
    std::string text;
    for (int a = 0; a < 6; ++a)
        for (int b = a + 1; b < 6; ++b)
            for (int c = b + 1; c < 6; ++c)
                text += ccz_str(a, b, c);

    auto hir_peep = hir_from(text.c_str());
    PeepholeFusionPass().run(hir_peep);
    size_t t_peephole = hir_peep.num_t_gates();
    auto f_before = hir_phase_fn(hir_peep, 6);

    auto hir = hir_from(text.c_str());
    PeepholeFusionPass().run(hir);
    TCountPhasePolyPass pass;
    pass.run(hir);

    INFO("t_peephole=" << t_peephole << " t_after=" << hir.num_t_gates()
                       << " tohpe_removed=" << pass.tohpe_removed());
    REQUIRE(hir.num_t_gates() < t_peephole);  // TOHPE beats peephole here
    REQUIRE(pass.tohpe_removed() > 0);
    REQUIRE(hir_phase_fn(hir, 6) == f_before);  // exact diagonal unitary
}

TEST_CASE("PhasePoly TOHPE: mixed-type blocks are left untouched and exact", "[tcount][tohpe]") {
    // Phase B reduces only single-Pauli-type (diagonal) blocks, where f(x) mod 8
    // is the exact unitary. A block whose axes mix X and Z -- here s_empty
    // conjugated by a Clifford -- must be left for plain folding; reducing it
    // would need quadratic-Clifford / global-phase tracking the f-check does not
    // provide. This pins the regression a bell-mixed conjugation exposed.
    std::string inner;
    for (uint32_t a = 1; a < 16; ++a)
        inner += t_on_parity(a, 4);  // s_empty(4): trivial diagonal phase

    const std::string h_only = "H 0\n" + inner + "H 0\n";
    const std::string bell = "CX 0 1\nH 0\n" + inner + "H 0\nCX 0 1\n";

    for (const auto& text : {h_only, bell}) {
        auto hir = hir_from(text.c_str());
        TCountPhasePolyPass pass;
        pass.run(hir);
        REQUIRE(pass.tohpe_removed() == 0);  // mixed block skipped, not reduced
        require_equiv(text.c_str());         // unitary preserved exactly
    }
}

// Every source line referenced by the current mapping, as a flat list.
static std::vector<uint32_t> all_source_lines(const HirModule& hir) {
    std::vector<uint32_t> v;
    for (const auto& lines : hir.source_map)
        for (uint32_t ln : lines)
            v.push_back(ln);
    return v;
}

TEST_CASE("PhasePoly: source map stays consistent after reduction", "[tcount][sourcemap]") {
    // T on 14 of the 15 nonzero parities of F_2^4: TOHPE collapses it to a
    // single T, deleting most ops and re-emitting the survivor as a different
    // Pauli. The source map must stay parallel to ops, reference only lines the
    // input already had, and consolidate the deleted ops' lines onto a survivor.
    std::string text;
    for (uint32_t a = 1; a < 15; ++a)  // skip a == 15 (full parity)
        text += t_on_parity(a, 4);

    auto hir = hir_from(text.c_str());
    REQUIRE(hir.source_map.size() == hir.ops.size());
    const std::vector<uint32_t> orig = all_source_lines(hir);
    const size_t t_before = hir.num_t_gates();

    TCountPhasePolyPass().run(hir);

    REQUIRE(hir.num_t_gates() < t_before);             // the block actually reduced
    REQUIRE(hir.source_map.size() == hir.ops.size());  // parallel after compaction
    size_t max_entries = 0;
    for (const auto& lines : hir.source_map) {
        for (uint32_t ln : lines)
            REQUIRE(std::find(orig.begin(), orig.end(), ln) != orig.end());  // no new line
        max_entries = std::max(max_entries, lines.size());
    }
    // The block's deleted ops are not dropped: their lines are consolidated onto
    // a surviving op, so at least one op carries more than its own single line.
    // (Without the reattach every op would map to exactly one line.)
    REQUIRE(max_entries >= 2);
}

TEST_CASE("PhasePoly: source map consistent after pure folding", "[tcount][sourcemap]") {
    // Three T on one axis fold to one T plus an S. No TOHPE fires, so this
    // exercises the folding re-emit path's source-map handling.
    auto hir = hir_from("T 0\nT 0\nT 0");
    REQUIRE(hir.source_map.size() == hir.ops.size());
    const std::vector<uint32_t> orig = all_source_lines(hir);

    TCountPhasePolyPass().run(hir);

    REQUIRE(hir.source_map.size() == hir.ops.size());  // parallel after compaction
    for (const auto& lines : hir.source_map)
        for (uint32_t ln : lines)
            REQUIRE(std::find(orig.begin(), orig.end(), ln) != orig.end());  // no new line
}

TEST_CASE("PhasePoly: max_verify_bits gates Phase B", "[tcount][tohpe]") {
    // S_empty(4) spans support 4. A verify cutoff below 4 leaves Phase B unable
    // to check the move, so the block is returned unchanged; the default reduces
    // it. This pins down that the parameter actually controls the behaviour.
    std::string text;
    for (uint32_t a = 1; a < 16; ++a)
        text += t_on_parity(a, 4);

    auto low = hir_from(text.c_str());
    TCountPhasePolyPass(/*enable_tohpe=*/true, /*max_verify_bits=*/3).run(low);
    REQUIRE(low.num_t_gates() == 15);  // cutoff below the support: no reduction

    auto def = hir_from(text.c_str());
    TCountPhasePolyPass(/*enable_tohpe=*/true, /*max_verify_bits=*/14).run(def);
    REQUIRE(def.num_t_gates() < 15);  // default cutoff: collapses the block
}
