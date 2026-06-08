// Unit tests for the GF(2) TOHPE T-count reducer (pure linear algebra, no HIR).

#include "clifft/optimizer/tcount_tohpe.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <vector>

using namespace clifft;

// Build a single-word ParityColumn from a bitmask.
static ParityColumn col(uint64_t bits) {
    return ParityColumn{std::vector<uint64_t>{bits}};
}

// Reference phase function f(x) = (sum_j [col_j . x]) mod 8 over all x.
static std::vector<int> phase_fn(const std::vector<ParityColumn>& cols,
                                 const std::vector<ResidualPhase>& res, uint32_t n_bits) {
    std::vector<int> f(1ULL << n_bits, 0);
    for (uint64_t x = 0; x < (1ULL << n_bits); ++x) {
        int acc = 0;
        for (auto& c : cols)
            acc += static_cast<int>(__builtin_popcountll(c.words[0] & x) & 1ULL);
        for (auto& r : res)
            acc +=
                r.coeff_mod8 * static_cast<int>(__builtin_popcountll(r.parity.words[0] & x) & 1ULL);
        f[x] = ((acc % 8) + 8) % 8;
    }
    return f;
}

TEST_CASE("TOHPE: empty and singleton are unchanged", "[tcount][tohpe]") {
    auto r0 = tohpe_reduce({}, 3);
    REQUIRE(r0.t_after == 0);

    auto r1 = tohpe_reduce({col(0b101)}, 3);
    REQUIRE(r1.t_after == 1);
}

TEST_CASE("TOHPE: duplicate columns cancel to an S residual", "[tcount][tohpe]") {
    // Two T on the same parity = S on that parity: 2 T -> 0 T.
    auto r = tohpe_reduce({col(0b011), col(0b011)}, 3);
    REQUIRE(r.t_before == 2);
    REQUIRE(r.t_after == 0);
    // Residual reproduces the phase exactly.
    REQUIRE(phase_fn(r.columns, r.residuals, 3) == phase_fn({col(0b011), col(0b011)}, {}, 3));
}

TEST_CASE("TOHPE: single CCZ (7 parities, n=3) is already optimal", "[tcount][tohpe]") {
    // CCZ on 3 qubits = T on all 7 nonzero parities of F_2^3. Amy-Maslov-Mosca:
    // for n < 4 no two distinct-T-count tables are equivalent, so 7 is optimal.
    std::vector<ParityColumn> ccz;
    for (uint64_t a = 1; a < 8; ++a)
        ccz.push_back(col(a));
    auto r = tohpe_reduce(ccz, 3);
    REQUIRE(r.t_after == 7);
}

TEST_CASE("TOHPE: S_empty (all 15 parities of F_2^4) reduces far below 15", "[tcount][tohpe]") {
    // Amy-Maslov-Mosca arXiv:1303.2042: applying T on every nonzero parity of
    // F_2^4 implements the trivial phase (f(x) = 8 = 0 mod 8 for all x), so all
    // 15 T gates are removable. Phase folding alone removes NONE (all 15 axes
    // are distinct), so any reduction here is a genuine multi-axis TODD effect.
    std::vector<ParityColumn> s_empty;
    for (uint64_t a = 1; a < 16; ++a)
        s_empty.push_back(col(a));

    auto f_before = phase_fn(s_empty, {}, 4);
    for (int v : f_before)
        REQUIRE(v == 0);  // confirm the polynomial is trivial

    auto r = tohpe_reduce(s_empty, 4);

    INFO("t_before=" << r.t_before << " t_after=" << r.t_after);
    REQUIRE(r.t_before == 15);
    REQUIRE(r.t_after < 15);  // TOHPE fires beyond folding
    // Exactly semantics-preserving.
    REQUIRE(phase_fn(r.columns, r.residuals, 4) == f_before);
}

TEST_CASE("TOHPE: reduction preserves the phase function on a random-ish block",
          "[tcount][tohpe]") {
    // A 4-qubit block with a redundant cubic relation embedded.
    std::vector<ParityColumn> cols = {col(0b0001), col(0b0010), col(0b0011), col(0b0100),
                                      col(0b0101), col(0b0110), col(0b0111), col(0b1000)};
    auto f_before = phase_fn(cols, {}, 4);
    auto r = tohpe_reduce(cols, 4);
    REQUIRE(r.t_after <= r.t_before);
    REQUIRE(phase_fn(r.columns, r.residuals, 4) == f_before);
}
