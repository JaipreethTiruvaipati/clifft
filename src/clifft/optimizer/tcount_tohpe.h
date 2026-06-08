#pragma once

// Ancilla-free TOHPE T-count reducer over a GF(2) gate-synthesis matrix.
//
// Implements the duplicate-and-destroy reduction of Vandaele, "Lower T-count
// with faster algorithms" (arXiv:2407.08695, 2024), Algorithm 2 / Theorem 1,
// which is the current state of the art and supersedes the TODD compiler of
// Heyfron and Campbell (arXiv:1712.01557, 2018). The reduction is grounded in
// the equivalence between T-count optimization and 3rd-order symmetric tensor
// rank / Reed-Muller decoding (Amy and Mosca, arXiv:1601.07363, 2019).
//
// This module is pure GF(2) linear algebra over a parity table and is unit
// tested in isolation (tests/test_tcount_tohpe.cc). It has no dependency on the
// HIR; the pass adapts a commuting block of Z-type T axes into a ParityTable,
// calls reduce(), and reads back the reduced columns plus the single-axis S
// residuals.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace clifft {

// A column of the gate-synthesis matrix: an n_bits-wide GF(2) parity vector,
// packed little-endian into 64-bit words.
struct ParityColumn {
    std::vector<uint64_t> words;

    bool get(uint32_t i) const { return (words[i / 64] >> (i % 64)) & 1ULL; }
    void set(uint32_t i, bool v) {
        uint64_t m = 1ULL << (i % 64);
        if (v)
            words[i / 64] |= m;
        else
            words[i / 64] &= ~m;
    }
    void xor_with(const ParityColumn& o) {
        for (size_t w = 0; w < words.size(); ++w)
            words[w] ^= o.words[w];
    }
    bool is_zero() const {
        for (uint64_t w : words)
            if (w)
                return false;
        return true;
    }
    bool operator==(const ParityColumn& o) const { return words == o.words; }
};

// An S/S_dag (eighth-turn coefficient 2 or 6) phase left on a single parity by
// the duplicate-destroy step. coeff is in Z_8 (always even: 2, 4, or 6).
struct ResidualPhase {
    ParityColumn parity;
    int coeff_mod8;  // even: 2 = S, 4 = Z, 6 = S_dag
};

// Result of a TOHPE reduction.
struct TohpeResult {
    std::vector<ParityColumn> columns;     // reduced odd-coefficient T parities
    std::vector<ResidualPhase> residuals;  // Clifford phases to re-absorb
    size_t t_before = 0;
    size_t t_after = 0;
};

// Reduce the T-count of the phase polynomial whose odd-coefficient parities are
// `columns` (each an n_bits-wide ParityColumn). Preserves the implemented
// diagonal unitary exactly: the returned columns plus residuals reproduce the
// same phase function. `n_bits` is the parity width; `max_cols` caps the work
// (TOHPE is polynomial but the cap keeps the prototype bounded on large blocks;
// blocks above the cap are returned unchanged).
TohpeResult tohpe_reduce(std::vector<ParityColumn> columns, uint32_t n_bits, size_t max_cols = 256);

}  // namespace clifft
