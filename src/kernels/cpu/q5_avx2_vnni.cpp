// src/kernels/cpu/q5_avx2_vnni.cpp - Q5_0 down rows on AVX-VNNI, several tokens per pass (see q5_avx2.hpp).
// Compiled with the AVX-VNNI flags; only called behind cpu_avxvnni_ok().
#include "basalt/kernels/cpu/q5_avx2.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "basalt/ggml/ggml_common.h"

#include <immintrin.h>

#include <cstdlib>
#include <cstring>

namespace basalt::kernels::cpu {
namespace {

static_assert(sizeof(block_q5_0) == 22, "Q5_0 block: d (2), qh (4), qs (16)");
static_assert(sizeof(block_q8_0) == 34, "Q8_0 block: d (2), qs (32)");

constexpr int prefetch_ahead = 2048;

inline void prefetch_step(const uint8_t*& pf, const uint8_t* end) {
    if (pf < end) {
        _mm_prefetch((const char*) pf, _MM_HINT_T0);
        pf += 64;
    }
}

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }

// ggml's hsum_float_8, operation for operation: the final sum must round as the ggml dot's does.
inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// ggml's bytes_from_nibbles_32: the low nibbles of qs[0..15] are values 0..15, the high nibbles values 16..31.
inline __m256i nibbles32(const uint8_t* qs) {
    const __m128i tmp = _mm_loadu_si128((const __m128i*) qs);
    const __m256i bytes = _mm256_set_m128i(_mm_srli_epi16(tmp, 4), tmp);
    return _mm256_and_si256(_mm256_set1_epi8(0xF), bytes);
}

// ggml's bytes_from_bits_32: byte j is 0xFF where bit j of qh is set.
inline __m256i bits32(const uint8_t* qh) {
    uint32_t x32;
    std::memcpy(&x32, qh, sizeof(x32));
    const __m256i shuf = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32((int) x32), shuf);
    bytes = _mm256_or_si256(bytes, _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe));
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}

// The weight's 32 values unsigned, 0..31: nibble | (high bit << 4).  ggml uses the same values minus 16.
inline __m256i weight_u5(const uint8_t* blk) {
    const __m256i hi = _mm256_and_si256(bits32(blk + 2), _mm256_set1_epi8(0x10));
    return _mm256_or_si256(nibbles32(blk + 6), hi);
}

// R rows by NT tokens: each weight block is decoded once and summed against every token with one vpdpbusd, seeded
// with -16 * (the token's 4-byte lane sums) so the lane holds sum((w - 16) * y), ggml's integer.
template <int R, int NT>
inline void q50_pass(const uint8_t* row0, size_t row_bytes, int nb, const block_q8_0* const* y, const float* dy,
                     const __m256i* seed, float* const* out, int r, const uint8_t*& pf) {
    const uint8_t* const pf_lim = row0 + (size_t) R * row_bytes + prefetch_ahead;
    __m256 acc[R][NT];
    for (int i = 0; i < R; ++i)
        for (int t = 0; t < NT; ++t) acc[i][t] = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        __m256i q8[NT];
        __m256 dyv[NT];
        for (int t = 0; t < NT; ++t) {
            q8[t] = _mm256_loadu_si256((const __m256i*) y[t][ib].qs);
            dyv[t] = _mm256_broadcast_ss(dy + t * q50_max_blocks + ib);
        }
        for (int i = 0; i < R; ++i) {
            prefetch_step(pf, pf_lim);
            const uint8_t* blk = row0 + (size_t) i * row_bytes + (size_t) ib * sizeof(block_q5_0);
            uint16_t dh;
            std::memcpy(&dh, blk, sizeof(dh));
            const __m256 dx = _mm256_set1_ps(h2f(dh));
            const __m256i wu = weight_u5(blk);
            for (int t = 0; t < NT; ++t) {
                const __m256i p = _mm256_dpbusd_avx_epi32(seed[t * q50_max_blocks + ib], wu, q8[t]);
                // ggml: d = fp16(x.d) * fp16(y.d) as one float product, then acc = fma(d, float(lanes), acc)
                acc[i][t] = _mm256_fmadd_ps(_mm256_mul_ps(dx, dyv[t]), _mm256_cvtepi32_ps(p), acc[i][t]);
            }
        }
    }
    for (int i = 0; i < R; ++i)
        for (int t = 0; t < NT; ++t) out[t][r + i] = hsum_float_8(acc[i][t]);
}

template <int R, int NT>
void q50_rows(const uint8_t* w, size_t row_bytes, int n, const block_q8_0* const* y, float* const* out, int r0, int r1) {
    const int nb = n / QK5_0;
    alignas(32) float dy[NT * q50_max_blocks];
    alignas(32) __m256i seed[NT * q50_max_blocks];
    const __m256i sixteen = _mm256_set1_epi8(16);
    for (int t = 0; t < NT; ++t)
        for (int ib = 0; ib < nb; ++ib) {
            dy[t * q50_max_blocks + ib] = h2f(y[t][ib].d);
            const __m256i q8 = _mm256_loadu_si256((const __m256i*) y[t][ib].qs);
            seed[t * q50_max_blocks + ib] =
                _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), sixteen, q8));
        }
    const uint8_t* pf = (const uint8_t*) ((uintptr_t) (w + (size_t) r0 * row_bytes + prefetch_ahead) & ~(uintptr_t) 63);
    int r = r0;
    for (; r + R <= r1; r += R) q50_pass<R, NT>(w + (size_t) r * row_bytes, row_bytes, nb, y, dy, seed, out, r, pf);
    for (; r < r1; ++r) q50_pass<1, NT>(w + (size_t) r * row_bytes, row_bytes, nb, y, dy, seed, out, r, pf);
}

}  // namespace

bool q50_vnni_fits(int n) noexcept { return n > 0 && n % QK5_0 == 0 && n / QK5_0 <= q50_max_blocks; }

bool q50_vnni_enabled() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("BASALT_Q50_VNNI");
        return !(e && e[0] == '0');
    }();
    return on;
}

void q50_down_rows_vnni(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt, float* const* out,
                        int r0, int r1) {
    if (!q50_vnni_fits(n) || nt < 1 || nt > 8) std::abort();   // callers check q50_vnni_fits; nt is at most MAXT (8)
    const block_q8_0* y[8];
    for (int t = 0; t < nt; ++t) y[t] = (const block_q8_0*) hq[t];
    switch (nt) {
        case 1: q50_rows<4, 1>(w, row_bytes, n, y, out, r0, r1); break;
        case 2: q50_rows<2, 2>(w, row_bytes, n, y, out, r0, r1); break;
        case 3: q50_rows<2, 3>(w, row_bytes, n, y, out, r0, r1); break;
        case 4: q50_rows<1, 4>(w, row_bytes, n, y, out, r0, r1); break;
        default:
            for (int t0 = 0; t0 < nt; t0 += 4) {
                const int k = nt - t0 < 4 ? nt - t0 : 4;
                q50_down_rows_vnni(w, row_bytes, n, hq + t0, k, out + t0, r0, r1);
            }
    }
}

}  // namespace basalt::kernels::cpu
