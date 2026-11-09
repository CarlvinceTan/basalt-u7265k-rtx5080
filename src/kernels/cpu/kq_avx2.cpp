// src/kernels/cpu/kq_avx2.cpp - Unsloth UD-Q4_K_XL's expert rows (Q4_K / Q5_K gate/up, Q5_1 and Q8_0 down) for
// several tokens at once, BIT-EXACT against ggml-cpu's own dot products.
//
// ggml-cpu is built for AVX2 here (PORTABLE: GGML_AVX2=ON, GGML_AVX512=OFF), so its vec_dot for these formats
// is the __AVX2__ branch of ggml-cpu/arch/x86/quants.c: ggml_vec_dot_q4_K_q8_K, ggml_vec_dot_q5_K_q8_K,
// ggml_vec_dot_q5_1_q8_1, ggml_vec_dot_q8_0_q8_0.  The functions below are those branches with the loop over tokens
// moved inside the loop over blocks: everything that depends only on the weights (the 4-bit unpacking, the 6-bit
// scales and mins, the fifth bits, the absolute values of the Q8_0 weights, the fp16 scales) is done once per block,
// and every token then runs exactly ggml's integer chain and ggml's float operations in ggml's order - so each
// token's result is ggml's to the bit (native_expert_parity and kq_vnni_test check it), whatever the number of
// tokens.  Unlike the i-quant kernels the group size therefore never changes an answer.
//
// The fork (basalt-u7265k-rtx5080) builds this file with AVX-VNNI and AVX-VNNI-INT8 (Arrow Lake has both): the
// madd + add pairs become vpdpwssd, Q5_1's maddubs + madd becomes vpdpbusd and Q8_0's sign/maddubs/madd chain one
// vpdpbssd - the same exact integers, so the same bits as the AVX2 build.  native_expert.cpp takes these rows only
// from the token counts where they beat ggml's single-token dot (kq_rows_pay).
//
// The gain is the weight-side work and the weight bytes, read once per verify window instead of once per token.
#include "basalt/kernels/cpu/kq_avx2.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "basalt/ggml/ggml_common.h"

#include <immintrin.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace basalt::kernels::cpu {
namespace {

constexpr int MAXT = 8;   // the verify window's tokens per group (kVerifyMaxT)

inline float h2f(ggml_half h) {
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) u)));
}
// the fp16 pair {d, dmin} / {d, m} / {d, s} a block starts with (a union in ggml_common.h's C++ declarations)
inline float half_at(const void* block, int i) {
    ggml_half h;
    std::memcpy(&h, (const uint8_t*) block + 2 * i, 2);
    return h2f(h);
}

// ---- ggml's helpers (ggml-cpu/arch/x86/quants.c), verbatim but for the AVX-VNNI forms
inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
inline __m256i bytes_from_bits_32(const uint8_t* x) {
    uint32_t x32;
    std::memcpy(&x32, x, sizeof(uint32_t));
    const __m256i shuf_mask = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101,
                                                0x0000000000000000);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32((int) x32), shuf_mask);
    const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    bytes = _mm256_or_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}
inline __m256i bytes_from_nibbles_32(const uint8_t* rsi) {
    const __m128i tmp = _mm_loadu_si128((const __m128i*) rsi);
    const __m256i bytes = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), _mm_srli_epi16(tmp, 4), 1);
    const __m256i lowMask = _mm256_set1_epi8(0xF);
    return _mm256_and_si256(lowMask, bytes);
}
inline __m256 sum_i16_pairs_float(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, x));
}
inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
#if defined(__AVXVNNI__)
    return _mm256_cvtepi32_ps(_mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), ax, sy));
#else
    return sum_i16_pairs_float(_mm256_maddubs_epi16(ax, sy));
#endif
}
alignas(32) const uint8_t k_shuffle[256] = {
     0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
     4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
     8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
    12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15};
inline __m256i get_scale_shuffle_k4(int i) { return _mm256_loadu_si256((const __m256i*) k_shuffle + i); }

inline void unpack_q4k_scales(const uint8_t* sc_bytes, __m128i& mins, __m256i& scales) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    uint32_t utmp[4];
    std::memcpy(utmp, sc_bytes, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32((int) utmp[3], (int) utmp[2],
                                                                       (int) utmp[1], (int) utmp[0]));
    mins = _mm256_extracti128_si256(mins_and_scales, 1);
    const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
    scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
}

// ---- Q4_K x Q8_K (ggml_vec_dot_q4_K_q8_K, bit-exact)
void q4k_dot(const block_q4_K* x, int nb, const void* const* act, int nt, float* res) {
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc[MAXT];
    __m128 acc_m[MAXT];
    for (int t = 0; t < nt; ++t) { acc[t] = _mm256_setzero_ps(); acc_m[t] = _mm_setzero_ps(); }
    for (int i = 0; i < nb; ++i) {
        const float xd = half_at(&x[i], 0), xdmin = half_at(&x[i], 1);
        __m128i mins;
        __m256i scales;
        unpack_q4k_scales(x[i].scales, mins, scales);
        __m256i scale_l[4], scale_h[4], q4l[4], q4h[4];
        const uint8_t* q4 = x[i].qs;
        for (int j = 0; j < QK_K / 64; ++j) {
            scale_l[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            scale_h[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
            const __m256i q4bits = _mm256_loadu_si256((const __m256i*) q4);
            q4 += 32;
            q4l[j] = _mm256_and_si256(q4bits, m4);
            q4h[j] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
        }
        for (int t = 0; t < nt; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t] + i;
            const float d = y->d * xd;
            const float dmin = -y->d * xdmin;
            const int8_t* q8 = y->qs;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            acc_m[t] = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m[t]);
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < QK_K / 64; ++j) {
                const __m256i q8l = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                const __m256i p16l = _mm256_maddubs_epi16(q4l[j], q8l);
                const __m256i q8h = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                const __m256i p16h = _mm256_maddubs_epi16(q4h[j], q8h);
#if defined(__AVXVNNI__)
                sumi = _mm256_dpwssd_avx_epi32(sumi, p16l, scale_l[j]);
                sumi = _mm256_dpwssd_avx_epi32(sumi, p16h, scale_h[j]);
#else
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(_mm256_madd_epi16(scale_l[j], p16l),
                                                               _mm256_madd_epi16(scale_h[j], p16h)));
#endif
            }
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) {
        __m128 m = acc_m[t];
        m = _mm_add_ps(m, _mm_movehl_ps(m, m));
        m = _mm_add_ss(m, _mm_movehdup_ps(m));
        res[t] = hsum_float_8(acc[t]) + _mm_cvtss_f32(m);
    }
}

// Fused gate + up for one Q4_K row across `nt` tokens: loads each activation block once for both projections.
void q4k_gu_row(const block_q4_K* xg, const block_q4_K* xu, int nb, const void* const* act, int nt, float* g, float* u) {
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc_g[MAXT], acc_u[MAXT];
    __m128 acc_mg[MAXT], acc_mu[MAXT];
    for (int t = 0; t < nt; ++t) {
        acc_g[t] = _mm256_setzero_ps(); acc_u[t] = _mm256_setzero_ps();
        acc_mg[t] = _mm_setzero_ps(); acc_mu[t] = _mm_setzero_ps();
    }
    for (int i = 0; i < nb; ++i) {
        const float xdg = half_at(&xg[i], 0), xdming = half_at(&xg[i], 1);
        const float xdu = half_at(&xu[i], 0), xdminu = half_at(&xu[i], 1);
        __m128i mins_g, mins_u;
        __m256i scales_g, scales_u;
        unpack_q4k_scales(xg[i].scales, mins_g, scales_g);
        unpack_q4k_scales(xu[i].scales, mins_u, scales_u);
        __m256i sgl[4], sgh[4], qgl[4], qgh[4], sul[4], suh[4], qul[4], quh[4];
        const uint8_t *pg = xg[i].qs, *pu = xu[i].qs;
        for (int j = 0; j < QK_K / 64; ++j) {
            const __m256i sh0 = get_scale_shuffle_k4(2 * j + 0), sh1 = get_scale_shuffle_k4(2 * j + 1);
            sgl[j] = _mm256_shuffle_epi8(scales_g, sh0);
            sgh[j] = _mm256_shuffle_epi8(scales_g, sh1);
            sul[j] = _mm256_shuffle_epi8(scales_u, sh0);
            suh[j] = _mm256_shuffle_epi8(scales_u, sh1);
            const __m256i bg = _mm256_loadu_si256((const __m256i*) (pg + 32 * j));
            const __m256i bu = _mm256_loadu_si256((const __m256i*) (pu + 32 * j));
            qgl[j] = _mm256_and_si256(bg, m4);
            qgh[j] = _mm256_and_si256(_mm256_srli_epi16(bg, 4), m4);
            qul[j] = _mm256_and_si256(bu, m4);
            quh[j] = _mm256_and_si256(_mm256_srli_epi16(bu, 4), m4);
        }
        for (int t = 0; t < nt; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t] + i;
            const float yd = y->d;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            acc_mg[t] = _mm_fmadd_ps(_mm_set1_ps(-yd * xdming), _mm_cvtepi32_ps(_mm_madd_epi16(mins_g, q8s)), acc_mg[t]);
            acc_mu[t] = _mm_fmadd_ps(_mm_set1_ps(-yd * xdminu), _mm_cvtepi32_ps(_mm_madd_epi16(mins_u, q8s)), acc_mu[t]);
            const int8_t* q8 = y->qs;
            __m256i sumg = _mm256_setzero_si256(), sumu = _mm256_setzero_si256();
            for (int j = 0; j < QK_K / 64; ++j) {
                const __m256i q8l = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                const __m256i q8h = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                const __m256i pgl = _mm256_maddubs_epi16(qgl[j], q8l);
                const __m256i pgh = _mm256_maddubs_epi16(qgh[j], q8h);
                const __m256i pul = _mm256_maddubs_epi16(qul[j], q8l);
                const __m256i puh = _mm256_maddubs_epi16(quh[j], q8h);
#if defined(__AVXVNNI__)
                sumg = _mm256_dpwssd_avx_epi32(sumg, pgl, sgl[j]);
                sumg = _mm256_dpwssd_avx_epi32(sumg, pgh, sgh[j]);
                sumu = _mm256_dpwssd_avx_epi32(sumu, pul, sul[j]);
                sumu = _mm256_dpwssd_avx_epi32(sumu, puh, suh[j]);
#else
                sumg = _mm256_add_epi32(sumg, _mm256_add_epi32(_mm256_madd_epi16(sgl[j], pgl), _mm256_madd_epi16(sgh[j], pgh)));
                sumu = _mm256_add_epi32(sumu, _mm256_add_epi32(_mm256_madd_epi16(sul[j], pul), _mm256_madd_epi16(suh[j], puh)));
#endif
            }
            acc_g[t] = _mm256_fmadd_ps(_mm256_set1_ps(yd * xdg), _mm256_cvtepi32_ps(sumg), acc_g[t]);
            acc_u[t] = _mm256_fmadd_ps(_mm256_set1_ps(yd * xdu), _mm256_cvtepi32_ps(sumu), acc_u[t]);
        }
    }
    for (int t = 0; t < nt; ++t) {
        __m128 mg = acc_mg[t];
        mg = _mm_add_ps(mg, _mm_movehl_ps(mg, mg));
        mg = _mm_add_ss(mg, _mm_movehdup_ps(mg));
        g[t] = hsum_float_8(acc_g[t]) + _mm_cvtss_f32(mg);
        __m128 mu = acc_mu[t];
        mu = _mm_add_ps(mu, _mm_movehl_ps(mu, mu));
        mu = _mm_add_ss(mu, _mm_movehdup_ps(mu));
        u[t] = hsum_float_8(acc_u[t]) + _mm_cvtss_f32(mu);
    }
}

// ---- Q5_K x Q8_K (ggml_vec_dot_q5_K_q8_K, bit-exact)
void q5k_dot(const block_q5_K* x, int nb, const void* const* act, int nt, float* res) {
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m128i mzero = _mm_setzero_si128();
    const __m256i mone = _mm256_set1_epi8(1);
    __m256 acc[MAXT];
    float summs[MAXT];
    for (int t = 0; t < nt; ++t) { acc[t] = _mm256_setzero_ps(); summs[t] = 0.f; }
    for (int i = 0; i < nb; ++i) {
        const float xd = half_at(&x[i], 0), xdmin = half_at(&x[i], 1);
        __m128i mins;
        __m256i scales;
        unpack_q4k_scales(x[i].scales, mins, scales);
        const __m256i hbits = _mm256_loadu_si256((const __m256i*) x[i].qh);
        __m256i hmask = mone;
        __m256i scale_0[4], scale_1[4], q5_0[4], q5_1[4];
        const uint8_t* q5 = x[i].qs;
        int bit = 0;
        for (int j = 0; j < QK_K / 64; ++j) {
            scale_0[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            scale_1[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
            const __m256i q5bits = _mm256_loadu_si256((const __m256i*) q5);
            q5 += 32;
            const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
            const __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            q5_0[j] = _mm256_add_epi8(q5l_0, q5h_0);
            hmask = _mm256_slli_epi16(hmask, 1);
            const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
            const __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            q5_1[j] = _mm256_add_epi8(q5l_1, q5h_1);
            hmask = _mm256_slli_epi16(hmask, 1);
        }
        for (int t = 0; t < nt; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t] + i;
            const float d = y->d * xd;
            const float dmin = -y->d * xdmin;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
            summs[t] += dmin * _mm_extract_epi32(hsum, 0);
            const int8_t* q8 = y->qs;
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < QK_K / 64; ++j) {
                const __m256i q8_0 = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                const __m256i p16_0 = _mm256_maddubs_epi16(q5_0[j], q8_0);
                const __m256i p16_1 = _mm256_maddubs_epi16(q5_1[j], q8_1);
#if defined(__AVXVNNI__)
                sumi = _mm256_dpwssd_avx_epi32(sumi, p16_0, scale_0[j]);
                sumi = _mm256_dpwssd_avx_epi32(sumi, p16_1, scale_1[j]);
#else
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(_mm256_madd_epi16(scale_0[j], p16_0),
                                                               _mm256_madd_epi16(scale_1[j], p16_1)));
#endif
            }
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) res[t] = hsum_float_8(acc[t]) + summs[t];
}

// ---- Q5_1 x Q8_1 (ggml_vec_dot_q5_1_q8_1, bit-exact)
void q5_1_dot(const block_q5_1* x, int nb, const void* const* act, int nt, float* res) {
    __m256 acc[MAXT];
    float summs[MAXT];
    for (int t = 0; t < nt; ++t) { acc[t] = _mm256_setzero_ps(); summs[t] = 0.0f; }
    for (int ib = 0; ib < nb; ++ib) {
        const __m256 dx = _mm256_set1_ps(half_at(&x[ib], 0));
        const float xm = half_at(&x[ib], 1);
        __m256i qx = bytes_from_nibbles_32(x[ib].qs);
        __m256i bxhi = bytes_from_bits_32(x[ib].qh);
        bxhi = _mm256_and_si256(bxhi, _mm256_set1_epi8(0x10));
        qx = _mm256_or_si256(qx, bxhi);
        for (int t = 0; t < nt; ++t) {
            const block_q8_1* y = (const block_q8_1*) act[t] + ib;
            summs[t] += xm * half_at(y, 1);
            const __m256 dy = _mm256_set1_ps(half_at(y, 0));
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y->qs);
            const __m256 q = mul_sum_us8_pairs_float(qx, qy);
            acc[t] = _mm256_fmadd_ps(q, _mm256_mul_ps(dx, dy), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) res[t] = hsum_float_8(acc[t]) + summs[t];
}

// ---- Q8_0 x Q8_0 (ggml_vec_dot_q8_0_q8_0, bit-exact)
void q8_0_dot(const block_q8_0* x, int nb, const void* const* act, int nt, float* res) {
    __m256 acc[MAXT];
    for (int t = 0; t < nt; ++t) acc[t] = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const float xd = h2f(x[ib].d);
        const __m256i qx = _mm256_loadu_si256((const __m256i*) x[ib].qs);
#if !defined(__AVXVNNIINT8__)
        const __m256i ax = _mm256_sign_epi8(qx, qx);
#endif
        for (int t = 0; t < nt; ++t) {
            const block_q8_0* y = (const block_q8_0*) act[t] + ib;
            const __m256 d = _mm256_set1_ps(xd * h2f(y->d));
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y->qs);
#if defined(__AVXVNNIINT8__)
            const __m256 q = _mm256_cvtepi32_ps(_mm256_dpbssd_epi32(_mm256_setzero_si256(), qx, qy));
#else
            const __m256i sy = _mm256_sign_epi8(qy, qx);
            const __m256 q = mul_sum_us8_pairs_float(ax, sy);
#endif
            acc[t] = _mm256_fmadd_ps(d, q, acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) res[t] = hsum_float_8(acc[t]);
}

void dot_rows(int type, const uint8_t* row, int n, const void* const* act, int nt, float* res) {
    switch (type) {
        case 12: q4k_dot((const block_q4_K*) row, n / QK_K, act, nt, res); break;
        case 13: q5k_dot((const block_q5_K*) row, n / QK_K, act, nt, res); break;
        case 7: q5_1_dot((const block_q5_1*) row, n / QK5_1, act, nt, res); break;
        case 8: q8_0_dot((const block_q8_0*) row, n / QK8_0, act, nt, res); break;
        default: break;
    }
}

// Software prefetch for the rows the pool streams through ggml's dot (and kq_avx2's rows): the experts come from
// DRAM, and the L2 streamer stops at every 4 KB page, so each row asks for the bytes `pf_ahead()` past itself, as the
// i-quant rows do (iq_avx2_rows.inc).  Only hints: the dots read the same bytes either way.  Opt-in: BASALT_PF=1 (2048
// bytes ahead) or BASALT_PF_AHEAD=N.  On a 265K with DDR5-6000 the pool is at the DRAM limit and it measured +0.5%.
inline int pf_ahead() {
    static const int a = [] {
        const char* on = std::getenv("BASALT_PF");
        const char* e = std::getenv("BASALT_PF_AHEAD");
        if (e) return std::atoi(e);
        return on && on[0] == '1' ? 2048 : 0;
    }();
    return a;
}
// the `n` bytes `ahead` past p, clipped at `end` (the expert's last byte + 1)
inline void pf_span(const uint8_t* p, size_t n, int ahead, const uint8_t* end) {
    if (ahead <= 0) return;
    const uint8_t* a = p + ahead;
    const uint8_t* b = a + n < end ? a + n : end;
    for (; a < b; a += 64) __builtin_prefetch(a, 0, 3);
}

}  // namespace

bool kq256_supported(int type) noexcept { return type == 12 || type == 13 || type == 7 || type == 8; }

void bf16_rows_dot_multi(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
    for (int r = 0; r < rows; ++r) {
        const uint16_t* wr = w + (size_t) r * (size_t) cols;
        __m256 acc[8];
        for (int t = 0; t < nt; ++t) acc[t] = _mm256_setzero_ps();
        for (int c = 0; c + 8 <= cols; c += 8) {
            const __m256 wf = _mm256_castsi256_ps(
                _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) (wr + c))), 16));
            for (int t = 0; t < nt; ++t) acc[t] = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x + (size_t) t * cols + c), acc[t]);
        }
        for (int t = 0; t < nt; ++t) out[(size_t) t * rows + r] = hsum_float_8(acc[t]);
    }
}

void kq256_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += MAXT) {
        const int m = nt - t0 < MAXT ? nt - t0 : MAXT;
        float g[MAXT], u[MAXT];
        const int ahead = pf_ahead();
        for (int r = r0; r < r1; ++r) {
            pf_span(blob + (size_t) r * gu_row, gu_row, ahead, blob + (size_t) r1 * gu_row);
            pf_span(blob + up_off + (size_t) r * gu_row, gu_row, ahead, blob + up_off + (size_t) r1 * gu_row);
            if (type == 12) {
                q4k_gu_row((const block_q4_K*) (blob + (size_t) r * gu_row),
                           (const block_q4_K*) (blob + up_off + (size_t) r * gu_row), n / QK_K, act + t0, m, g, u);
            } else {
                dot_rows(type, blob + (size_t) r * gu_row, n, act + t0, m, g);
                dot_rows(type, blob + up_off + (size_t) r * gu_row, n, act + t0, m, u);
            }
            for (int t = 0; t < m; ++t) ff[t0 + t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
        }
    }
}

void kq256_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += MAXT) {
        const int m = nt - t0 < MAXT ? nt - t0 : MAXT;
        float s[MAXT];
        const int ahead = pf_ahead();
        for (int r = r0; r < r1; ++r) {
            pf_span(w + (size_t) r * row_bytes, row_bytes, ahead, w + (size_t) r1 * row_bytes);
            dot_rows(type, w + (size_t) r * row_bytes, n, act + t0, m, s);
            for (int t = 0; t < m; ++t) out[t0 + t][r] = s[t];
        }
    }
}

}  // namespace basalt::kernels::cpu
