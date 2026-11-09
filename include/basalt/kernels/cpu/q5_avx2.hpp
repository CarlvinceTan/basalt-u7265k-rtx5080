// include/basalt/kernels/cpu/q5_avx2.hpp - Q5_0 down rows against Q8_0 activations on AVX-VNNI, several tokens per
// pass over the weights.
//
// The experts' down projections of Unsloth's dynamic packs (UD-IQ4_XS: 33 of 48 layers) are Q5_0.  Without this file
// they go through ggml's single-token dot (ggml_vec_dot_q5_0_q8_0, the AVX2 `maddubs` build), which decodes every
// weight block once per token.  These rows decode a block once for up to four tokens and sum each block with one
// `vpdpbusd` (AVX-VNNI: Arrow Lake, Alder Lake, Zen 5 without AVX-512 VNNI on the path).
//
// Bit for bit equal to ggml_vec_dot_q5_0_q8_0 per token and row, for activation codes in [-127, 127] (every Q8_0
// quantizer's range: the scale is amax / 127): each int32 lane holds the same exact integer as ggml's
// abs/sign+maddubs+madd lane (the weight is used unsigned, 0..31, and 16 * sum(y) is taken off through the seed),
// the block scale is the same float product, the float accumulation runs over the blocks in the same order with the
// same FMA, and the final horizontal sum is ggml's hsum_float_8.  A -128 code would differ: ggml's sign trick wraps
// it, these rows don't.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::kernels::cpu {

/// Rows of n values these rows take: whole Q5_0 blocks, at most q50_max_blocks of them (n <= 2048).
constexpr int q50_max_blocks = 64;
bool q50_vnni_fits(int n) noexcept;

/// out[t][r] = w_r . hq[t] for rows r in [r0, r1); w: rows of row_bytes bytes of Q5_0 blocks, hq[t]: n / 32 ggml
/// block_q8_0 of token t, 1 <= nt <= 8.  Call only behind cpu_avxvnni_ok() and q50_vnni_fits(n).
void q50_down_rows_vnni(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt,
                        float* const* out, int r0, int r1);

/// The rows above are used unless BASALT_Q50_VNNI=0 is set (an A/B switch for measuring them).
bool q50_vnni_enabled() noexcept;

}  // namespace basalt::kernels::cpu
