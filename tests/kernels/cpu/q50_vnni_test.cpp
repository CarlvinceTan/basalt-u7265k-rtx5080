// tests/kernels/cpu/q50_vnni_test.cpp - the Q5_0 AVX-VNNI down rows against ggml's Q5_0 dot, which the engine uses
// for these rows without them.
//
//     build/q50_vnni_test [--bench [iterations]]
//
// The rows must equal ggml_vec_dot_q5_0_q8_0 (through the CPU type traits, the engine's call) to the bit, per token
// and row: random weights and activations with the down projection's geometry (H rows of FF values), 1 to 8 tokens
// per call, row ranges of every length and start, and weights and activations pinned to their extreme codes.
// Activation codes stay in [-127, 127] (every Q8_0 quantizer's range); a -128 code is only checked against a double
// reference, since ggml's sign trick wraps it and these rows don't.  Skipped (exit 0) without AVX-VNNI.
// `--bench` times one expert's down rows (H x FF) for 1, 2, 4 and 8 tokens on both paths, on the calling thread.
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/q5_avx2.hpp"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = basalt::kernels::cpu;

namespace {

constexpr int H = 2560, FF = 640, MAXT = 8;
constexpr int NB = FF / 32;
constexpr size_t QBLK = 22, ABLK = 34;
constexpr size_t ROW = (size_t) NB * QBLK;

uint16_t f2h(float f) { return (uint16_t) _cvtss_sh(f, 0); }

// mode 0 random; 1 every weight code 0 (value -16); 2 every code 31 (value 15); 3 alternating 0 / 31
void fill_weights(std::vector<uint8_t>& w, uint64_t seed, int mode) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> sc(1e-4f, 4e-2f);
    w.assign((size_t) H * ROW, 0);
    for (int r = 0; r < H; ++r)
        for (int b = 0; b < NB; ++b) {
            uint8_t* blk = w.data() + (size_t) r * ROW + (size_t) b * QBLK;
            const uint16_t d = f2h(sc(rng) * ((rng() & 1) ? 1.f : -1.f));
            std::memcpy(blk, &d, 2);
            for (int i = 2; i < (int) QBLK; ++i) {
                uint8_t v = (uint8_t) rng();
                if (mode == 1) v = 0x00;
                if (mode == 2) v = 0xFF;
                if (mode == 3) v = (uint8_t) ((i & 1) ? 0xFF : 0x00);
                blk[i] = v;
            }
        }
}

// mode 0 random in [-127, 127]; 1 all +127; 2 all -127; 3 alternating; 4 random including -128 (reference only)
void fill_acts(std::vector<uint8_t>& a, uint64_t seed, int mode) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> sc(1e-3f, 2e-1f);
    a.assign((size_t) NB * ABLK, 0);
    for (int b = 0; b < NB; ++b) {
        uint8_t* blk = a.data() + (size_t) b * ABLK;
        const uint16_t d = f2h(sc(rng));
        std::memcpy(blk, &d, 2);
        for (int i = 0; i < 32; ++i) {
            int v = (int) (rng() % 255) - 127;
            if (mode == 1) v = 127;
            if (mode == 2) v = -127;
            if (mode == 3) v = (i & 1) ? 127 : -127;
            if (mode == 4) v = (int) (rng() % 256) - 128;
            blk[2 + i] = (uint8_t) (int8_t) v;
        }
    }
}

ggml_vec_dot_t ggml_dot() { return ggml_get_type_traits_cpu(GGML_TYPE_Q5_0)->vec_dot; }

void ggml_rows(const std::vector<uint8_t>& w, const void* const* hq, int nt, float* const* out, int r0, int r1) {
    const ggml_vec_dot_t dot = ggml_dot();
    for (int r = r0; r < r1; ++r)
        for (int t = 0; t < nt; ++t) {
            float s = 0.f;
            dot(FF, &s, 0, w.data() + (size_t) r * ROW, 0, hq[t], 0, 1);
            out[t][r] = s;
        }
}

int bench(int iters) {
    std::vector<uint8_t> w;
    fill_weights(w, 7, 0);
    std::vector<std::vector<uint8_t>> acts(MAXT);
    const void* hq[MAXT];
    for (int t = 0; t < MAXT; ++t) { fill_acts(acts[(size_t) t], 100 + (uint64_t) t, 0); hq[t] = acts[(size_t) t].data(); }
    std::vector<std::vector<float>> o(MAXT, std::vector<float>(H));
    float* op[MAXT];
    for (int t = 0; t < MAXT; ++t) op[t] = o[(size_t) t].data();
    std::printf("one expert's Q5_0 down rows: %d x %d (%.2f MB of weights), %d iterations, this thread\n", H, FF,
                (double) w.size() / 1e6, iters);
    for (int nt : {1, 2, 4, 8}) {
        double best[2] = {1e30, 1e30};
        for (int path = 0; path < 2; ++path)
            for (int rep = 0; rep < 3; ++rep) {
                const auto t0 = std::chrono::steady_clock::now();
                for (int it = 0; it < iters; ++it) {
                    if (path == 0) ggml_rows(w, hq, nt, op, 0, H);
                    else cpu::q50_down_rows_vnni(w.data(), ROW, FF, hq, nt, op, 0, H);
                }
                const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / iters;
                best[path] = std::min(best[path], s);
            }
        std::printf("  nt %d: ggml %8.1f us  vnni rows %8.1f us  x%.2f  (%.1f vs %.1f GB/s of weights)\n", nt,
                    best[0] * 1e6, best[1] * 1e6, best[0] / best[1], (double) w.size() / best[0] / 1e9,
                    (double) w.size() / best[1] / 1e9);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (!cpu::cpu_avxvnni_ok()) {
        std::printf("q50_vnni_test: no AVX-VNNI on this CPU, skipped\n");
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--bench") return bench(argc > 2 ? std::atoi(argv[2]) : 200);

    long compared = 0, bad = 0;
    std::vector<uint8_t> w;
    std::vector<std::vector<uint8_t>> acts(MAXT);
    const void* hq[MAXT];
    std::vector<std::vector<float>> og(MAXT, std::vector<float>(H)), ov(MAXT, std::vector<float>(H));
    float *pg[MAXT], *pv[MAXT];
    for (int t = 0; t < MAXT; ++t) { pg[t] = og[(size_t) t].data(); pv[t] = ov[(size_t) t].data(); }

    for (int wmode = 0; wmode < 4; ++wmode)
        for (int amode = 0; amode < 4; ++amode) {
            fill_weights(w, 11 + (uint64_t) wmode, wmode);
            for (int t = 0; t < MAXT; ++t) { fill_acts(acts[(size_t) t], 1000 + (uint64_t) (amode * 16 + t), amode); hq[t] = acts[(size_t) t].data(); }
            for (int nt = 1; nt <= MAXT; ++nt) {
                ggml_rows(w, hq, nt, pg, 0, H);
                cpu::q50_down_rows_vnni(w.data(), ROW, FF, hq, nt, pv, 0, H);
                for (int t = 0; t < nt; ++t) {
                    ++compared;
                    if (std::memcmp(pg[t], pv[t], H * sizeof(float))) {
                        int r = 0;
                        while (r < H && std::memcmp(&pg[t][r], &pv[t][r], 4) == 0) ++r;
                        std::printf("MISMATCH w%d a%d nt %d token %d row %d: ggml %.9g vnni %.9g\n", wmode, amode, nt, t, r,
                                    (double) pg[t][r], (double) pv[t][r]);
                        ++bad;
                    }
                }
                // row ranges of 1..13 rows: every pass width and remainder, against the full call
                std::vector<std::vector<float>> part(nt, std::vector<float>(H, 0.f));
                float* pp[MAXT];
                for (int t = 0; t < nt; ++t) pp[t] = part[(size_t) t].data();
                for (int r = 0; r < H;) {
                    const int r1 = std::min(H, r + 1 + (r * 7 + nt) % 13);
                    cpu::q50_down_rows_vnni(w.data(), ROW, FF, hq, nt, pp, r, r1);
                    r = r1;
                }
                for (int t = 0; t < nt; ++t) {
                    ++compared;
                    if (std::memcmp(pp[t], pv[t], H * sizeof(float))) { std::printf("MISMATCH row ranges w%d a%d nt %d token %d\n", wmode, amode, nt, t); ++bad; }
                }
            }
        }

    // -128 activation codes: against a double reference (ggml's sign trick wraps them; these rows sum them exactly)
    double worst = 0.0;
    {
        fill_weights(w, 99, 0);
        const auto* tw = ggml_get_type_traits(GGML_TYPE_Q5_0);
        const auto* ta = ggml_get_type_traits(GGML_TYPE_Q8_0);
        std::vector<float> wrow(FF), hf(FF);
        for (int t = 0; t < MAXT; ++t) { fill_acts(acts[(size_t) t], 5000 + (uint64_t) t, 4); hq[t] = acts[(size_t) t].data(); }
        cpu::q50_down_rows_vnni(w.data(), ROW, FF, hq, MAXT, pv, 0, H);
        for (int t = 0; t < MAXT; ++t) {
            ta->to_float(acts[(size_t) t].data(), hf.data(), FF);
            for (int r : {0, 1, 777, H - 1}) {
                tw->to_float(w.data() + (size_t) r * ROW, wrow.data(), FF);
                double want = 0.0, mag = 0.0;
                for (int i = 0; i < FF; ++i) { want += (double) wrow[(size_t) i] * hf[(size_t) i]; mag += std::fabs((double) wrow[(size_t) i] * hf[(size_t) i]); }
                worst = std::max(worst, std::fabs((double) pv[t][r] - want) / (mag > 0 ? mag : 1.0));
            }
        }
    }
    std::printf("q50_vnni_test: %ld comparisons, %ld mismatches; -128 codes vs double reference: worst error %.2e of the terms' magnitude\n",
                compared, bad, worst);
    if (worst > 1e-5) { std::printf("FAIL: -128 reference error too large\n"); return 1; }
    return bad ? 1 : 0;
}
