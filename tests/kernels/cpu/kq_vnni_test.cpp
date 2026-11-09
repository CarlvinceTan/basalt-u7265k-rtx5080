// tests/kernels/cpu/kq_vnni_test.cpp - the fork's AVX-VNNI CPU expert paths against the upstream AVX2 ones, bit for
// bit: the engine must produce the same numbers it did before these paths existed.
//
//     build/kq_vnni_test [--bench [iterations]]
//
// 1. ggml's dot table built with AVX-VNNI (+INT8) against the AVX2 build, every quantizable type the experts use.
// 2. The multi-token K-quant rows (kq256_rows / kq256_gu_rows: Q4_K, Q5_K gate/up, Q5_1, Q8_0 down) against the AVX2
//    build's single-token vec_dot, per token, for 1..8 tokens and partial row ranges.
// Weights come from ggml's own quantizer; activations from the AVX2 table's quantizer, plus extreme rows.
// `--bench` times 1, 2, 4 and 8 tokens of one expert's rows on one thread: AVX2 vec_dot, VNNI vec_dot, kq256 rows.
// Skipped (exit 0) on a CPU without AVX-VNNI and AVX-VNNI-INT8.
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/kq_avx2.hpp"

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

extern "C" const ggml_type_traits_cpu* ggml_get_type_traits_cpu_avxvnni(ggml_type type);

namespace cpu = basalt::kernels::cpu;

namespace {

constexpr int H = 2560, FF = 640, MAXT = 8;

struct Case {
    ggml_type type;
    int n;      // values per row (the dot length)
    int rows;
};

std::vector<uint8_t> quant_rows(ggml_type t, int rows, int n, uint64_t seed, int mode) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 0.02f);
    std::vector<float> x((size_t) rows * n);
    for (size_t i = 0; i < x.size(); ++i) {
        float v = nd(rng);
        if (mode == 1) v = (i % 7 == 0) ? 0.5f : v;              // outliers
        if (mode == 2) v = (i & 1) ? 0.03f : -0.03f;             // alternating extremes of a block
        x[i] = v;
    }
    std::vector<uint8_t> q(ggml_row_size(t, n) * (size_t) rows);
    ggml_quantize_chunk(t, x.data(), q.data(), 0, rows, n);
    return q;
}

std::vector<uint8_t> quant_act(ggml_type at, int n, uint64_t seed, int mode) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) n);
    for (int i = 0; i < n; ++i) {
        float v = nd(rng);
        if (mode == 1) v = (i % 32 == 3) ? 40.f : v * 0.1f;     // one spike per block: codes at +-127
        if (mode == 2) v = (i & 1) ? 3.f : -3.f;
        x[(size_t) i] = v;
    }
    std::vector<uint8_t> a(ggml_row_size(at, n));
    ggml_get_type_traits_cpu_avx2(at)->from_float(x.data(), a.data(), n);
    return a;
}

long g_compared = 0, g_bad = 0;

void expect_same(const char* what, const float* want, const float* got, int count) {
    ++g_compared;
    if (std::memcmp(want, got, (size_t) count * 4) == 0) return;
    int i = 0;
    while (i < count && std::memcmp(want + i, got + i, 4) == 0) ++i;
    std::printf("MISMATCH %s at %d: avx2 %.9g got %.9g\n", what, i, (double) want[i], (double) got[i]);
    ++g_bad;
}

void check_table(const Case& c) {
    const auto* ta = ggml_get_type_traits_cpu_avx2(c.type);
    const auto* tv = ggml_get_type_traits_cpu_avxvnni(c.type);
    if (!ta || !ta->vec_dot || !tv || !tv->vec_dot) return;
    const size_t rb = ggml_row_size(c.type, c.n);
    char what[96];
    for (int wm = 0; wm < 3; ++wm)
        for (int am = 0; am < 3; ++am) {
            const auto w = quant_rows(c.type, c.rows, c.n, 10 + wm, wm);
            const auto a = quant_act(ta->vec_dot_type, c.n, 20 + am, am);
            std::vector<float> ra((size_t) c.rows), rv((size_t) c.rows);
            for (int r = 0; r < c.rows; ++r) {
                ta->vec_dot(c.n, &ra[(size_t) r], 0, w.data() + (size_t) r * rb, 0, a.data(), 0, 1);
                tv->vec_dot(c.n, &rv[(size_t) r], 0, w.data() + (size_t) r * rb, 0, a.data(), 0, 1);
            }
            std::snprintf(what, sizeof what, "table %s w%d a%d", ggml_type_name(c.type), wm, am);
            expect_same(what, ra.data(), rv.data(), c.rows);
            // the activation quantizer of both tables
            if (ta->vec_dot_type != c.type) {
                const auto* qa = ggml_get_type_traits_cpu_avx2(ta->vec_dot_type);
                const auto* qv = ggml_get_type_traits_cpu_avxvnni(ta->vec_dot_type);
                if (qa && qa->from_float && qv && qv->from_float) {
                    std::vector<float> x((size_t) c.n);
                    std::mt19937_64 rng(30 + am);
                    std::normal_distribution<float> nd(0.f, 1.f);
                    for (auto& v : x) v = nd(rng);
                    std::vector<uint8_t> b1(ggml_row_size(ta->vec_dot_type, c.n)), b2(b1.size());
                    qa->from_float(x.data(), b1.data(), c.n);
                    qv->from_float(x.data(), b2.data(), c.n);
                    ++g_compared;
                    if (std::memcmp(b1.data(), b2.data(), b1.size())) {
                        std::printf("MISMATCH quantizer %s\n", ggml_type_name(ta->vec_dot_type));
                        ++g_bad;
                    }
                }
            }
        }
}

void check_kq(const Case& c, bool gate_up) {
    const auto* ta = ggml_get_type_traits_cpu_avx2(c.type);
    const size_t rb = ggml_row_size(c.type, c.n);
    const size_t up_off = rb * (size_t) c.rows;
    char what[96];
    for (int wm = 0; wm < 3; ++wm) {
        // gate/up: gate rows then up rows in one blob, as a native expert lays them out
        const auto w = quant_rows(c.type, gate_up ? 2 * c.rows : c.rows, c.n, 40 + wm, wm);
        std::vector<std::vector<uint8_t>> acts;
        const void* ap[MAXT];
        for (int t = 0; t < MAXT; ++t) {
            acts.push_back(quant_act(ta->vec_dot_type, c.n, 50 + t + 8 * wm, t % 3));
            ap[t] = acts.back().data();
        }
        // the AVX2 reference, per token and row: the upstream engine's path
        std::vector<float> ref((size_t) MAXT * c.rows);
        for (int t = 0; t < MAXT; ++t)
            for (int r = 0; r < c.rows; ++r) {
                float g = 0.f;
                ta->vec_dot(c.n, &g, 0, w.data() + (size_t) r * rb, 0, ap[t], 0, 1);
                if (gate_up) {
                    float u = 0.f;
                    ta->vec_dot(c.n, &u, 0, w.data() + up_off + (size_t) r * rb, 0, ap[t], 0, 1);
                    g = (g / (1.f + std::exp(-g))) * u;   // native_gu_rows' SwiGLU expression
                }
                ref[(size_t) t * c.rows + r] = g;
            }
        std::vector<float> got((size_t) MAXT * c.rows);
        for (int nt = 1; nt <= MAXT; ++nt) {
            float* op[MAXT];
            for (int t = 0; t < nt; ++t) op[t] = got.data() + (size_t) t * c.rows;
            // full range, then the same rows in uneven pieces
            for (int pass = 0; pass < 2; ++pass) {
                std::fill(got.begin(), got.end(), -1.f);
                if (pass == 0) {
                    if (gate_up) cpu::kq256_gu_rows(c.type, w.data(), rb, up_off, c.n, ap, nt, op, 0, c.rows);
                    else cpu::kq256_rows(c.type, w.data(), rb, c.n, ap, nt, op, 0, c.rows);
                } else {
                    for (int r = 0; r < c.rows;) {
                        const int r1 = std::min(c.rows, r + 1 + (r * 5 + nt) % 11);
                        if (gate_up) cpu::kq256_gu_rows(c.type, w.data(), rb, up_off, c.n, ap, nt, op, r, r1);
                        else cpu::kq256_rows(c.type, w.data(), rb, c.n, ap, nt, op, r, r1);
                        r = r1;
                    }
                }
                for (int t = 0; t < nt; ++t) {
                    std::snprintf(what, sizeof what, "kq256 %s %s w%d nt%d t%d pass%d", gate_up ? "gu" : "rows",
                                  ggml_type_name(c.type), wm, nt, t, pass);
                    expect_same(what, ref.data() + (size_t) t * c.rows, op[t], c.rows);
                }
            }
        }
    }
}

template <class F>
double best_us(int iters, F&& f) {
    double best = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) f();
        best = std::min(best, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters);
    }
    return best;
}

void bench(const Case& c, bool gate_up, int iters) {
    const auto* ta = ggml_get_type_traits_cpu_avx2(c.type);
    const auto* tv = ggml_get_type_traits_cpu_avxvnni(c.type);
    const size_t rb = ggml_row_size(c.type, c.n), up_off = rb * (size_t) c.rows;
    const auto w = quant_rows(c.type, gate_up ? 2 * c.rows : c.rows, c.n, 7, 0);
    std::vector<std::vector<uint8_t>> acts;
    const void* ap[MAXT];
    for (int t = 0; t < MAXT; ++t) { acts.push_back(quant_act(ta->vec_dot_type, c.n, 70 + t, 0)); ap[t] = acts.back().data(); }
    std::vector<float> o((size_t) MAXT * c.rows);
    float* op[MAXT];
    for (int t = 0; t < MAXT; ++t) op[t] = o.data() + (size_t) t * c.rows;
    const double mb = (double) w.size() / 1e6;
    std::printf("%s %s: %d x %d (%.2f MB)\n", ggml_type_name(c.type), gate_up ? "gate+up" : "down", c.rows, c.n, mb);
    for (int nt : {1, 2, 4, 8}) {
        auto per_token = [&](const ggml_type_traits_cpu* tc) {
            for (int r = 0; r < c.rows; ++r)
                for (int t = 0; t < nt; ++t) {
                    tc->vec_dot(c.n, op[t] + r, 0, w.data() + (size_t) r * rb, 0, ap[t], 0, 1);
                    if (gate_up) tc->vec_dot(c.n, op[t] + r, 0, w.data() + up_off + (size_t) r * rb, 0, ap[t], 0, 1);
                }
        };
        const double a2 = best_us(iters, [&] { per_token(ta); });
        const double vn = best_us(iters, [&] { per_token(tv); });
        const double kq = best_us(iters, [&] {
            if (gate_up) cpu::kq256_gu_rows(c.type, w.data(), rb, up_off, c.n, ap, nt, op, 0, c.rows);
            else cpu::kq256_rows(c.type, w.data(), rb, c.n, ap, nt, op, 0, c.rows);
        });
        std::printf("  nt %d: avx2 vec_dot %7.1f us  vnni vec_dot %7.1f us (x%.2f)  kq256 %7.1f us (x%.2f, %.1f GB/s)\n", nt,
                    a2, vn, a2 / vn, kq, a2 / kq, mb * 1e3 / kq);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (!cpu::cpu_avxvnni_ok() || !cpu::cpu_avxvnni_int8_ok()) {
        std::printf("kq_vnni_test: no AVX-VNNI + AVX-VNNI-INT8 on this CPU, skipped\n");
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--bench") {
        const int it = argc > 2 ? std::atoi(argv[2]) : 50;
        bench({GGML_TYPE_Q4_K, H, FF}, true, it);
        bench({GGML_TYPE_Q5_K, H, FF}, true, it);
        bench({GGML_TYPE_Q5_1, FF, H}, false, it);
        bench({GGML_TYPE_Q8_0, FF, H}, false, it);
        return 0;
    }
    // 1. the dot tables: gate/up geometry for the 256-block formats, down geometry for the 32-block ones
    for (ggml_type t : {GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K})
        check_table({t, H, 256});
    for (ggml_type t : {GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0})
        check_table({t, FF, 512});
    // 2. the K-quant multi-token rows against the AVX2 per-token dot
    check_kq({GGML_TYPE_Q4_K, H, FF}, true);
    check_kq({GGML_TYPE_Q5_K, H, FF}, true);
    check_kq({GGML_TYPE_Q4_K, H, 64}, false);
    check_kq({GGML_TYPE_Q5_K, H, 64}, false);
    check_kq({GGML_TYPE_Q5_1, FF, H}, false);
    check_kq({GGML_TYPE_Q8_0, FF, H}, false);
    std::printf("kq_vnni_test: %ld comparisons, %ld mismatches\n", g_compared, g_bad);
    return g_bad ? 1 : 0;
}
