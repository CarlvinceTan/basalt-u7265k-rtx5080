// tests/kernels/cpu/pool_tasks_test.cpp - the pool's row tasks do not change a bit of the output.
//
//     build/pool_tasks_test [--bench [workers [arena GiB [batches]]]]
//
// A native layer is cut into row tasks the workers claim; how many (ExpertPool::set_tasks_per_thread, the hybrid rule
// of kTasksPerThreadHybrid) only decides which thread computes which rows.  This runs the same layers at upstream's
// count, the fork's hybrid rule and several fixed counts, for the three expert layouts of Unsloth's UD-Q4_K_XL and
// layers of 1 to 24 experts with 1 to 8 tokens each, and compares every output bit for bit.
//
// `--bench` streams random experts from a cold arena (far larger than the caches, with the ~110 us pause a decode
// layer's GPU part leaves between batches) and prints the pool's rate per task count: the measurement behind the rule.
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/cpu/pool.hpp"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace cpu = basalt::kernels::cpu;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int H = 2560, FF = 640;

uint64_t g_rng = 0x9e3779b97f4a7c15ull;
uint64_t rnd() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return g_rng;
}

// `n` experts of random weights with sane block scales (a random fp16 scale is often a NaN or huge)
std::vector<uint8_t> make_arena(const cpu::NativeFmt& f, size_t n) {
    std::vector<uint8_t> a(n * f.bytes + 64);
    for (size_t o = 0; o + 8 <= a.size(); o += 8) {
        const uint64_t v = rnd();
        std::memcpy(a.data() + o, &v, 8);
    }
    const uint16_t d = (uint16_t) _cvtss_sh(0.01f, 0), m = (uint16_t) _cvtss_sh(-0.05f, 0);
    const size_t gu_blk = ggml_type_size((ggml_type) f.gu_type), d_blk = ggml_type_size((ggml_type) f.d_type);
    for (size_t e = 0; e < n; ++e) {
        uint8_t* b = a.data() + e * f.bytes;
        for (size_t o = 0; o < f.down_off; o += gu_blk) {   // K-quant blocks start with d and dmin
            std::memcpy(b + o, &d, 2);
            std::memcpy(b + o + 2, &d, 2);
        }
        for (size_t o = f.down_off; o < f.bytes; o += d_blk) {
            std::memcpy(b + o, &d, 2);
            if (f.d_type == GGML_TYPE_Q5_1) std::memcpy(b + o + 2, &m, 2);
        }
    }
    return a;
}

struct Window {
    std::vector<uint8_t> nact;   // MAXT activations in the layer's dot type
    explicit Window(const cpu::NativeFmt& f) : nact((size_t) cpu::MAXT * cpu::kNativeActBytes) {
        std::mt19937 gen(7);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> x(H);
        for (int t = 0; t < cpu::MAXT; ++t) {
            for (float& v : x) v = nd(gen);
            cpu::native_quant_act(f, x.data(), nact.data() + (size_t) t * cpu::kNativeActBytes);
        }
    }
};

int parity() {
    struct Layout { int gu, d; const char* name; };
    const Layout layouts[] = {{12, 7, "Q4_K/Q5_1"}, {12, 8, "Q4_K/Q8_0"}, {13, 8, "Q5_K/Q8_0"}};
    const int counts[] = {0, 1, 8, 12, 32, 96};   // set_tasks_per_thread, each against the hybrid rule (-1)
    const int workers = (int) std::max(3u, std::min(8u, std::thread::hardware_concurrency()) - 1);
    cpu::ExpertPool pool(workers);
    int compared = 0, mismatches = 0;
    for (const Layout& lay : layouts) {
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt(lay.gu, lay.d, H, FF, f, err, 0.0f, -1)) {
            std::printf("%s: %s\n", lay.name, err.c_str());
            return 1;
        }
        constexpr int kExperts = 24;
        const std::vector<uint8_t> arena = make_arena(f, kExperts);
        const Window win(f);
        std::vector<cpu::ExpertJobMulti> jobs(kExperts);
        std::vector<float> ref((size_t) kExperts * cpu::MAXT * H), out(ref.size());
        for (int nb : {1, 2, 5, 11, 24}) {
            int nts[kExperts];
            for (int j = 0; j < nb; ++j) nts[j] = 1 + (int) (rnd() % (j % 3 == 0 ? cpu::MAXT : 2));
            auto run = [&](int tasks, std::vector<float>& dst) {
                std::fill(dst.begin(), dst.end(), 0.0f);
                for (int j = 0; j < nb; ++j) {
                    cpu::ExpertJobMulti& jb = jobs[(size_t) j];
                    jb.blob = arena.data() + (size_t) j * f.bytes;
                    jb.nt = nts[j];
                    for (int t = 0; t < jb.nt; ++t) {
                        jb.nact[t] = win.nact.data() + (size_t) ((j + t) % cpu::MAXT) * cpu::kNativeActBytes;
                        jb.out[t] = dst.data() + ((size_t) j * cpu::MAXT + (size_t) t) * H;
                    }
                }
                pool.set_tasks_per_thread(tasks);
                pool.run_split_multi_native(f, jobs.data(), nb);
            };
            run(-1, ref);
            for (int tasks : counts) {
                run(tasks, out);
                ++compared;
                if (std::memcmp(ref.data(), out.data(), ref.size() * sizeof(float)) != 0) {
                    ++mismatches;
                    std::printf("%s, %d experts: %d tasks a thread differs from the hybrid rule\n", lay.name, nb, tasks);
                }
            }
        }
    }
    std::printf("pool_tasks_test: %d comparisons, %d mismatches (%d workers)\n", compared, mismatches, workers);
    return mismatches == 0 ? 0 : 1;
}

int bench(int workers, double gib, int batches) {
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(12, 7, H, FF, f, err, 0.0f, -1)) return 1;
    const size_t n_exp = (size_t) (gib * 1073741824.0) / f.bytes;
    const std::vector<uint8_t> arena = make_arena(f, n_exp);
    const Window win(f);
    constexpr int kMaxJobs = 32;
    std::vector<cpu::ExpertJobMulti> jobs(kMaxJobs);
    std::vector<float> out((size_t) kMaxJobs * cpu::MAXT * H);
    cpu::ExpertPool pool(workers);
    std::printf("%d workers, %.1f GiB arena, %d batches each, 110 us between batches; GB/s by experts a batch\n", workers,
                gib, batches);
    std::printf("%-22s %8s %8s %8s %8s\n", "tasks a thread", "2", "4", "10.5", "24");
    for (int tasks : {0, 12, 24, 32, 48, -1}) {
        char label[32];
        std::snprintf(label, sizeof label, tasks < 0 ? "hybrid rule" : tasks == 0 ? "upstream" : "%d", tasks);
        std::printf("%-22s", label);
        for (int nb10 : {20, 40, 105, 240}) {
            pool.set_tasks_per_thread(tasks);
            double ms = 0.0;
            int64_t bytes = 0;
            for (int it = -30; it < batches; ++it) {
                const int nb = nb10 / 10 + ((int) (rnd() % 10) < nb10 % 10 ? 1 : 0);
                size_t picked[kMaxJobs];
                for (int j = 0; j < nb; ++j) {
                    for (bool dup = true; dup;) {
                        picked[j] = rnd() % n_exp;
                        dup = false;
                        for (int q = 0; q < j; ++q) dup |= picked[q] == picked[j];
                    }
                    cpu::ExpertJobMulti& jb = jobs[(size_t) j];
                    jb.blob = arena.data() + picked[j] * f.bytes;
                    const unsigned r = (unsigned) (rnd() % 100);
                    jb.nt = r < 78 ? 1 : r < 95 ? 2 : r < 99 ? 3 : 4;   // 1.28 tokens a CPU expert, as live windows
                    for (int t = 0; t < jb.nt; ++t) {
                        jb.nact[t] = win.nact.data() + (size_t) (rnd() % cpu::MAXT) * cpu::kNativeActBytes;
                        jb.out[t] = out.data() + ((size_t) j * cpu::MAXT + (size_t) t) * H;
                    }
                }
                const auto gap = Clock::now();
                while (std::chrono::duration<double, std::micro>(Clock::now() - gap).count() < 110.0) _mm_pause();
                const auto a = Clock::now();
                pool.run_split_multi_native(f, jobs.data(), nb);
                if (it < 0) continue;   // warm-up
                ms += std::chrono::duration<double, std::milli>(Clock::now() - a).count();
                bytes += (int64_t) nb * (int64_t) f.bytes;
            }
            std::printf(" %8.1f", (double) bytes / ms / 1e6);
        }
        std::printf("\n");
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0)
        return bench(argc > 2 ? std::atoi(argv[2]) : 18, argc > 3 ? std::atof(argv[3]) : 3.0,
                     argc > 4 ? std::atoi(argv[4]) : 1000);
    return parity();
}
