# basalt-u7265k-rtx5080

A fork of [Basalt](https://github.com/jesdga95/basalt) tuned for one machine: an **Intel Core Ultra 7 265K** (Arrow Lake-S,
8 Lion Cove P-cores + 12 Skymont E-cores, AVX2 + AVX-VNNI + AVX-VNNI-INT8, no AVX-512), **2 x 48 GB DDR5-6000**
(dual channel, dual rank) and an **RTX 5080** (16 GB, PCIe 5.0 x16), serving **Qwen3.8-Flash-Next UD-Q4_K_XL**.

The branch `basalt-u7265k-rtx5080` is upstream `main` plus a short stack of commits; `fork/update.sh` rebases it on
upstream, rebuilds, runs the parity tests and only then deploys, so upstream's work keeps flowing in.

Every change below either leaves the output **bit for bit** what upstream computes (checked by the tests named), or is
off unless switched on.

## What changed, and why

Decode on this machine streams the experts the GPU does not hold from DDR5 through the CPU pool. A verify window
(about 2.1 tokens at 4K context) spends ~70% of its time there and ~22% on the GPU (the dense weights from VRAM).

| Change | Effect on this machine | Exactness |
| --- | --- | --- |
| **Q4_K gate/up in the decode glue** (`dec_moe.cuh`). Upstream fuses Q5_K/Q6_K but not Q4_K, so UD-Q4_K_XL's 47 Q4_K layers fell back to the unfused path (`fallback: glue off ... types 12 / 7`). | GPU time per window -6% | `dec_moe_parity --kq`: all bitwise equal |
| **ggml's dot table built for AVX-VNNI + AVX-VNNI-INT8** on CPUs without AVX-512 (`ggml_dots_avxvnni`). Upstream builds AVX2 and AVX-512 only, so the `vpdpbusd` / `vpdpbssd` branches in ggml's code never compiled here. | Q8_0 dot +19-27%, Q5_1 +6% | `kq_vnni_test`: identical to the AVX2 table |
| **K-quant multi-token rows on AVX-VNNI, wired in** (`kq_avx2.cpp`, `native_expert.cpp`). Upstream wrote them but left them unwired (no gain without VNNI). Taken only from the token counts where they win (`kq_rows_pay`): at one token ggml's dot is faster. Q5_K added. | pool +1-2% | `kq_vnni_test`: 1431 comparisons, 0 mismatches vs upstream's AVX2 path |
| **Q5_0 down rows on AVX-VNNI** (`q5_avx2_vnni.cpp`), for the UD i-quant packs (33 of 48 down layers of UD-IQ4_XS). | 1.8-2.8x per row | `q50_vnni_test`: 1152 comparisons, 0 mismatches |
| **BF16 n-gram (engram) table** (`ngram.cpp`): Unsloth ships the table in BF16 (lossless); upstream reads IQ4_NL, Q5_0 and Q8_0 only. | lossless table; decode and prefill unchanged vs IQ4_NL | rows are the checkpoint's values |

Opt-in switches (measured, kept, but off - none was a clear win here):

| Switch | What | Measured on this machine |
| --- | --- | --- |
| `BASALT_THP=1` | transparent huge pages for the resident expert arena | no pool gain; loads up to 4x longer (compaction) |
| `BASALT_PF=1` / `BASALT_PF_AHEAD=N` | software prefetch on the ggml / K-quant rows | +0.5% (the pool is at the DRAM limit) |
| `BASALT_WARM=1` [`BASALT_WARM_N=n`] | the CPU warm-up for packs whose layers' experts differ in size (upstream skips them), top-N experts | +2-3% pool rate, offset by as much host time |
| `BASALT_PIN=1`, `BASALT_SPIN_US=N` | pin the pool's workers to cores; spin N us before yielding | pool rate unchanged; pinning crowds the host and I/O threads |
| `BASALT_KQ256=0`, `BASALT_GLUE_Q4K=0`, `BASALT_Q50_VNNI=0` | turn the fork's kernels off (A/B) | - |

Tried and dropped: upstream PR #282 (CPU misses staged over PCIe in single-stream windows) - its copies compete with the
pool for the same DDR5 bandwidth: pool 83 -> 77 GB/s, slower overall at 19 workers. The spin-then-yield wait patch and
`--kv-resident` (earlier sessions) - no clear gain.

Also measured and left at upstream's defaults: `--spec` 4/5/6 (tighter caps raise the acceptance rate but not the
accepted tokens), `--suffix-draft` 2/4, `--adapt-swaps` 48/160, an uncore (ring) clock pinned at 3.8 GHz, 16 workers
(79 GB/s vs 83 at 18).  A Fable 5.1 second-opinion review proposed pinning, longer spins and swap-rate changes; all
measured, none moved the pool rate or the hit rate.

## Configuration (`fork/q4kxl.json`)

| Setting | Why |
| --- | --- |
| `pool-affinity all`, `pool-workers 18` | upstream's hybrid rule picks 13 workers (P-cores - 1 + half the E-cores, from Alder/Raptor Lake). On Arrow Lake more cores keep more DRAM reads in flight: 13 workers stream ~72 GB/s, 18-19 ~83 GB/s (of 96 theoretical). 18 matches 19 and leaves a core for the system |
| `prefill-cpu-share 0.5` | basalt-probe's balance point on this machine (+9.6% prompt reads over 0.6) |
| `expert-profile-save` | the adaptive tier's learned placement survives restarts (the media gateway restarts Basalt after media jobs) |
| the pack | Unsloth UD-Q4_K_XL with the **BF16 n-gram table** (lossless; Basalt's published pack has IQ4_NL, relative row error 0.0757) and **Basalt's self-distilled MTP drafter** (its published UD-Q4_K_XL pack has Qwen's original). The expert and dense weights are byte-identical to Basalt's published pack (all 1,079 tensors): Unsloth's Nov 2026 upload of UD-Q4_K_XL moved the table into its own shard. The BF16 table measured no slower; the tuned drafter accepted 79.7% vs 77.4% (greedy) and 77.2% vs 75.1% (sampled) of drafts on realistic prompts, +2.5-3% decode. See "Building the pack" |

## Measured (UD-Q4_K_XL, fixed prompts, mean of runs 2-3)

Decode and prefill, tok/s, fixed synthetic prompts (sampled code-review answers, 400 tokens), runs 2-3 of 3:

| Context | Upstream Basalt + its published UD-Q4_K_XL pack | This fork + the pack below | Change |
| --- | --- | --- | --- |
| 4K | 76.2 decode / 2836 prefill | 92.6 / 2753 | +22% decode |
| 32K | 76.0 / 4773 | 89.4 / 4771 | +18% |
| 128K | 68.5 / 4504 | 82.5 / 4563 | +20% |
| 250K | 63.7 / 4046 | 78.9 / 4105 | +24% |

Realistic prompts (code writing, explanation, refactoring, shell, review, chat, prose, JSON; 4K context): ~110 tok/s,
draft acceptance 79.7% greedy / 77.2% sampled.  Short-prompt prefill is PCIe-bound (~87% of the 55.9 GB/s link), long
prompts GPU-compute-bound; neither moved much.

Upstream used its defaults (13 workers, 700 MiB reserve).  The synthetic prompts are harder to draft than real text,
so their tok/s is lower; run-to-run noise is about +-5%, which is why changes were judged on the decode-timing
breakdown (pool GB/s, GPU time per window, hit rate) as well.

Greedy output of the fork is byte-identical to upstream's at 4K, 32K and 128K (`--adapt-swaps 0`).

## Building the pack

```bash
G=~/models/unsloth-q4kxl-v31
hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-Q4_K_XL/*" --exclude "UD-Q4_K_XL/*-00004-of-00004.gguf" --local-dir $G
hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "engram/*BF16*" --local-dir $G   # an --include beside explicit
hf download unsloth/Qwen3.8-Flash-Next-GGUF engram/set_split.py --local-dir $G            # file names is ignored
python3 $G/engram/set_split.py $G/engram/Qwen3.8-Flash-Next-engram-BF16.gguf_file 4 4
# shards 1-3 + the table as shard 4 in one folder, then (Basalt's tuned drafter and vision from its V3.1 UD-IQ4_XS pack):
python3 tools/convert/convert.py --gguf SET/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
    --sections-from Qwen3.8-FN-UD-IQ4_XS.basalt --compat-bf16 --out Qwen3.8-FN-UD-Q4_K_XL-v31-e16.basalt
```

## Updating from upstream

```bash
fork/update.sh            # rebase on origin/main, build, parity tests, freeze bin-deployed/
fork/update.sh --deploy   # ... and restart the basalt service on it (fork/fork.conf drop-in)
```

Revert to the unit's own binary: `rm ~/.config/systemd/user/basalt.service.d/fork.conf && systemctl --user daemon-reload && systemctl --user restart basalt`.

## Hardware notes

- Decode is bound by DDR5 bandwidth (the pool reads ~1.45 GB of experts per 4K-context window). Faster memory is the
  remaining lever: DDR5-7200/8000 (CUDIMM, Intel's 200S Boost profile allows 8000) would raise it 20-30%. Raising only
  the D2D/NGU clocks mostly lowers latency, not bandwidth.
- The n-gram table is read from the SSD per token (4 KiB direct reads). Heavy writes to the same QLC drive (downloads,
  media output) raise read latency and slow decode measurably (~2 ms per window instead of ~0.03).
- The models live on a QLC, DRAM-less NVMe (Crucial P3 Plus).  Every other load of a pack reads at about a third of the
  speed (0.6 vs 1.9 GB/s: ~190 s instead of ~60 s), not thermal, no host writes - most likely the drive's own background
  work after re-reading the same ~75 GB.  The first requests after a slow load run slower too (n-gram reads wait).  A
  TLC NVMe with DRAM for the models would remove both; it matters because the media gateway restarts Basalt after media
  jobs.
- Arrow Lake reports AVX-VNNI-INT8 through CPUID even where `/proc/cpuinfo` has no flag name; Basalt's `cpu_avxvnni_int8_ok`
  reads CPUID directly.
