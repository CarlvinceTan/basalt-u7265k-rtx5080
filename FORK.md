# basalt-u7265k-rtx5080

A fork of [Basalt](https://github.com/jesdga95/basalt) tuned for one machine: an **Intel Core Ultra 7 265K** (Arrow Lake-S,
8 Lion Cove P-cores + 12 Skymont E-cores, AVX2 + AVX-VNNI + AVX-VNNI-INT8, no AVX-512), **2 x 48 GB DDR5-6000**
(dual channel, dual rank) and an **RTX 5080** (16 GB, PCIe 5.0 x16), serving **Qwen3.8-Flash-Next UD-Q4_K_XL**.

The branch `basalt-u7265k-rtx5080` is upstream `main` plus a short stack of commits; `fork/update.sh` rebases it on
upstream, rebuilds, runs the parity tests and only then deploys, so upstream's work keeps flowing in.  It sits on
upstream v0.3.2 (2026-11-09).

Every change below either leaves the output **bit for bit** what upstream computes (checked by the tests named), or is
off unless switched on.  One exception is marked: where experts are placed, which moves replies the way upstream's own
adaptive tier does (the GPU and the CPU round differently) and no further.

## What changed, and why

Decode on this machine streams the experts the GPU does not hold from DDR5 through the CPU pool. A verify window
(about 2.1 tokens at 4K context) spends ~70% of its time there and ~22% on the GPU (the dense weights from VRAM).

| Change | Effect on this machine | Exactness |
| --- | --- | --- |
| **Q4_K gate/up in the decode glue** (`dec_moe.cuh`). Upstream fuses Q5_K/Q6_K but not Q4_K, so UD-Q4_K_XL's 47 Q4_K layers fell back to the unfused path (`fallback: glue off ... types 12 / 7`). | GPU time per window -6% | `dec_moe_parity --kq`: all bitwise equal |
| **ggml's dot table built for AVX-VNNI + AVX-VNNI-INT8** on CPUs without AVX-512 (`ggml_dots_avxvnni`). Upstream builds AVX2 and AVX-512 only, so the `vpdpbusd` / `vpdpbssd` branches in ggml's code never compiled here. | Q8_0 dot +19-27%, Q5_1 +6% | `kq_vnni_test`: identical to the AVX2 table |
| **K-quant multi-token rows on AVX-VNNI, wired in** (`kq_avx2.cpp`, `native_expert.cpp`). Upstream wrote them but left them unwired (no gain without VNNI). Taken only from the token counts where they win (`kq_rows_pay`): at one token ggml's dot is faster. Q5_K added. | pool +1-2% | `kq_vnni_test`: 1431 comparisons, 0 mismatches vs upstream's AVX2 path |
| **Q5_0 down rows on AVX-VNNI** (`q5_avx2_vnni.cpp`), for the UD i-quant packs (33 of 48 down layers of UD-IQ4_XS). | 1.8-2.8x per row | `q50_vnni_test`: 1152 comparisons, 0 mismatches |
| **Finer row tasks in the CPU pool on a hybrid CPU** (`pool.cpp`). A layer's experts are cut into row tasks the workers claim. Upstream cuts 12 a thread, and 8 on AVX2 since v0.3.2 (#255); P- and E-cores differ 2x in speed, so the tail of each phase is longer here and finer tasks pay: 32 a thread, each kept between 24 and 48 KB of weights. | the pool alone, 10.5 experts a layer: 86.2 GB/s (8), 88.4 (12), 90.8 (the rule); 24 experts: 87.4 / 87.8 / 91.2. In the engine: +1.8% decode over upstream's 8, +0.4% over 12 (identical greedy runs) | `pool_tasks_test`: 90 comparisons across task counts, layouts and layer sizes, 0 mismatches; greedy replies byte-identical |
| **The adaptive tier's rounds rank across layers within a blob size** (`tiers.cpp`). Upstream lets VRAM slots follow a request's layers only when every layer's experts are one size; a pack with a few layers of another size (Unsloth's dynamic quants: 43 + 4 + 1 layers here) fell back to per-layer rounds, each layer's slots fixed by the profile. A slot takes any expert of its own size, so the fork ranks once per size. | realistic prompts, 3 runs of 16 requests each way: hit rate 59.2% -> 60.5%, pool time per window position -2.7%, decode +2.5%. Fixed one-topic prompts: within noise | **placement only** - not bit for bit (see above). `BASALT_XQ_CLASSES=0` is upstream's rule. Checked at 128K context (the KV takes cache slots) and with an image request (the vision path borrows them) |

The BF16 n-gram table the fork used to add is upstream's since v0.3.2 (#185); the fork's commit went at that rebase.
Each request's log also ends with the table's read statistics (`ple io: ...`: rows, row-cache hits, SSD read latency,
time blocked), because a slow drive shows there first (Hardware notes).

Opt-in switches (measured, kept, but off - none was a clear win here):

| Switch | What | Measured on this machine |
| --- | --- | --- |
| `BASALT_THP=1` | transparent huge pages for the resident expert arena | no pool gain; loads up to 4x longer (compaction) |
| `BASALT_PF=1` / `BASALT_PF_AHEAD=N` | software prefetch on the ggml / K-quant rows | +0.5% in the engine, none on the pool alone |
| `BASALT_WARM=1` [`BASALT_WARM_N=n`] | the CPU warm-up for packs whose layers' experts differ in size (upstream skips them), top-N experts | +2-3% pool rate, offset by as much host time |
| `BASALT_PIN=1`, `BASALT_SPIN_US=N` | pin the pool's workers to cores; spin N us before yielding | pool rate unchanged; pinning crowds the host and I/O threads |
| `BASALT_SWAP_GAIN=G`, `BASALT_SWAP_MIN=M` | the adaptive rounds' thresholds in routing counts: a candidate's lead over the resident it replaces (upstream 1.5) and its own least count | a gain of 3.0 or 4.5 halves the swaps and frees the bus for the pool (+1-3 GB/s), and loses as much in hit rate (-1 to -2.5 points): pool time per position within 1.5% either way |
| `BASALT_POOL_TASKS=N` | N row tasks a thread with no size bounds (0 = upstream's count) | see the table above |
| `BASALT_KQ256=0`, `BASALT_GLUE_Q4K=0`, `BASALT_Q50_VNNI=0`, `BASALT_XQ_CLASSES=0` | turn the fork's kernels and its cross-layer rounds off (A/B) | - |

Tried and dropped: upstream PR #282 (CPU misses staged over PCIe in single-stream windows) - its copies compete with the
pool for the same DDR5 bandwidth: pool 83 -> 77 GB/s, slower overall at 19 workers. The spin-then-yield wait patch and
`--kv-resident` (earlier sessions) - no clear gain.

Also measured and left at upstream's defaults: `--spec` 4/5/6 (tighter caps raise the acceptance rate but not the
accepted tokens), `--suffix-draft` 2/4, an uncore (ring) clock pinned at 3.8 GHz, 16 workers (79 GB/s vs 83 at 18).
A Fable 5.1 second-opinion review proposed pinning, longer spins and swap-rate changes; all measured, none moved the
pool rate or the hit rate.

`--adapt-swaps` (the swaps a round keeps, best first; default 96): 48 and 160 change nothing, because a round finds
about 115 swaps and a saturated round may keep four times the cap.  24 / 12 / 6 do bind, and cut both ways: the swap
copies are ~5 GB/s of reads on the same DDR5 bus (a 5 GB/s reader beside the pool alone costs it 88 -> 84 GB/s), so
the pool speeds up, but on realistic prompts, each a new topic, the hit rate falls from 59% to 55 / 51 / 45% and pool
time per position rises 4 / 13 / 22%.  The swaps earn about three reads for each one they cost; the default stays.

## Configuration (`fork/q4kxl.json`)

| Setting | Why |
| --- | --- |
| `pool-affinity all`, `pool-workers 18` | upstream's hybrid rule picks 13 workers (P-cores - 1 + half the E-cores, from Alder/Raptor Lake). On Arrow Lake more cores keep more DRAM reads in flight: 13 workers stream ~72 GB/s, 18-19 ~83 GB/s (of 96 theoretical). 18 matches 19 and leaves a core for the system |
| `prefill-cpu-share 0.5` | basalt-probe's balance point on this machine (+9.6% prompt reads over 0.6) |
| `conversation-cache-mib 4096`, `conversation-cache-min-tokens 8192`, `conversation-cache-disk-mib 16384` | OpenCode sends small side requests (titles, summaries) between an agent session’s turns; with one session slot and the cache off (upstream default) each evicted a ~105K-token session, which then re-read from scratch (23-28 s). Parked, a 44K session restores in ~40 ms (time to first token 0.3 s); the disk tier also survives restarts |
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

Greedy output of the fork is byte-identical to upstream's at 4K, 32K and 128K (`--adapt-swaps 0`).  The v0.3.2 rebase
with the row-task rule gives the same bytes as the build before it (six prompts at 4K and 32K).

The rebase onto v0.3.2 (2026-11-10), realistic prompts, without the drive's waits (Hardware notes): the build before it
108.7 tok/s, v0.3.2 with upstream's row tasks 106.4, with the fork's rule 109.3-109.7, with the cross-layer rounds
112.2-112.5.  Changes this small were judged on the decode-timing breakdown per window position (the pool's time, the
hit rate), which does not ride on how the drafts happened to fall; a fixed prompt replays the same text, so its
repeats are one sample.

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

- Decode is bound by DDR5 bandwidth (the pool reads ~1.45 GB of experts per 4K-context window).  Plain reads on 18
  threads reach 93 GB/s on this machine; the pool alone reaches 91 (`build/pool_tasks_test --bench`) and about 84 in
  the engine, where the adaptive tier's copies into VRAM take ~5 GB/s of the same bus and buy the hit rate with it.
  The kernels are not the limit: one expert in cache runs at 24 GB/s on a P-core and 12.6 on an E-core, and the
  workers are busy ~95% of the pool's phase.  Faster memory is the remaining lever: DDR5-7200/8000 (CUDIMM, Intel's
  200S Boost profile allows 8000) would raise it 20-30%.  Raising only the D2D/NGU clocks mostly lowers latency, not
  bandwidth.
- The n-gram table is read from the SSD per token (4 KiB direct reads). Measured 2026-11-10: a healthy drive blocked a
  decode window 0.15 ms for its rows (p50 0.37 ms a read, p99 3 ms); after a slow pack load, 4.2 ms (p50 0.40 ms, p99
  26 ms - the tail is what costs, 8-14% of decode). Heavy writes to the same QLC drive (downloads, media output) are
  the cause.
- The models live on a QLC, DRAM-less NVMe (Crucial P3 Plus).  Every other load of a pack reads at about a third of the
  speed (0.6 vs 1.9 GB/s: ~190 s instead of ~60 s), not thermal, no host writes - most likely the drive's own background
  work after re-reading the same ~75 GB.  The first requests after a slow load run slower too (n-gram reads wait).  A
  TLC NVMe with DRAM for the models would remove both; it matters because the media gateway restarts Basalt after media
  jobs.  Measured 2026-11-10 with 4 KiB direct reads of the pack: healthy, 0.21 ms a read and 1.1 ms for the 48 reads
  of one window; during a slow load the drive reads 430-650 MB/s and one small read takes 20-50 ms; it recovered about
  30 s after the load ended.  In the affected stretches a decode window waits 3-6 ms for its rows (15-20% of decode);
  that day, with a load every few minutes, 11 of 16 loads were slow.
- The n-gram table was tried on the second drive (a SATA 860 QVO) to get it out of the pack reads' way: the table's
  102 GB copied to a file of its own and read from there (`BASALT_PLE_FILE`, a fork change since reverted).  It lost
  clearly - read p50 2.9 ms against the pack's 0.37 ms, a decode window blocked 3.8 ms instead of 0.15, 4K decode 78
  against 91 tok/s.  The QVO serves the first reads of a fresh copy from its SLC cache (0.25-0.4 ms) and ~3 ms once the
  fold moves it to QLC; standalone 4 KiB random reads of the copy made it look fast.  The table stays on the pack's
  drive; only a TLC drive with DRAM is worth trying it on.
- Arrow Lake reports AVX-VNNI-INT8 through CPUID even where `/proc/cpuinfo` has no flag name; Basalt's `cpu_avxvnni_int8_ok`
  reads CPUID directly.
