# Fork ledger

## 2026-11-28: upstream main (+150 commits: Gemma 4, ERNIE-5.5-Flash), NVFP4-DQ8 pack

Merge `9c5a7c26` (`merge-main-trial` -> `upgrade-0.5.7`). Machine: Core Ultra 7 265K, 2x48 GB DDR5-6000, RTX 5080 (360 W), CUDA 13.0, driver 580.159.04.
Pack and flags as in the 2026-11-27 entry. Two conflicts, both where upstream rewrote a file the fork also owns:

- `CMakeLists.txt`: upstream's `mxfp4_avx2*.cpp` and `nvfp4_avx2_dpbusd.cpp` added beside the fork's VNNI flags on `kq_avx2.cpp` (upstream had lowered that file to plain AVX2; the fork's multi-token rows need VNNI).
- `src/core/ram_load.cpp`: #834 asks for huge pages on every RAM allocation; the fork keeps its `BASALT_THP` gate around the same `advise_hugepages` helper. The gate is what holds up: plain upstream took 9.4 min to load the pack, the fork and the merge 3.5 (load row below), because compaction has to be waited out.

A/B (`basalt-bench -r 3 -p 8k,64k,250k -t conv,struct,prose --seed 7`), order cur/up/up/cur then merge/merge, one fresh engine per arm, medians of the warm repetitions (2 and 3), n=4 per arm:

| Metric | Fork `3a21685a` | Upstream `e284ecf9` | Merge `9c5a7c26` |
| --- | --- | --- | --- |
| prefill 8K (engine tok/s) | 4764 | 4775 | 4771 |
| prefill 64K | 5723 | 5711 | 5705 |
| prefill 250K | 5328 | 5316 | 5314 |
| decode C=1 struct (acc 98%) | 183.4 | 181.6 | 184.7 |
| decode C=1 conv (acc 68-71%) | 118.6 | 120.4 | 118.3 |
| decode C=1 prose (acc 68-70%) | 113.7 | 118.3 | 114.1 |
| model load (wall, expert fault-in) | ~3.5 min | ~9.4 min | ~3.5 min |

Nothing in the 150 commits touches this pack: the new AVX-VNNI NVFP4 rung (#688) is for CPUs without AVX-VNNI-INT8, and the MXFP4/MXFP8 rows are other formats. Prefill is within 0.2% across the three arms; decode struct, the only tight decode column (sd ~1%), puts the fork and the merge about 1% over plain upstream. Conv and prose swing ±10% between repetitions in every arm, so those columns carry no signal. The merge is neutral on speed and is taken for Gemma 4 / ERNIE-5.5-Flash and the kernel correctness fixes (#789, Q8_1/Q8_K edge cases), not for tok/s.

Service: `fork/update.sh --no-fetch --deploy` rebuilt it (all models), `kq_vnni_test` 1431/0, `q50_vnni_test` 1152/0, `pool_tasks_test` 90/0; GPU parity skipped because the live service held the card. Chat verified on :11001 after the restart (thinking off, and `BASALT_ANY_MODEL=1` answering a client's model name), load 3.5 min, no config changes.
Raw results: `~/bench/basalt-0.5.7/ab3-{cur,up,merge}-{0,1}/` (`result.json` per arm).

## 2026-11-27: upstream 0.5.5 -> main (v0.5.7 + 136 commits), NVFP4-DQ8 pack

Merge `3a21685a` (branch `upgrade-0.5.7` -> `basalt-u7265k-rtx5080`). Machine: Core Ultra 7 265K, 2x48 GB DDR5-6000, RTX 5080 (360 W), CUDA 13.0, driver 580.159.04.
Pack `Qwen3.8-FN-NVFP4-DQ8.basalt`, production flags (`--kv int8 --resident-experts --pool-affinity all --pool-workers 16 --prefill-cpu-share 0.6 --vram-reserve-mib 300 --kv-resident 20480 --max-context 262144`), `basalt-bench -r 3 -p 8k,64k,250k -t conv,struct,prose --seed 7`.
Order A B B A (old, new, new, old), one fresh engine per arm, live service stopped and the GPU otherwise idle. Medians of the warm repetitions (2 and 3 of each arm, n=4); the first repetition is cold and left out.

| Metric | Old (0.5.6, `889a88d9`) | New (0.5.7+, `3a21685a`) | Change |
| --- | --- | --- | --- |
| prefill 8K (engine tok/s) | 4241 | 4775 | +12.6% |
| prefill 64K | 5631 | 5717 | +1.5% |
| prefill 250K (260,138 tokens, 262,144 ctx) | 4972 | 5324 | +7.1% |
| decode C=1 conv (tok/s, sampled, acc 72-76%) | 118.7 | 117.9 | -0.7% (noise: sd ~4) |
| decode C=1 struct (acc 98%) | 183.4 | 184.5 | +0.6% |
| decode C=1 prose (acc 70%) | 123.9 | 124.7 | +0.6% |
| decode C=4 total tok/s (`--parallel 4`, 4 streams, 2 runs each, A B B A) | 83.9 / 83.4 | 86.9 / 86.7 | +3.7% |
| short-prompt TTFT, server (s, 4 samples) | 0.35-0.37 | 0.26-0.33 | lower |

Acceptance and tokens per window are unchanged (same drafter and verify path; Flash-Next keeps its MTP drafter, the DSpark and DFly2 drafters belong to other models).
C=4 needs `vram-reserve-mib` 1500 and no `kv-resident`: with the production values both builds fail the batch graph capture (`instantiate: out of memory`), so production stays single-slot.

Functional checks (server on :11011, `func.py`, greedy): chat with thinking on, tool call and tool round trip, `json_object`, `json_schema`, vision (image), `/v1/messages`, a repeated greedy request identical: all pass on the new build.
The old build fails two that the new one passes: a top-level `enable_thinking: false` (0.5.7, #625) and `json_schema` output.
Needle in a haystack through the server: found at 133,099 tokens (50% of the context) and at 255,470 tokens (97%, the needle at 93% depth), 38.6 s for the long one. The 250K-token `basalt-bench` prefills ran 12 times without a fault. CPU parity tests (`kq_vnni_test`, `q50_vnni_test`, `pool_tasks_test`) and GPU parity tests (`dec_moe_parity --kq`, `native_expert_parity` x3) pass.

Config changes needed: none. The `.basalt` format stays 2 and the config stays version 2; `gpu: "0"` is unchanged. One build change: `-DBASALT_MODELS=all` is now required (#731); the fork's `update.sh` passes it.

Merge notes: three conflicts. `ggml_cpu_dispatch.cpp` (the fork's AVX-VNNI table kept beside upstream's NEON/arm64 dispatch), `native_expert.cpp` (the fork's kq256 gate/up and Q5_0 VNNI down rows kept beside upstream's new MXFP8 rows), `dec_moe_host.cuh` (the fork's Q4_K glue, type 12, kept beside upstream's type 8). Upstream covers none of the fork's patches yet.

Service: `fork.conf` now runs `~/src/basalt-0.5.7/bin-deployed/basalt` with the config still at `~/src/basalt-new/fork/active.json` (`basalt-profile` keeps switching that symlink).

Rollback (one line, about 1 minute of reload): `cp ~/bench/basalt-0.5.7/fork.conf.0.5.6-backup ~/.config/systemd/user/basalt.service.d/fork.conf && systemctl --user daemon-reload && systemctl --user restart basalt` (it runs `~/src/basalt-new/bin-deployed`, the 0.5.6 binary, untouched).
Raw results: `~/bench/basalt-0.5.7/` (`bench-*/result.json`, `func-*.json`, `needle-*.json`, `c4-*.log`).
