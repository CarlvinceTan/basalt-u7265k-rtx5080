# Fork ledger

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
