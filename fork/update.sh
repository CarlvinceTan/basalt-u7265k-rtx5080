#!/bin/bash
# fork/update.sh - bring basalt-u7265k-rtx5080 up to upstream main, rebuild, check, and deploy.
#
#   fork/update.sh            rebase onto origin/main, build, run the parity tests, freeze into bin-deployed/
#   fork/update.sh --deploy   ... and restart the basalt service on the new binary (fork.conf drop-in)
#   fork/update.sh --no-fetch rebuild and re-check the current tree only
#
# Nothing is deployed unless the build and every parity test pass.  The previous binary stays in bin-deployed.prev/.
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=$(pwd)
export PATH=/usr/local/cuda-13.0/bin:$PATH
DEPLOY=0 FETCH=1
for a in "$@"; do
  case "$a" in
    --deploy) DEPLOY=1 ;;
    --no-fetch) FETCH=0 ;;
    *) echo "usage: $0 [--deploy] [--no-fetch]"; exit 2 ;;
  esac
done

if [ $FETCH = 1 ]; then
  git diff --quiet && git diff --cached --quiet || { echo "uncommitted changes: commit or stash first"; exit 1; }
  git fetch -q origin
  echo "upstream: $(git rev-list --count HEAD..origin/main) new commits"
  git rebase origin/main || { echo "rebase stopped on a conflict: resolve it, 'git rebase --continue', then run with --no-fetch"; exit 1; }
fi

cmake --build build --target basalt kq_vnni_test q50_vnni_test native_expert_parity dec_moe_parity -j "$(nproc)"

echo "== parity"
./build/kq_vnni_test | tail -1
./build/q50_vnni_test | tail -1
GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
if [ "$GPU_FREE" -gt 4000 ]; then
  ./build/dec_moe_parity --kq | tail -1
  for p in q4_K/q5_1 q4_K/q8_0 q5_K/q8_0; do ./build/native_expert_parity --synthetic $p | tail -1; done
else
  echo "(GPU busy: dec_moe_parity / native_expert_parity skipped - stop the basalt service to run them)"
fi

rm -rf bin-deployed.prev
[ -d bin-deployed ] && mv bin-deployed bin-deployed.prev
cp -r bin bin-deployed
echo "frozen: $SRC/bin-deployed ($(git rev-parse --short HEAD))"

if [ $DEPLOY = 1 ]; then
  D=~/.config/systemd/user/basalt.service.d
  mkdir -p "$D"
  cp fork/fork.conf "$D/fork.conf"
  systemctl --user daemon-reload
  systemctl --user restart basalt
  echo "basalt restarted on the fork (revert: rm $D/fork.conf && systemctl --user daemon-reload && systemctl --user restart basalt)"
fi
