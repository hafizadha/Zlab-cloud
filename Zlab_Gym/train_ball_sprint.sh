#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    printf 'Usage: %s [updates] [num-envs per GPU] [checkpoint]\n' "$0"
    exit 0
fi

updates="${1:-4500}"
num_envs="${2:-64}"
save_interval="${ZLAB_SAVE_INTERVAL:-500}"
if (( $# > 3 )) || [[ ! "$updates" =~ ^[1-9][0-9]*$ || ! "$num_envs" =~ ^[1-9][0-9]*$ || ! "$save_interval" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Updates, environment count, and ZLAB_SAVE_INTERVAL must be positive integers.\n' >&2
    exit 2
fi

log_root="${ZLAB_LOG_ROOT:-logs}"
mkdir -p -- "$log_root"
exec 9>"$log_root/ball-sprint.lock"
flock -n 9 || { printf 'Ball-sprint training is already running in this log directory.\n' >&2; exit 1; }

checkpoint="${3:-}"
if [[ -z "$checkpoint" ]]; then
    shopt -s nullglob
    best=-1
    for candidate in "$log_root"/rsl_rl/z01_ball_sprint/*/model_*.pt; do
        if [[ "${candidate##*/}" =~ ^model_([0-9]+)\.pt$ ]]; then
            iteration=$((10#${BASH_REMATCH[1]}))
            if (( iteration >= best )) && unzip -tqq "$candidate" >/dev/null 2>&1; then
                best="$iteration"
                checkpoint="$candidate"
            fi
        fi
    done
fi
checkpoint="$(realpath -e -- "${checkpoint:-checkpoints/model_15000.pt}")"
unzip -tqq "$checkpoint"
printf '[BALL SPRINT] Resuming %s for %s additional updates.\n' "$checkpoint" "$updates"

export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/zlab-warp-cache}"
export MJLAB_WARP_QUIET="${MJLAB_WARP_QUIET:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
gpu_args=()
if [[ "${ZLAB_USE_ALL_GPUS:-0}" == 1 ]]; then
    gpu_args=(--gpu-ids all)
fi
exec "${ZLAB_PYTHON:-python}" -u scripts/train.py Z01-StdAddHead-Soccer-BallSprint \
    "${gpu_args[@]}" --agent.resume True --agent.load-checkpoint="$checkpoint" \
    --log-root="$log_root" --env.scene.num-envs="$num_envs" \
    --agent.max-iterations="$updates" --agent.save-interval="$save_interval" \
    --agent.run-name=ball-sprint
