#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    printf 'Usage: %s CHECKPOINT [updates] [num-envs per GPU]\n' "$0"
    printf '%s\n' 'Pass the original model_1400.pt to start, or an evaluated stability checkpoint to continue.'
    exit 0
fi

updates="${2:-100}"
num_envs="${3:-128}"
save_interval="${ZLAB_SAVE_INTERVAL:-25}"
if (( $# < 1 || $# > 3 )) || [[ ! "$updates" =~ ^[1-9][0-9]*$ || ! "$num_envs" =~ ^[1-9][0-9]*$ || ! "$save_interval" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Supply a checkpoint and positive integer counts. See %s --help.\n' "$0" >&2
    exit 2
fi
checkpoint="$(realpath -e -- "$1")"

if revision="$(git rev-parse --short HEAD 2>/dev/null)"; then
    printf '[BALL STABILITY] Zlab-cloud commit %s\n' "$revision"
fi

log_root="${ZLAB_LOG_ROOT:-logs}"
mkdir -p -- "$log_root"
exec 9>"$log_root/ball-stability.lock"
flock -n 9 || { printf 'Ball-stability training is already running in this log directory.\n' >&2; exit 1; }

export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/zlab-warp-cache}"
export MJLAB_WARP_QUIET="${MJLAB_WARP_QUIET:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
gpu_args=()
if [[ "${ZLAB_USE_ALL_GPUS:-0}" == 1 ]]; then
    gpu_args=(--gpu-ids all)
fi
exec "${ZLAB_PYTHON:-python}" -u scripts/train.py \
    "${gpu_args[@]}" --checkpoint="$checkpoint" \
    --log-root="$log_root" --env.scene.num-envs="$num_envs" \
    --agent.max-iterations="$updates" --agent.save-interval="$save_interval" \
    --torso-push-scale="${ZLAB_TORSO_PUSH_SCALE:-0}" \
    --observation-noise-scale="${ZLAB_OBSERVATION_NOISE_SCALE:-0}" \
    --foot-friction-min="${ZLAB_FOOT_FRICTION_MIN:-0.35}" --foot-friction-max="${ZLAB_FOOT_FRICTION_MAX:-1.15}" \
    --agent.seed="${ZLAB_TRAIN_SEED:-42}" --agent.run-name=ball-stability-2p8
