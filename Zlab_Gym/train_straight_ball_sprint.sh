#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    printf 'Usage: %s [updates] [num-envs per GPU] [straight-sprint checkpoint to resume]\n' "$0"
    printf '%s\n' 'Set ZLAB_HF_REPO_ID to owner/model-repo to mirror checkpoints to Hugging Face.'
    exit 0
fi

updates="${1:-500}"
num_envs="${2:-64}"
save_interval="${ZLAB_SAVE_INTERVAL:-100}"
if (( $# > 3 )) || [[ ! "$updates" =~ ^[1-9][0-9]*$ || ! "$num_envs" =~ ^[1-9][0-9]*$ || ! "$save_interval" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Updates, environment count, and ZLAB_SAVE_INTERVAL must be positive integers.\n' >&2
    exit 2
fi

if revision="$(git rev-parse --short HEAD 2>/dev/null)"; then
    printf '[BALL SPRINT] Zlab-cloud commit %s\n' "$revision"
fi

log_root="${ZLAB_LOG_ROOT:-logs}"
mkdir -p -- "$log_root"
exec 9>"$log_root/ball-sprint.lock"
flock -n 9 || { printf 'Ball-sprint training is already running in this log directory.\n' >&2; exit 1; }

checkpoint="${3:-}"
if [[ -n "$checkpoint" ]]; then
    checkpoint="$(realpath -e -- "$checkpoint")"
    unzip -tqq "$checkpoint"
    checkpoint_args=(--agent.resume True --agent.load-checkpoint="$checkpoint")
    printf '[BALL SPRINT] Resuming %s for %s additional updates.\n' "$checkpoint" "$updates"
else
    checkpoint="$(realpath -e -- checkpoints/ball_sprint_seed.pt)"
    unzip -tqq "$checkpoint"
    checkpoint_args=(--init-actor-checkpoint="$checkpoint")
    printf '[BALL SPRINT] Starting from the bundled actor for %s updates.\n' "$updates"
fi

export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/zlab-warp-cache}"
export MJLAB_WARP_QUIET="${MJLAB_WARP_QUIET:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
gpu_args=()
if [[ "${ZLAB_USE_ALL_GPUS:-0}" == 1 ]]; then
    gpu_args=(--gpu-ids all)
fi
exec "${ZLAB_PYTHON:-python}" -u scripts/train.py \
    "${gpu_args[@]}" "${checkpoint_args[@]}" \
    --log-root="$log_root" --env.scene.num-envs="$num_envs" \
    --agent.max-iterations="$updates" --agent.save-interval="$save_interval" \
    --command-speed="${ZLAB_COMMAND_SPEED:-2.8}" --torso-push-scale="${ZLAB_TORSO_PUSH_SCALE:-0}" \
    --agent.run-name=short-ball-sprint-4mps
