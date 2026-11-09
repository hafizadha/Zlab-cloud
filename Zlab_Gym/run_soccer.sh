#!/usr/bin/env bash
set -euo pipefail

gym_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${ZLAB_SOCCER_PYTHON:-$gym_dir/../.venv-soccer/bin/python}"
mode="${1:-view}"

case "$mode" in
    -h|--help)
        printf '%s\n' \
            "Usage: $0 [view|train|eval] [extra options]" \
            'view   Open the MuJoCo viewer (default).' \
            'train  Resume field training for 200 PPO iterations.' \
            'eval   Measure goals and falls over 32 episodes.' \
            'Set ZLAB_SOCCER_CHECKPOINT to use a specific .pt file.'
        exit 0
        ;;
    view|train|eval) ;;
    *)
        printf 'Unknown mode %s. Run %s --help.\n' "$mode" "$0" >&2
        exit 2
        ;;
esac
if (($#)); then shift; fi

if [[ ! -x "$python_bin" ]]; then
    printf 'Soccer Python environment missing at %s\n' "$python_bin" >&2
    exit 1
fi

python_bin="$(realpath -s -- "$python_bin")"

if [[ -z "${ZLAB_SOCCER_CHECKPOINT:-}" ]]; then
    checkpoint_dir="$gym_dir/logs/rsl_rl/z01_soccer_approach"
    if [[ -d "$checkpoint_dir" ]]; then
        IFS= read -r -d '' ZLAB_SOCCER_CHECKPOINT < <(
            find "$checkpoint_dir" -mindepth 2 -maxdepth 2 -type f \
                -name 'model_[0-9]*.pt' -print0 | sort -zV | tail -z -n 1
        ) || true
    fi
fi
checkpoint="${ZLAB_SOCCER_CHECKPOINT:-}"
if [[ ! -f "$checkpoint" ]]; then
    printf 'Field checkpoint missing. Set ZLAB_SOCCER_CHECKPOINT to a .pt file.\n' >&2
    exit 1
fi
checkpoint="$(realpath -- "$checkpoint")"
cd "$gym_dir"
printf '[SOCCER] Using %s\n' "$checkpoint"

export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/zlab-warp-cache}"
export MJLAB_WARP_QUIET="${MJLAB_WARP_QUIET:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

task=Z01-StdAddHead-Soccer-Approach
case "$mode" in
    view)
        exec "$python_bin" -u scripts/play.py "$task" --checkpoint-file="$checkpoint" \
            --num-envs=1 --viewer=auto "$@"
        ;;
    train)
        exec "$python_bin" -u scripts/train.py "$task" --agent.resume True \
            --agent.load-checkpoint="$checkpoint" --env.scene.num-envs=128 \
            --agent.max-iterations=200 --agent.save-interval=16 --agent.run-name=field-continue "$@"
        ;;
    eval)
        exec "$python_bin" -u scripts/play.py "$task" --checkpoint-file="$checkpoint" \
            --num-envs=16 --eval-episodes=32 --seed=2026 "$@"
        ;;
esac
