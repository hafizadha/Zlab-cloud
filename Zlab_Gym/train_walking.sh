#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    printf '%s\n' "Usage: $0 [--fast | --stride | --long-stride | --robust STAGE | --recovery] [updates] [num-envs] [checkpoint]" \
        'Fine-tune the bundled Zoomlion walking actor for 500 updates with 64 environments.' \
        'By default, resume the furthest usable checkpoint from the walking experiment.' \
        'Use --fast to build on a walking checkpoint and practice speeds up to 1.6 m/s.' \
        'Use --stride to build on a fast-walking checkpoint and practice longer, quicker steps.' \
        'Use --long-stride to continue a stride checkpoint with longer steps and quicker swings.' \
        'Use --robust 1, 2, or 3 to train with friction and damping, pushes, then uneven ground.' \
        'Use --recovery to continue stage 2 with balance recovery practice for 1,500 updates.'
    exit 0
fi

task=Z01-StdAddHead-Velocity-Walk
run_name=walking
experiments=(z01_walking)
default_updates=500
if [[ "${1:-}" == --fast ]]; then
    shift
    task=Z01-StdAddHead-Velocity-FastWalk
    run_name=fast-walking
    experiments=(z01_fast_walking z01_walking)
elif [[ "${1:-}" == --stride ]]; then
    shift
    task=Z01-StdAddHead-Velocity-StrideWalk
    run_name=stride-walking
    experiments=(z01_stride_walking z01_fast_walking)
elif [[ "${1:-}" == --long-stride ]]; then
    shift
    task=Z01-StdAddHead-Velocity-LongStrideWalk
    run_name=long-stride-walking
    experiments=(z01_long_stride_walking z01_stride_walking)
elif [[ "${1:-}" == --robust ]]; then
    stage="${2:-}"
    if [[ ! "$stage" =~ ^[123]$ ]]; then
        printf 'Robust stage must be 1, 2, or 3.\n' >&2
        exit 2
    fi
    shift 2
    task="Z01-StdAddHead-Velocity-RobustWalk$stage"
    run_name="robust-walking-$stage"
    experiments=("z01_robust_walking_$stage")
    if [[ "$stage" == 1 ]]; then
        experiments+=(z01_stride_walking)
    elif [[ "$stage" == 2 ]]; then
        # Prefer a Stage 2 save before looking at Stage 1 runs.
        bundled_checkpoint=checkpoints/robust-walking-2-model_7300.pt
        experiments+=(z01_robust_walking_1)
    else
        experiments+=("z01_robust_walking_$((stage - 1))")
    fi
elif [[ "${1:-}" == --recovery ]]; then
    shift
    task=Z01-StdAddHead-Velocity-BalanceRecovery
    run_name=balance-recovery
    experiments=(z01_balance_recovery z01_robust_walking_2)
    default_updates=1500
fi

updates="${1:-$default_updates}"
num_envs="${2:-64}"
if (( $# > 3 )) || [[ ! "$updates" =~ ^[1-9][0-9]*$ || ! "$num_envs" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Usage: %s [--fast | --stride | --long-stride | --robust STAGE | --recovery] [updates] [num-envs] [checkpoint]\n' "$0" >&2
    exit 2
fi

log_root="${ZLAB_LOG_ROOT:-logs}"
mkdir -p -- "$log_root"
exec 9>"$log_root/walking.lock"
flock -n 9 || { printf 'Walking training is already running for this log directory.\n' >&2; exit 1; }

checkpoint="${3:-}"
if [[ -z "$checkpoint" ]]; then
    shopt -s nullglob
    for experiment in "${experiments[@]}"; do
        best=-1
        for candidate in "$log_root/rsl_rl/$experiment/"*/model_*.pt; do
            if [[ "${candidate##*/}" =~ ^model_([0-9]+)\.pt$ ]]; then
                iteration=$((10#${BASH_REMATCH[1]}))
                if (( iteration >= best )) && unzip -tqq "$candidate" >/dev/null 2>&1; then
                    best="$iteration"
                    checkpoint="$candidate"
                fi
            fi
        done
        if [[ -n "$checkpoint" ]]; then break; fi
        if [[ "$experiment" == z01_robust_walking_2 && -n "${bundled_checkpoint:-}" ]] &&
            unzip -tqq "$bundled_checkpoint" >/dev/null 2>&1; then
            checkpoint="$bundled_checkpoint"
            break
        fi
    done
fi
if [[ -n "$checkpoint" ]]; then
    checkpoint="$(realpath -e -- "$checkpoint")"
    unzip -tqq "$checkpoint"
    start_args=(--agent.resume True --agent.load-checkpoint="$checkpoint")
    printf '[WALK] Resuming %s for %s additional updates.\n' "$checkpoint" "$updates"
else
    if [[ "$run_name" != walking ]]; then
        printf '%s needs a saved checkpoint from the preceding stage.\n' "$run_name" >&2
        exit 1
    fi
    start_args=(--warm-start-walking True)
    printf '[WALK] Starting from the bundled Zoomlion walking policy for %s updates.\n' "$updates"
fi

export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/zlab-warp-cache}"
export MJLAB_WARP_QUIET="${MJLAB_WARP_QUIET:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
python_bin="${ZLAB_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    if [[ -x ../.venv-soccer/bin/python ]]; then
        python_bin=../.venv-soccer/bin/python
    else
        python_bin=python
    fi
fi
gpu_args=()
if [[ "${ZLAB_USE_ALL_GPUS:-0}" == 1 ]]; then
    gpu_args=(--gpu-ids all)
fi
exec "$python_bin" -u scripts/train.py "$task" \
    "${gpu_args[@]}" "${start_args[@]}" --log-root="$log_root" --env.scene.num-envs="$num_envs" \
    --agent.max-iterations="$updates" --agent.save-interval=100 --agent.run-name="$run_name"
