#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    printf '%s\n' "Usage: $0 [--fast | --stride] [updates] [num-envs] [checkpoint]" \
        'Fine-tune the bundled Zoomlion walking actor for 500 updates with 64 environments.' \
        'By default, resume the furthest usable checkpoint from the walking experiment.' \
        'Use --fast to build on a walking checkpoint and practice speeds up to 1.6 m/s.' \
        'Use --stride to build on a fast-walking checkpoint and practice longer, quicker steps.'
    exit 0
fi

task=Z01-StdAddHead-Velocity-Walk
run_name=walking
experiments=(z01_walking)
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
fi

updates="${1:-500}"
num_envs="${2:-64}"
if (( $# > 3 )) || [[ ! "$updates" =~ ^[1-9][0-9]*$ || ! "$num_envs" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Usage: %s [--fast | --stride] [updates] [num-envs] [checkpoint]\n' "$0" >&2
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
exec "$python_bin" -u scripts/train.py "$task" \
    "${start_args[@]}" --log-root="$log_root" --env.scene.num-envs="$num_envs" \
    --agent.max-iterations="$updates" --agent.save-interval=100 --agent.run-name="$run_name"
