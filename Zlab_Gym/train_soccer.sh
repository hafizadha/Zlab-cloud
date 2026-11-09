#!/usr/bin/env bash
set -euo pipefail

gym_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    printf '%s\n' \
        "Usage: $0 [iterations] [extra training options]" \
        'Resume field training for 6,000 PPO updates by default.' \
        'Save a checkpoint every 50 iterations and log console output under logs/soccer-console/.'
    exit 0
fi

iterations=6000
if (($#)) && [[ "$1" != --* ]]; then
    iterations="$1"
    shift
fi
if [[ ! "$iterations" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Iterations must be a positive whole number.\n' >&2
    exit 2
fi

console_dir="$gym_dir/logs/soccer-console"
mkdir -p -- "$console_dir"
log_file="$console_dir/$(date +%Y-%m-%d_%H-%M-%S)_$$.log"

printf '[SOCCER] Training for %s PPO updates.\n' "$iterations"
printf '[SOCCER] Console log %s\n' "$log_file"
"$gym_dir/run_soccer.sh" train --agent.max-iterations="$iterations" \
    --agent.save-interval=50 --agent.run-name=field-long "$@" 2>&1 | tee "$log_file"
