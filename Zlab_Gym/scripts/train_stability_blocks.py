"""Continue PPO across bounded blocks and retain the best evaluated checkpoint."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


PROJECT = Path(__file__).resolve().parents[1]
# Torso push and sensor noise scales. Every block runs at 2.8 m/s.
PROFILES = ((0.0, 0.0), (0.25, 0.25), (0.5, 0.5), (1.0, 1.0))
EPISODES = 48
EVALUATION_SEED = 2401


def evaluate(checkpoint, block_dir, name, profile, *, seed=EVALUATION_SEED):
    metrics_file = block_dir / f"{name}.json"
    push, noise = profile
    subprocess.run([
        sys.executable, str(PROJECT / "scripts/play.py"),
        "--checkpoint-file", str(checkpoint),
        "--num-envs", str(EPISODES), "--eval-episodes", str(EPISODES),
        "--device", "cuda:0", "--seed", str(seed), "--disturbances", "True",
        "--torso-push-scale", str(push), "--observation-noise-scale", str(noise),
        "--foot-friction-min", os.environ.get("ZLAB_FOOT_FRICTION_MIN", "0.35"),
        "--foot-friction-max", os.environ.get("ZLAB_FOOT_FRICTION_MAX", "1.15"),
        "--metrics-file", str(metrics_file),
    ], cwd=PROJECT, check=True)
    return json.loads(metrics_file.read_text())


def passes_gate(metrics):
    return (
        metrics["stops"] >= 46
        and metrics["touches"] == 0
        and metrics["falls"] <= 1
        and metrics["lane_offset_p95_m"] <= 0.30
        and metrics["heading_error_p95_deg"] <= 15
    )


def rank(metrics):
    return (
        passes_gate(metrics), metrics["stops"],
        -metrics["touches"], -metrics["falls"],
        -metrics["lane_offset_p95_m"], -metrics["heading_error_p95_deg"],
    )


def save_state(state_file, state):
    pending = state_file.with_suffix(".tmp")
    pending.write_text(json.dumps(state, indent=2) + "\n")
    pending.replace(state_file)


def train_blocks(checkpoint, work_dir, hours, updates, num_envs):
    work_dir.mkdir(parents=True, exist_ok=True)
    state_file = work_dir / "progress.json"
    state = {
        "checkpoint": str(checkpoint), "training_checkpoint": str(checkpoint),
        "block": 0, "profile": 0, "metrics": None, "complete": False,
    }
    if state_file.exists():
        state = json.loads(state_file.read_text())
        state.setdefault("training_checkpoint", state["checkpoint"])
    deadline = time.monotonic() + hours * 3600
    block_seconds = 0.0

    # A restart with different traction settings must re-establish the gates.
    friction = [os.environ.get("ZLAB_FOOT_FRICTION_MIN", "0.35"),
                os.environ.get("ZLAB_FOOT_FRICTION_MAX", "1.15")]
    if state.get("foot_friction") != friction:
        state.update(foot_friction=friction, profile=0, metrics=None, complete=False,
                     training_checkpoint=state["checkpoint"], last_gate_check=None)

    while not state["complete"] and time.monotonic() + block_seconds < deadline:
        started = time.monotonic()
        block = state["block"] + 1
        profile = PROFILES[state["profile"]]
        block_dir = Path(tempfile.mkdtemp(prefix=f"block-{block:03d}-", dir=work_dir))
        incumbent = Path(state["checkpoint"])
        if state["metrics"] is None:
            state["metrics"] = evaluate(incumbent, block_dir, "incumbent", profile)
        training_checkpoint = state["training_checkpoint"]

        runtime_env = dict(
            os.environ,
            ZLAB_PYTHON=sys.executable,
            ZLAB_LOG_ROOT=str(block_dir),
            ZLAB_TRAIN_SEED=str(41 + block),
            ZLAB_TORSO_PUSH_SCALE=str(profile[0]),
            ZLAB_OBSERVATION_NOISE_SCALE=str(profile[1]),
        )
        print(f"[BALL STABILITY] Block {block}, push/noise {profile}, starting from {training_checkpoint}", flush=True)
        subprocess.run([
            str(PROJECT / "train_ball_stability.sh"), training_checkpoint, str(updates), str(num_envs),
        ], cwd=PROJECT, env=runtime_env, check=True)
        candidate = max(
            block_dir.glob("rsl_rl/*/*/model_*.pt"),
            key=lambda path: int(path.stem.split("_")[-1]),
        )
        metrics = evaluate(candidate, block_dir, "candidate", profile)
        state.update(training_checkpoint=str(candidate), latest_metrics=metrics)
        if rank(metrics) > rank(state["metrics"]):
            # Check that the candidate still works without the extra disturbances.
            if state["profile"] == 0 or passes_gate(
                evaluate(candidate, block_dir, "candidate-clean", PROFILES[0])
            ):
                state.update(checkpoint=str(candidate), metrics=metrics)

        selected = Path(state["checkpoint"])
        if passes_gate(metrics):
            gate_checkpoint, gate_metrics = candidate, metrics
        else:
            gate_checkpoint, gate_metrics = selected, state["metrics"]
        if passes_gate(gate_metrics) and str(gate_checkpoint) != state.get("last_gate_check"):
            held_out = evaluate(gate_checkpoint, block_dir, "held-out", profile, seed=9187)
            clean = (held_out if profile == PROFILES[0] else
                     evaluate(gate_checkpoint, block_dir, "clean", PROFILES[0], seed=9187))
            if gate_checkpoint == selected:
                state["last_gate_check"] = str(selected)
            if passes_gate(held_out) and passes_gate(clean):
                state.update(checkpoint=str(gate_checkpoint), metrics=gate_metrics)
                if state["profile"] == len(PROFILES) - 1:
                    state["complete"] = True
                else:
                    state.update(profile=state["profile"] + 1, metrics=None,
                                 training_checkpoint=str(gate_checkpoint), last_gate_check=None)

        state["block"] = block
        save_state(state_file, state)
        print(f"[BALL STABILITY] Block {block} result: "
              f"{metrics['stops']}/{EPISODES} clean stops, {metrics['falls']} falls, "
              f"{metrics['touches']} touches. Best evaluated checkpoint {state['checkpoint']}. "
              f"Training checkpoint {state['training_checkpoint']}.", flush=True)
        block_seconds = max(block_seconds, time.monotonic() - started)

    print(f"[BALL STABILITY] {'Passed all profiles' if state['complete'] else 'Time budget reached'}. "
          f"Selected checkpoint {state['checkpoint']}. Progress: {state_file}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=9.0)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--num-envs", type=int, default=512)
    args = parser.parse_args()
    if not 0 < args.hours < float("inf") or min(args.updates, args.num_envs) <= 0:
        parser.error("Use a finite, positive time budget and positive update and environment counts.")
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    with (work_dir / "training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"Training is already running in {work_dir}.")
        train_blocks(args.checkpoint.expanduser().resolve(strict=True),
                     work_dir, args.hours, args.updates, args.num_envs)


if __name__ == "__main__":
    main()
