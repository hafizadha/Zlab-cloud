"""Continue PPO across bounded blocks and retain the best evaluated checkpoint."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


PROJECT = Path(__file__).resolve().parents[1]
# Torso push and sensor noise scales. Every block runs at 2.8 m/s.
PROFILES = ((0.0, 0.0), (0.25, 0.25), (0.5, 0.5), (1.0, 1.0))
EPISODES = 48
EVALUATION_SEED = 2401
PILOT_SEEDS = (EVALUATION_SEED, 9187)


def evaluate(checkpoint, block_dir, name, profile, *, seed=EVALUATION_SEED, lateral_correction=False):
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
        "--lateral-correction", str(lateral_correction),
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


def train_pilot(checkpoint, work_dir, blocks, updates, num_envs, control=None, *, lateral_pilot=False):
    """Evaluate every save and stop after the first block without a better policy."""
    state_file = work_dir / "progress.json"
    finishing_pilot = control is not None
    mode = "lateral" if lateral_pilot else "finishing" if finishing_pilot else "pilot"
    friction = [os.environ.get("ZLAB_FOOT_FRICTION_MIN", "0.35"),
                os.environ.get("ZLAB_FOOT_FRICTION_MAX", "1.15")]
    if state_file.exists():
        state = json.loads(state_file.read_text())
        if state.get("mode") != mode:
            raise ValueError("Use a separate work directory for the stability pilot.")
        if state["foot_friction"] != friction:
            raise ValueError("Use a fresh pilot work directory when changing foot friction.")
    else:
        baseline = work_dir / "baseline.pt"
        shutil.copyfile(checkpoint, baseline)
        state = {"mode": mode,
                 "checkpoint": str(baseline), "reports": None,
                 "block": 0, "stop_reason": None, "foot_friction": friction}
        save_state(state_file, state)

    def evaluate_both(path, directory, name):
        return [
            evaluate(path, directory, f"{name}-seed-{seed}", PROFILES[0],
                     seed=seed, lateral_correction=lateral_pilot)
            for seed in PILOT_SEEDS
        ]

    if state["stop_reason"] == "no_improvement":
        print(f"[BALL STABILITY] Pilot already stopped. Checkpoint: {state['checkpoint']}", flush=True)
        return

    for block in range(state["block"] + 1, blocks + 1):
        directory = Path(tempfile.mkdtemp(prefix=f"pilot-{block:03d}-", dir=work_dir))
        incumbent = Path(state["checkpoint"])
        if state["reports"] is None:
            state["reports"] = evaluate_both(incumbent, directory, "baseline")
            save_state(state_file, state)
        if finishing_pilot and "control_reports" not in state:
            state["control_reports"] = evaluate_both(control, directory, "block-50-control")
            save_state(state_file, state)
        runtime_env = dict(
            os.environ, ZLAB_PYTHON=sys.executable, ZLAB_LOG_ROOT=str(directory),
            ZLAB_TRAIN_SEED=str(41 + block), ZLAB_TORSO_PUSH_SCALE="0",
            ZLAB_OBSERVATION_NOISE_SCALE="0", ZLAB_SAVE_INTERVAL="100",
            ZLAB_LATERAL_CORRECTION=str(lateral_pilot),
            ZLAB_FINISHING_START_FRACTION="0.35" if finishing_pilot else "0",
        )
        subprocess.run([
            str(PROJECT / "train_ball_stability.sh"), str(incumbent), str(updates), str(num_envs),
        ], cwd=PROJECT, env=runtime_env, check=True)
        candidates = sorted(
            directory.glob("rsl_rl/*/*/model_*.pt"), key=lambda path: int(path.stem.split("_")[-1])
        )
        if not candidates:
            raise RuntimeError(f"Training completed without a checkpoint in {directory}")

        qualified = []
        comparisons = []
        for candidate in candidates:
            reports = evaluate_both(candidate, directory, candidate.stem)
            improved = all(
                new["stops"] > old["stops"]
                and new["lane_offset_p95_m"] <= old["lane_offset_p95_m"] + (
                    0.05 if finishing_pilot else -0.05
                )
                and (not finishing_pilot or new["timeouts"] < old["timeouts"])
                and (
                    not lateral_pilot
                    or new["timeout_outside_lateral_band"] < old["timeout_outside_lateral_band"]
                )
                and new["falls"] <= old["falls"]
                and new["touches"] <= old["touches"]
                for old, new in zip(state["reports"], reports, strict=True)
            )
            comparisons.append({"checkpoint": str(candidate), "reports": reports,
                                "improved_on_both_seeds": improved})
            if improved:
                qualified.append((candidate, reports))
            print(f"[BALL STABILITY] {candidate.name}: "
                  f"{'improved on both seeds' if improved else 'missed the improvement criteria'}",
                  flush=True)

        save_state(directory / "comparison.json", {
            "seeds": PILOT_SEEDS, "baseline": str(incumbent),
            "baseline_reports": state["reports"], "candidates": comparisons,
            **({"block_50_control_reports": state["control_reports"]} if finishing_pilot else {}),
        })
        state.update(block=block, latest_comparison=str(directory / "comparison.json"))
        if not qualified:
            state["stop_reason"] = "no_improvement"
            save_state(state_file, state)
            print(f"[BALL STABILITY] Stopping: no save improved on both seeds. "
                  f"Kept {incumbent}.", flush=True)
            return

        selected, reports = max(qualified, key=lambda entry: (
            sum(report["stops"] for report in entry[1]),
            -sum(report["lane_offset_p95_m"] for report in entry[1]),
        ))
        state.update(checkpoint=str(selected), reports=reports, stop_reason=None)
        save_state(state_file, state)
        print(f"[BALL STABILITY] Selected {selected}.", flush=True)
    print(f"[BALL STABILITY] Pilot block limit reached. Checkpoint: {state['checkpoint']}", flush=True)


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
    parser.add_argument("--pilot-blocks", type=int, metavar="N")
    parser.add_argument("--lateral-pilot", action="store_true")
    parser.add_argument("--finishing-control", type=Path, metavar="CHECKPOINT",
                        help="Run the finishing curriculum and evaluate this reference checkpoint.")
    args = parser.parse_args()
    if not 0 < args.hours < float("inf") or min(args.updates, args.num_envs) <= 0:
        parser.error("Use a finite, positive time budget and positive update and environment counts.")
    if args.pilot_blocks is not None and args.pilot_blocks <= 0:
        parser.error("--pilot-blocks must be positive.")
    if args.finishing_control is not None and args.pilot_blocks is None:
        parser.error("--finishing-control requires --pilot-blocks.")
    if args.lateral_pilot and args.finishing_control is None:
        parser.error("--lateral-pilot requires --finishing-control.")
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    with (work_dir / "training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"Training is already running in {work_dir}.")
        checkpoint = args.checkpoint.expanduser().resolve(strict=True)
        if args.pilot_blocks is None:
            train_blocks(checkpoint, work_dir, args.hours, args.updates, args.num_envs)
        else:
            control = args.finishing_control
            if control is not None:
                control = control.expanduser().resolve(strict=True)
            train_pilot(checkpoint, work_dir, args.pilot_blocks, args.updates, args.num_envs,
                        control, lateral_pilot=args.lateral_pilot)


if __name__ == "__main__":
    main()
