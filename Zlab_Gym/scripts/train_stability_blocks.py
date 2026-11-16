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
VALIDATION_SEEDS = (27491, 36187)


def evaluate(checkpoint, block_dir, name, profile, *, seed=EVALUATION_SEED,
             lateral_correction=False, early_braking=False, recovery_starts=False):
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
        "--early-braking", str(early_braking),
        "--recovery-starts", str(recovery_starts),
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


def validation_criteria(candidate, baseline):
    paired = list(zip(candidate, baseline, strict=True))
    return {
        "more_clean_stops_overall": sum(r["stops"] for r in candidate) > sum(r["stops"] for r in baseline),
        "clean_stops_no_regression_by_seed": all(new["stops"] >= old["stops"] for new, old in paired),
        "falls_no_regression_by_seed": all(new["falls"] <= old["falls"] for new, old in paired),
        "no_more_ball_touches_overall": sum(r["touches"] for r in candidate) <= sum(r["touches"] for r in baseline),
        "lane_offset_no_regression_by_seed": all(
            new["lane_offset_p95_m"] <= old["lane_offset_p95_m"] for new, old in paired
        ),
    }


def validate_candidate(checkpoint, control, work_dir, alternative=None):
    seeds = VALIDATION_SEEDS if alternative is None else (46219, 53831)
    report = {
        "seeds": seeds,
        "episodes_per_seed": EPISODES,
        "foot_friction": [float(os.environ.get("ZLAB_FOOT_FRICTION_MIN", "0.35")),
                          float(os.environ.get("ZLAB_FOOT_FRICTION_MAX", "1.15"))],
    }
    checkpoints = {"control": control, "candidate": checkpoint}
    if alternative is not None:
        checkpoints["alternative"] = alternative
    for name, path in checkpoints.items():
        report[name] = {"checkpoint": str(path), "reports": [
            evaluate(path, work_dir, f"{name}-seed-{seed}", PROFILES[0],
                     seed=seed, lateral_correction=True)
            for seed in seeds
        ]}

    qualified = []
    for name in checkpoints.keys() - {"control"}:
        reports = report[name]["reports"]
        criteria = validation_criteria(reports, report["control"]["reports"])
        report[name]["promotion_criteria"] = criteria
        if all(criteria.values()):
            rank = (sum(r["stops"] for r in reports), -sum(r["falls"] for r in reports),
                    -sum(r["touches"] for r in reports), -sum(r["lane_offset_p95_m"] for r in reports))
            qualified.append((rank, name))
    report["promotion_criteria"] = report["candidate"]["promotion_criteria"]
    if alternative is not None:
        report["diagnostic_seed"] = EVALUATION_SEED
        for name in ("control", "candidate"):
            report[name]["diagnostic_report"] = evaluate(
                checkpoints[name], work_dir, f"{name}-diagnostic-{EVALUATION_SEED}",
                PROFILES[0], seed=EVALUATION_SEED, lateral_correction=True,
            )
    selected_name = max(qualified)[1] if qualified else "control"
    selected = checkpoints[selected_name]
    report["selected_checkpoint"] = str(selected)
    shutil.copyfile(selected, work_dir / "selected.pt")
    save_state(work_dir / "validation.json", report)
    print(f"[BALL STABILITY] Selected {selected.name} after fresh-seed validation.", flush=True)


def arm_posture_improved(candidate, baseline):
    return (
        candidate["stops_with_relaxed_arms"] > baseline["stops_with_relaxed_arms"]
        and candidate["stops"] >= baseline["stops"]
        and candidate["falls"] <= baseline["falls"]
        and candidate["touches"] <= baseline["touches"]
        and candidate["lane_offset_p95_m"] <= baseline["lane_offset_p95_m"] + 0.05
    )


def cohort_total(reports, key):
    return sum(report[key] for report in reports)


def arm_error(reports):
    stops = cohort_total(reports, "stops")
    if not stops:
        return float("inf")
    return sum(report["mean_final_arm_error_rad"] * report["stops"]
               for report in reports if report["stops"]) / stops


def staged_pilot_passes(candidate, baseline, reference, *, arms):
    outcome_keys = ("stops", "falls", "touches")
    improvement = False
    for cohort in (slice(0, 2), slice(2, None)):
        new, old, control = candidate[cohort], baseline[cohort], reference[cohort]
        new_stops, new_falls, new_touches = (cohort_total(new, k) for k in outcome_keys)
        old_stops, old_falls, old_touches = (cohort_total(old, k) for k in outcome_keys)
        if (new_stops < old_stops
                or new_falls > min(old_falls, cohort_total(control, "falls"))
                or new_touches > min(old_touches, cohort_total(control, "touches"))
                or any(a["lane_offset_p95_m"] > b["lane_offset_p95_m"] + 0.05
                       for a, b in zip(new, old, strict=True))):
            return False
        if arms:
            new_arms = cohort_total(new, "stops_with_relaxed_arms")
            old_arms = cohort_total(old, "stops_with_relaxed_arms")
            if new_arms < old_arms or arm_error(new) >= arm_error(old):
                return False
            improvement |= new_arms > old_arms
        else:
            improvement |= new_stops > old_stops or new_falls < old_falls or new_touches < old_touches
    return improvement


def train_braking_then_arms(checkpoint, reference, work_dir, updates, num_envs, *, seeds, train_seed):
    """Run two conditional PPO blocks with separate regular and recovery cohorts."""
    progress = work_dir / "progress.json"
    baseline_file = work_dir / "baseline.pt"
    reference_file = work_dir / "reference.pt"
    if progress.exists():
        state = json.loads(progress.read_text())
        if state["mode"] != "braking_then_arms" or state["seeds"] != list(seeds):
            raise ValueError("Use a fresh work directory for a different pilot mode or seed set.")
    else:
        shutil.copyfile(checkpoint, baseline_file)
        shutil.copyfile(reference, reference_file)
        state = {"mode": "braking_then_arms", "checkpoint": str(baseline_file), "block": 0,
                 "complete": False, "stop_reason": None, "seeds": list(seeds)}
    reference = reference_file
    if state["complete"]:
        print(f"[BALL STABILITY] Pilot already completed. Selected {state['checkpoint']}.", flush=True)
        return

    def evaluate_suite(path, directory, name):
        return [evaluate(
            path, directory, f"{name}-{cohort}-{seed}", PROFILES[0], seed=seed,
            lateral_correction=True, early_braking=True, recovery_starts=(cohort == "recovery"),
        ) for seed, cohort in zip(seeds, ("regular", "regular", "recovery"), strict=True)]

    if "reference_reports" not in state:
        state["reference_reports"] = evaluate_suite(reference, work_dir, "reference")
        state["reports"] = evaluate_suite(baseline_file, work_dir, "baseline")
        save_state(progress, state)

    for block in range(state["block"] + 1, 3):
        arms = block == 2
        stage = "arms" if arms else "braking"
        directory = Path(tempfile.mkdtemp(prefix=f"{stage}-", dir=work_dir))
        incumbent = Path(state["checkpoint"])
        subprocess.run([
            str(PROJECT / "train_ball_stability.sh"), str(incumbent), str(updates), str(num_envs),
        ], cwd=PROJECT, check=True, env=dict(
            os.environ, ZLAB_PYTHON=sys.executable, ZLAB_LOG_ROOT=str(directory),
            ZLAB_TRAIN_SEED=str(train_seed + block - 1), ZLAB_TORSO_PUSH_SCALE="0",
            ZLAB_OBSERVATION_NOISE_SCALE="0", ZLAB_SAVE_INTERVAL="100",
            ZLAB_LATERAL_CORRECTION="True", ZLAB_EARLY_BRAKING="True",
            ZLAB_FINISHING_START_FRACTION="0.35", ZLAB_RECOVERY_START_FRACTION="0.20",
            ZLAB_RELAXED_STOP_ARMS=str(arms),
        ))
        candidates = sorted(directory.glob("rsl_rl/*/*/model_*.pt"))
        if not candidates:
            raise RuntimeError(f"Training completed without a checkpoint in {directory}")
        comparisons = []
        for candidate in candidates:
            reports = evaluate_suite(candidate, directory, candidate.stem)
            passed = staged_pilot_passes(reports, state["reports"], state["reference_reports"], arms=arms)
            comparisons.append({"checkpoint": str(candidate), "reports": reports, "passed": passed})
        comparison = directory / "comparison.json"
        save_state(comparison, {
            "stage": stage, "seeds": seeds, "cohorts": ["regular", "regular", "recovery"],
            "baseline": str(incumbent), "baseline_reports": state["reports"],
            "reference": str(reference), "reference_reports": state["reference_reports"],
            "candidates": comparisons,
        })
        qualified = [entry for entry in comparisons if entry["passed"]]
        state.update(block=block, latest_comparison=str(comparison))
        if not qualified:
            if not arms:
                state.update(checkpoint=str(reference), reports=state["reference_reports"])
            state.update(stop_reason=f"no_{stage}_improvement", complete=True)
            save_state(progress, state)
            print(f"[BALL STABILITY] No {stage} save qualified. Selected {state['checkpoint']}.", flush=True)
            return
        selected = max(qualified, key=lambda entry: (
            cohort_total(entry["reports"], "stops"),
            -cohort_total(entry["reports"], "falls"),
            -cohort_total(entry["reports"], "touches"),
            cohort_total(entry["reports"], "stops_with_relaxed_arms") if arms else 0,
            -arm_error(entry["reports"]) if arms else 0,
        ))
        state.update(checkpoint=selected["checkpoint"], reports=selected["reports"], complete=arms)
        save_state(progress, state)
        print(f"[BALL STABILITY] Selected {state['checkpoint']} after {stage}.", flush=True)


def train_pilot(checkpoint, work_dir, blocks, updates, num_envs, control=None, *,
                lateral_pilot=False, seeds=PILOT_SEEDS, train_seed=42, early_braking=False,
                arm_posture_pilot=False):
    """Evaluate every save and stop after the first block without a better policy."""
    state_file = work_dir / "progress.json"
    finishing_pilot = control is not None
    mode = "lateral" if lateral_pilot else "finishing" if finishing_pilot else "pilot"
    if arm_posture_pilot:
        mode = "arm_posture"
    friction = [os.environ.get("ZLAB_FOOT_FRICTION_MIN", "0.35"),
                os.environ.get("ZLAB_FOOT_FRICTION_MAX", "1.15")]
    if state_file.exists():
        state = json.loads(state_file.read_text())
        if state.get("mode") != mode:
            raise ValueError("Use a separate work directory for the stability pilot.")
        if state["foot_friction"] != friction:
            raise ValueError("Use a fresh pilot work directory when changing foot friction.")
        if state.get("seeds", list(PILOT_SEEDS)) != list(seeds):
            raise ValueError("Use a fresh pilot work directory when changing evaluation seeds.")
        if state.get("train_seed", 42) != train_seed:
            raise ValueError("Use a fresh pilot work directory when changing the training seed.")
        if state.get("early_braking", False) != early_braking:
            raise ValueError("Use a fresh pilot work directory when changing the braking controller.")
    else:
        baseline = work_dir / "baseline.pt"
        shutil.copyfile(checkpoint, baseline)
        state = {"mode": mode,
                 "checkpoint": str(baseline), "reports": None,
                 "block": 0, "stop_reason": None, "foot_friction": friction,
                 "seeds": list(seeds), "train_seed": train_seed, "early_braking": early_braking}
        save_state(state_file, state)

    def evaluate_both(path, directory, name):
        return [
            evaluate(path, directory, f"{name}-seed-{seed}", PROFILES[0],
                     seed=seed, lateral_correction=lateral_pilot, early_braking=early_braking)
            for seed in seeds
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
            state["control_reports"] = evaluate_both(control, directory, "control")
            save_state(state_file, state)
        runtime_env = dict(
            os.environ, ZLAB_PYTHON=sys.executable, ZLAB_LOG_ROOT=str(directory),
            ZLAB_TRAIN_SEED=str(train_seed + block - 1), ZLAB_TORSO_PUSH_SCALE="0",
            ZLAB_OBSERVATION_NOISE_SCALE="0", ZLAB_SAVE_INTERVAL="100",
            ZLAB_LATERAL_CORRECTION=str(lateral_pilot),
            ZLAB_EARLY_BRAKING=str(early_braking),
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
            paired = zip(state["reports"], reports, strict=True)
            improved = all(arm_posture_improved(new, old) for old, new in paired) if arm_posture_pilot else all(
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
                for old, new in paired
            )
            comparisons.append({"checkpoint": str(candidate), "reports": reports,
                                "improved_on_both_seeds": improved})
            if improved:
                qualified.append((candidate, reports))
            print(f"[BALL STABILITY] {candidate.name}: "
                  f"{'improved on both seeds' if improved else 'missed the improvement criteria'}",
                  flush=True)

        save_state(directory / "comparison.json", {
            "seeds": seeds, "baseline": str(incumbent), "early_braking": early_braking,
            "baseline_reports": state["reports"], "candidates": comparisons,
            **({"control_checkpoint": str(control), "control_reports": state["control_reports"]}
               if finishing_pilot else {}),
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
            sum(report["stops_with_relaxed_arms"] for report in entry[1]) if arm_posture_pilot else 0,
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
    parser.add_argument("--pilot-seeds", nargs=2, type=int, default=PILOT_SEEDS,
                        metavar=("FIRST", "SECOND"))
    parser.add_argument("--pilot-train-seed", type=int, default=42)
    parser.add_argument("--braking-then-arms-reference", type=Path, metavar="CHECKPOINT")
    parser.add_argument("--recovery-seed", type=int, default=428027)
    parser.add_argument("--lateral-pilot", action="store_true")
    parser.add_argument("--arm-posture-pilot", action="store_true",
                        help="Require more arms-down stops on each seed while preserving stability.")
    parser.add_argument("--early-braking", action="store_true",
                        help="Use earlier braking and train lateral stability during the stop.")
    parser.add_argument("--finishing-control", type=Path, metavar="CHECKPOINT",
                        help="Run the finishing curriculum and evaluate this reference checkpoint.")
    parser.add_argument("--validate-against", type=Path, metavar="CHECKPOINT",
                        help="Compare the candidate against this control on fresh seeds without PPO.")
    parser.add_argument("--validation-alternative", type=Path, metavar="CHECKPOINT",
                        help="Evaluate an additional candidate with the five-second hold protocol.")
    args = parser.parse_args()
    if not 0 < args.hours < float("inf") or min(args.updates, args.num_envs) <= 0:
        parser.error("Use a finite, positive time budget and positive update and environment counts.")
    if args.pilot_blocks is not None and args.pilot_blocks <= 0:
        parser.error("--pilot-blocks must be positive.")
    if min(*args.pilot_seeds, args.pilot_train_seed) < 0 or len(set(args.pilot_seeds)) != 2:
        parser.error("Use distinct nonnegative pilot evaluation seeds and a nonnegative training seed.")
    if args.finishing_control is not None and args.pilot_blocks is None:
        parser.error("--finishing-control requires --pilot-blocks.")
    if args.lateral_pilot and args.finishing_control is None:
        parser.error("--lateral-pilot requires --finishing-control.")
    if args.arm_posture_pilot and (args.pilot_blocks is None or not args.lateral_pilot):
        parser.error("--arm-posture-pilot requires --pilot-blocks and --lateral-pilot.")
    if args.early_braking and args.pilot_blocks is None:
        parser.error("--early-braking requires --pilot-blocks.")
    if args.validate_against is not None and args.pilot_blocks is not None:
        parser.error("--validate-against cannot be used with --pilot-blocks.")
    if args.validation_alternative is not None and args.validate_against is None:
        parser.error("--validation-alternative requires --validate-against.")
    if args.braking_then_arms_reference is not None:
        if args.pilot_blocks is not None or args.validate_against is not None:
            parser.error("Choose one pilot or validation mode.")
        if args.recovery_seed < 0 or args.recovery_seed in args.pilot_seeds:
            parser.error("Use a separate nonnegative recovery seed.")
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    with (work_dir / "training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"Training is already running in {work_dir}.")
        checkpoint = args.checkpoint.expanduser().resolve(strict=True)
        if args.braking_then_arms_reference is not None:
            reference = args.braking_then_arms_reference.expanduser().resolve(strict=True)
            train_braking_then_arms(checkpoint, reference, work_dir, args.updates, args.num_envs,
                                    seeds=(*args.pilot_seeds, args.recovery_seed),
                                    train_seed=args.pilot_train_seed)
        elif args.validate_against is not None:
            control = args.validate_against.expanduser().resolve(strict=True)
            alternative = args.validation_alternative
            if alternative is not None:
                alternative = alternative.expanduser().resolve(strict=True)
            validate_candidate(checkpoint, control, work_dir, alternative)
        elif args.pilot_blocks is None:
            train_blocks(checkpoint, work_dir, args.hours, args.updates, args.num_envs)
        else:
            control = args.finishing_control
            if control is not None:
                control = control.expanduser().resolve(strict=True)
            train_pilot(checkpoint, work_dir, args.pilot_blocks, args.updates, args.num_envs,
                        control, lateral_pilot=args.lateral_pilot, seeds=args.pilot_seeds,
                        train_seed=args.pilot_train_seed, early_braking=args.early_braking,
                        arm_posture_pilot=args.arm_posture_pilot)


if __name__ == "__main__":
    main()
