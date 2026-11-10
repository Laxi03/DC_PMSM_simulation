"""One CLI for baseline simulation, TD3 training and checkpoint evaluation."""
import argparse
import math
from pathlib import Path
import re

from .experiments import Experiment, MODES, PERTURBATION_TARGETS


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--motor", choices=("dc", "pmsm"), required=True)
    common.add_argument("--task", choices=("current", "speed"), default="speed")
    common.add_argument("--seed", type=int, default=123)
    common.add_argument("--workspace", type=Path, default=Path.cwd())
    common.add_argument("--control-mode", choices=MODES, help="Current-task rotor mode (default: locked)")
    common.add_argument("--sinus-amplitude", type=float, default=300.0, help="PMSM imposed speed amplitude in rad/s")

    plots = argparse.ArgumentParser(add_help=False)
    plots.add_argument("--duration", type=float, help="Simulation seconds; caps a learned rollout for preview")
    plots.add_argument("--show", action="store_true", help="Open the tracking figure as well as saving it")
    plots.add_argument("--perturbation", choices=PERTURBATION_TARGETS, help="PMSM parameter to change")
    plots.add_argument("--relative-change", type=float, default=0.3)
    plots.add_argument("--perturbation-time", type=float, default=2.0)

    baseline = commands.add_parser("baseline", parents=[common, plots], help="Run PI or FOC without checkpoints")
    reference = baseline.add_mutually_exclusive_group()
    reference.add_argument("--reference", type=float, help="Fixed setpoint in A or rad/s")
    reference.add_argument("--random-reference", action="store_true", help="Use the baseline's seeded step profile")

    training = commands.add_parser("train", parents=[common], help="Train a new TD3 model")
    training.add_argument("--run", required=True, help="A new run name")
    training.add_argument("--steps", type=int, help="Environment transitions (defaults: 1M, PMSM speed 2M)")
    training.add_argument("--workers", type=int, default=1)
    training.add_argument("--device", default="cpu")

    evaluation = commands.add_parser("eval", parents=[common, plots], help="Evaluate a trusted TD3 checkpoint")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    return parser


def validate(args):
    if not 0 <= args.seed < 2**32:
        raise ValueError("Seed must be in [0, 2^32)")
    if args.control_mode and args.task != "current":
        raise ValueError("--control-mode is only supported for current tracking")
    if args.motor == "dc" and args.control_mode == "sinus_speed":
        raise ValueError("Imposed sinusoidal speed is only supported for PMSM")
    if args.motor == "dc" and args.command != "baseline" and args.control_mode == "free":
        raise ValueError("The DC TD3 current environment uses a locked rotor")
    if not math.isfinite(args.sinus_amplitude) or args.sinus_amplitude < 0:
        raise ValueError("Sinusoidal speed amplitude must be finite and nonnegative")

    if args.command == "train":
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.run):
            raise ValueError("Run name must start with a letter or digit and contain only letters, digits, '.', '_' or '-'")
        if not 1 <= args.workers <= 32 or args.seed + args.workers > 2**32:
            raise ValueError("Use 1–32 workers and a seed that leaves room for per-worker seeds")
        if args.steps is not None and args.steps < 1:
            raise ValueError("Training steps must be positive")
        return

    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        raise ValueError("Duration must be finite and positive")
    if args.perturbation and args.motor != "pmsm":
        raise ValueError("Runtime parameter perturbations are only supported for PMSM")
    if not math.isfinite(args.relative_change) or args.relative_change <= -1:
        raise ValueError("Relative change must be finite and greater than -1")
    if not math.isfinite(args.perturbation_time) or args.perturbation_time < 0:
        raise ValueError("Perturbation time must be finite and nonnegative")
    if args.command == "baseline" and args.reference is not None and not math.isfinite(args.reference):
        raise ValueError("Reference must be finite")


def run(args):
    workspace = args.workspace.expanduser().resolve()
    case = Experiment(
        motor=args.motor, task=args.task, seed=args.seed, control_mode=args.control_mode,
        sinus_amplitude=args.sinus_amplitude, perturbation=getattr(args, "perturbation", None),
        relative_change=getattr(args, "relative_change", 0.3),
        perturbation_time=getattr(args, "perturbation_time", 2.0),
    )
    if args.command == "train":
        from .training import train

        output = workspace / "models" / case.motor / case.task / args.run
        steps = case.training_steps if args.steps is None else args.steps
        checkpoint = train(case, output, steps=steps, workers=args.workers, device=args.device)
        print(f"Saved: {checkpoint}")
        return

    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    from .evaluation import baseline, evaluate, save_report

    if args.command == "baseline":
        trace = baseline(case, duration=args.duration, reference=args.reference, random=args.random_reference)
        controller = "pi" if case.motor == "dc" else "foc"
        settings = {"duration": args.duration, "reference": args.reference, "random_reference": args.random_reference}
    else:
        checkpoint = args.checkpoint.expanduser().resolve()
        trace = evaluate(case, checkpoint, duration=args.duration)
        controller = "td3"
        settings = {"checkpoint": str(checkpoint), "duration_cap": args.duration}
    save_report(trace, case, controller, workspace, show=args.show, settings=settings)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate(args)
        run(args)
    except ModuleNotFoundError as exc:
        if exc.name != "stable_baselines3":
            raise
        parser.error("TD3 commands require the optional learning stack: python -m pip install '.[rl]'")
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))
