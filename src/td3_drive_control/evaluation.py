"""Deterministic rollouts and shared, unit-labelled reporting."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .experiments import Experiment, environment_factory


@dataclass(frozen=True)
class Trace:
    columns: dict[str, np.ndarray]
    ended_by: str
    episode_return: float | None = None


def baseline(case: Experiment, *, duration: float | None, reference: float | None, random: bool) -> Trace:
    """Use the original PI/FOC plant and gain defaults."""
    if case.motor == "dc":
        from .dc import pi as controller
    else:
        from .pmsm import foc as controller

    args = controller.build_parser().parse_args([])
    args.mode = case.task
    args.seed = case.seed
    args.random_ref = random
    if duration is not None:
        args.t_end = duration
    if args.t_end < args.dt:
        raise ValueError(f"Duration must be at least one physical step ({args.dt:g} s)")

    if case.motor == "dc":
        args.lock_rotor = case.task == "current" and (case.control_mode or "locked") == "locked"
        if reference is not None:
            setattr(args, "w_ref" if case.task == "speed" else "i_ref", reference)
        d = controller.simulate(args)
        columns = {
            "time_s": d["t"] + args.dt,
            "reference": d["omega_ref"] if case.task == "speed" else d["i_ref"],
            "measurement": d["omega"] if case.task == "speed" else d["i"],
            "speed_rad_s": d["omega"], "current_A": d["i"], "voltage_V": d["u"],
        }
    else:
        args.current_control_mode = case.control_mode or "locked"
        args.sinus_w_amp = case.sinus_amplitude
        args.perturb = case.perturbation is not None
        args.perturb_target = case.perturbation or "R"
        args.perturb_change = case.relative_change
        args.perturb_time = case.perturbation_time
        if reference is not None:
            setattr(args, "w_ref" if case.task == "speed" else "iq_ref", reference)
        d = controller.simulate(args)
        columns = {
            "time_s": d["t"] + args.dt,
            "reference": d["w_ref"] if case.task == "speed" else d["iq_ref"],
            "measurement": d["w"] if case.task == "speed" else d["iq"],
            "speed_rad_s": d["w"],
            "current_A": np.hypot(d["id"], d["iq"]),
            "voltage_V": np.hypot(d["vd"], d["vq"]),
            "id_A": d["id"], "iq_A": d["iq"], "vd_V": d["vd"], "vq_V": d["vq"],
            **{key: d[key] for key in ("R", "Ld", "Lq", "lam", "Vdc")},
        }
    return Trace(columns, "duration")


def _validate_checkpoint(case: Experiment, checkpoint: Path):
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    sidecar = checkpoint.parent / "manifest.json"
    if checkpoint.name == "model.zip" and sidecar.is_file():
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata.get("schema") == "td3-drive-control/training-v1":
            if metadata.get("status") != "complete":
                raise ValueError(f"Training run is not complete: {sidecar}")
            settings = metadata["experiment"]
            if (case.motor, case.task) != (settings["motor"], settings["task"]):
                raise ValueError("Selected motor/task does not match the checkpoint manifest")


def evaluate(case: Experiment, checkpoint: Path, *, duration: float | None = None) -> Trace:
    """Load a trusted Stable-Baselines3 checkpoint and run a seeded episode."""
    from stable_baselines3 import TD3

    _validate_checkpoint(case, checkpoint)
    env = environment_factory(case, training=False)()
    rows = []
    total_reward = 0.0
    keys = ["time_s", "reference", "measurement", "speed_rad_s", "current_A", "voltage_V"]
    if case.motor == "pmsm":
        keys += ["id_A", "iq_A", "vd_V", "vq_V", "R", "Ld", "Lq", "lam", "Vdc"]
    try:
        model = TD3.load(checkpoint, env=env, device="cpu")
        obs, _ = env.reset(seed=case.seed)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            state = env.state
            if case.motor == "dc":
                reference, measurement = ((env.last_wref, state.omega) if case.task == "speed"
                                          else (env.last_iref, state.i))
                rows.append((env.t, reference, measurement, state.omega, state.i, env.u_applied))
            else:
                reference, measurement = ((env.last_wref, state.w) if case.task == "speed"
                                          else (env.last_iq_ref, state.iq))
                rows.append((env.t, reference, measurement, state.w,
                             np.hypot(state.id, state.iq), np.hypot(env.vd, env.vq),
                             state.id, state.iq, env.vd, env.vq, env.R, env.Ld, env.Lq, env.lam, env.Vdc))
            if terminated or truncated:
                ended_by = "terminated" if terminated else "time_limit"
                break
            if duration is not None and env.t >= duration:
                ended_by = "preview"
                break
    finally:
        env.close()
    return Trace(dict(zip(keys, np.asarray(rows, dtype=float).T, strict=True)), ended_by, total_reward)


def metrics(trace: Trace, task: str) -> dict:
    columns = trace.columns
    error = columns["reference"] - columns["measurement"]
    if not error.size or not all(np.isfinite(values).all() for values in columns.values()):
        raise ValueError("Simulation did not produce a finite, nonempty trace")
    return {
        "tracking_unit": "rad/s" if task == "speed" else "A",
        "tracking_mae": float(np.mean(np.abs(error))),
        "tracking_rmse": float(np.sqrt(np.mean(error**2))),
        "tracking_peak_abs_error": float(np.max(np.abs(error))),
        "rms_current_A": float(np.sqrt(np.mean(columns["current_A"]**2))),
        "peak_abs_current_A": float(np.max(np.abs(columns["current_A"]))),
        "rms_voltage_V": float(np.sqrt(np.mean(columns["voltage_V"]**2))),
        "samples": int(error.size),
        "simulated_seconds": float(columns["time_s"][-1]),
        "ended_by": trace.ended_by,
        "episode_return": trace.episode_return,
    }


def save_report(trace: Trace, case: Experiment, controller: str, workspace: Path, *,
                show: bool = False, settings: dict | None = None) -> Path:
    import matplotlib.pyplot as plt

    summary = metrics(trace, case.task)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = workspace / "outputs" / case.motor / case.task / f"{controller}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    data = trace.columns
    np.savetxt(output / "timeseries.csv", np.column_stack(list(data.values())),
               delimiter=",", header=",".join(data), comments="")
    for name, content in (
        ("metrics", summary),
        ("manifest", {"experiment": asdict(case), "controller": controller, "settings": settings or {}}),
    ):
        (output / f"{name}.json").write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, layout="constrained")
    try:
        axes[0].plot(data["time_s"], data["reference"], "--", color="#333333", label="Reference")
        axes[0].plot(data["time_s"], data["measurement"], color="#0072B2", label="Measured")
        axes[0].set_ylabel(f"{case.task.title()} [{summary['tracking_unit']}]")
        axes[0].legend()
        axes[1].plot(data["time_s"], data["current_A"], color="#009E73")
        axes[1].set_ylabel("Current [A]")
        axes[2].plot(data["time_s"], data["voltage_V"], color="#D55E00")
        axes[2].set(ylabel="Voltage [V]", xlabel="Time [s]")
        for axis in axes:
            axis.grid(alpha=0.2)
        fig.suptitle(f"{case.motor.upper()} · {controller.upper()} · {case.task} tracking")
        fig.savefig(output / "tracking.png", dpi=160)
        if show:
            plt.show()
    finally:
        plt.close(fig)
    print(f"MAE: {summary['tracking_mae']:.4g} {summary['tracking_unit']} ({trace.ended_by})")
    print(f"Saved: {output}")
    return output
