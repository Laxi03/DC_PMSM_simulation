"""Select original motor/task environments without importing the learning stack."""
from dataclasses import dataclass
from importlib import import_module

PERTURBATION_TARGETS = ("R", "Ld", "Lq", "lam", "Vdc")
MODES = ("locked", "free", "sinus_speed")


@dataclass(frozen=True)
class Experiment:
    motor: str
    task: str
    seed: int = 123
    control_mode: str | None = None
    sinus_amplitude: float = 300.0
    perturbation: str | None = None
    relative_change: float = 0.3
    perturbation_time: float = 2.0

    @property
    def training_steps(self) -> int:
        return 2_000_000 if (self.motor, self.task) == ("pmsm", "speed") else 1_000_000

    @property
    def noise_std(self) -> float:
        return 0.08 if (self.motor, self.task) == ("dc", "current") else 0.1


def environment_factory(case: Experiment, *, training: bool, rank: int = 0):
    """Return a spawn-safe factory; evaluation never enables training randomization."""
    module = import_module(f".{case.motor}.{case.task}", package=__package__)
    if case.motor == "dc":
        return module.DCMotorCurrentEnv if case.task == "current" else module.DCMotorEnv

    options = {}
    if case.task == "current":
        options.update(control_mode=case.control_mode or "locked", sinus_w_amp=case.sinus_amplitude)
    if training:
        return module.make_env(base_seed=case.seed, rank=rank, **options)

    cls = module.PMSMCurrentEnv if case.task == "current" else module.PMSMSpeedEnv
    perturbation = module.EvalPerturbation(
        enabled=case.perturbation is not None,
        target=case.perturbation or "R",
        rel_change=case.relative_change,
        random_time=False,
        t_start=case.perturbation_time,
    )
    from functools import partial

    return partial(cls, training_mode=False, eval_perturbation=perturbation, seed=case.seed, **options)
