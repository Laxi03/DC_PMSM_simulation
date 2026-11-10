"""PMSM current-tracking environment from the TD3 study."""
import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class PMSM:
    R_pp: float = 10.8
    Ld_pp: float = 1.61e-3
    Lq_pp: float = 1.61e-3
    kT: float = 22.4e-3
    J: float = 2e-5
    B: float = 5.5e-6
    pole_pairs: int = 4
    Vdc: float = 24.0

    def derived(self):
        R = self.R_pp / 2.0
        Ld = self.Ld_pp / 2.0
        Lq = self.Lq_pp / 2.0
        lam = self.kT / (1.5 * self.pole_pairs)
        return R, Ld, Lq, lam


@dataclass
class State:
    id: float
    iq: float
    w: float
    theta: float


# ==========================================================
# REFERENCE
# ==========================================================

def generate_iq_profile(rng: np.random.Generator):
    n = rng.integers(8, 12)
    iq_vals = rng.uniform(-1.0, 1.0, n)
    durations = rng.uniform(0.4, 1.0, n)
    times = np.cumsum(durations)
    return iq_vals, times


def iq_ref_func(t, vals, times):
    for i in range(len(times)):
        if t < times[i]:
            return vals[i]
    return vals[-1]


# ==========================================================
# TRAIN-TIME RANDOMIZATION / EVAL PERTURBATION
# ==========================================================

@dataclass
class TrainRandomization:
    """
    Training-time uncertainty.

    By default only R is randomized, but Ld / Lq can also be enabled.
    All jumps are defined relative to nominal values.
    """
    enabled: bool = True

    randomize_R: bool = True
    R_range: tuple = (0.85, 1.15)

    randomize_Ld: bool = False
    Ld_range: tuple = (0.90, 1.10)

    randomize_Lq: bool = False
    Lq_range: tuple = (0.90, 1.10)

    jump_enabled: bool = True
    jump_prob: float = 0.7
    jump_time_min: float = 1.0
    jump_time_max: float = 4.0

    jump_target: str = "R"     # R | Ld | Lq | lam | Vdc
    jump_range: tuple = (0.8, 1.35)


@dataclass
class EvalPerturbation:
    """
    Controlled evaluation-time perturbation.

    target in {R, Ld, Lq, lam, Vdc}
    """
    enabled: bool = False
    target: str = "R"          # R | Ld | Lq | lam | Vdc
    rel_change: float = 0.3
    random_time: bool = True
    t_start: float = 2.0
    t_min: float = 1.0
    t_max: float = 4.5


# ==========================================================
# MODEL
# ==========================================================

def electrical_speed(motor: PMSM, w_mech: float) -> float:
    return motor.pole_pairs * w_mech


def sinus_speed_profile_one_period(t: float, t_end: float, w_amp: float):
    """
    One full sinusoidal period over [0, t_end].

    w(t) = w_amp * sin(2*pi*t/t_end)
    theta_e is integrated analytically through theta_mech.
    """
    if t_end <= 0:
        return 0.0, 0.0

    omega = 2.0 * math.pi / t_end
    w = w_amp * math.sin(omega * t)

    # theta_mech(t) = integral w dt = w_amp/omega * (1 - cos(omega t))
    theta_mech = (w_amp / omega) * (1.0 - math.cos(omega * t))
    return w, theta_mech


def derivatives(motor, R, Ld, Lq, lam, x, vd, vq, lock_rotor):
    p = motor.pole_pairs
    we = p * x.w

    did = (vd - R * x.id + we * Lq * x.iq) / Ld
    diq = (vq - R * x.iq - we * (Ld * x.id + lam)) / Lq

    # salient PMSM torque
    Te = 1.5 * p * (lam * x.iq + (Ld - Lq) * x.id * x.iq)

    if lock_rotor:
        dw = 0.0
        dth = 0.0
    else:
        dw = (Te - motor.B * x.w) / motor.J
        dth = we

    return State(did, diq, dw, dth)


def rk4(motor, R, Ld, Lq, lam, x, dt, vd, vq, lock_rotor):
    def f(s):
        return derivatives(motor, R, Ld, Lq, lam, s, vd, vq, lock_rotor)

    k1 = f(x)
    k2 = f(State(
        x.id + dt * k1.id / 2.0,
        x.iq + dt * k1.iq / 2.0,
        x.w + dt * k1.w / 2.0,
        x.theta + dt * k1.theta / 2.0
    ))
    k3 = f(State(
        x.id + dt * k2.id / 2.0,
        x.iq + dt * k2.iq / 2.0,
        x.w + dt * k2.w / 2.0,
        x.theta + dt * k2.theta / 2.0
    ))
    k4 = f(State(
        x.id + dt * k3.id,
        x.iq + dt * k3.iq,
        x.w + dt * k3.w,
        x.theta + dt * k3.theta
    ))

    return State(
        x.id + dt * (k1.id + 2.0 * k2.id + 2.0 * k3.id + k4.id) / 6.0,
        x.iq + dt * (k1.iq + 2.0 * k2.iq + 2.0 * k3.iq + k4.iq) / 6.0,
        x.w + dt * (k1.w + 2.0 * k2.w + 2.0 * k3.w + k4.w) / 6.0,
        x.theta + dt * (k1.theta + 2.0 * k2.theta + 2.0 * k3.theta + k4.theta) / 6.0
    )


# ==========================================================
# ENVIRONMENT
# ==========================================================

class PMSMCurrentEnv(gym.Env):
    """
    RL environment for dq current control.

    Modes:
    - locked: rotor locked, w = 0
    - free: free mechanical response
    - sinus_speed: prescribed one-period sinusoidal speed profile during current control

    RL directly commands vd and vq.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        control_mode="locked",          # locked | free | sinus_speed
        training_mode=False,
        train_randomization=None,
        eval_perturbation=None,
        seed=None,
        sinus_w_amp=300.0,
    ):
        super().__init__()

        self.control_mode = control_mode
        self.training_mode = training_mode

        self.train_randomization = (
            train_randomization if train_randomization is not None else TrainRandomization()
        )
        self.eval_perturbation = (
            eval_perturbation if eval_perturbation is not None else EvalPerturbation()
        )

        self.dt = 5e-5
        self.frame_skip = 20
        self.episode_length = 6.0

        self.motor = PMSM()

        # Nominal/reference parameter values
        self.R_nom, self.Ld_nom, self.Lq_nom, self.lam_nom = self.motor.derived()
        self.Vdc_nom = self.motor.Vdc

        # Actual runtime plant parameter values
        self.R = self.R_nom
        self.Ld = self.Ld_nom
        self.Lq = self.Lq_nom
        self.lam = self.lam_nom
        self.Vdc = self.Vdc_nom

        self.state = State(0.0, 0.0, 0.0, 0.0)

        self.Imax = 2.0
        self.sinus_w_amp = sinus_w_amp

        self.vd = 0.0
        self.vq = 0.0
        self.prev_vd = 0.0
        self.prev_vq = 0.0

        self.t = 0.0
        self.int_eq = 0.0

        self.profile_vals = None
        self.profile_times = None
        self.last_iq_ref = 0.0

        # eval perturbation
        self.perturb_time = None
        self.perturb_applied = False

        # training jump
        self.train_jump_enabled_this_ep = False
        self.train_jump_time = None
        self.train_jump_factor = 1.0
        self.train_jump_applied = False
        self.train_jump_target = "R"

        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

        # [iq, id, iq_ref, e_q, e_d, int_eq, vd, vq, w]
        self.observation_space = spaces.Box(-1.0, 1.0, (9,), dtype=np.float32)

    @property
    def Vmax(self):
        return self.Vdc / math.sqrt(3.0)

    def _apply_param_relative_to_nominal(self, target: str, factor: float):
        if target == "R":
            self.R = self.R_nom * factor
        elif target == "Ld":
            self.Ld = self.Ld_nom * factor
        elif target == "Lq":
            self.Lq = self.Lq_nom * factor
        elif target == "lam":
            self.lam = self.lam_nom * factor
        elif target == "Vdc":
            self.Vdc = self.Vdc_nom * factor

    def _setup_training_randomization(self):
        self.train_jump_enabled_this_ep = False
        self.train_jump_time = None
        self.train_jump_factor = 1.0
        self.train_jump_applied = False
        self.train_jump_target = self.train_randomization.jump_target

        if not self.training_mode:
            self.R = self.R_nom
            self.Ld = self.Ld_nom
            self.Lq = self.Lq_nom
            return

        if not self.train_randomization.enabled:
            self.R = self.R_nom
            self.Ld = self.Ld_nom
            self.Lq = self.Lq_nom
            return

        if self.train_randomization.randomize_R:
            lo, hi = self.train_randomization.R_range
            self.R = self.R_nom * self.rng.uniform(lo, hi)

        if self.train_randomization.randomize_Ld:
            lo, hi = self.train_randomization.Ld_range
            self.Ld = self.Ld_nom * self.rng.uniform(lo, hi)

        if self.train_randomization.randomize_Lq:
            lo, hi = self.train_randomization.Lq_range
            self.Lq = self.Lq_nom * self.rng.uniform(lo, hi)

        if self.train_randomization.jump_enabled and (self.rng.random() < self.train_randomization.jump_prob):
            self.train_jump_enabled_this_ep = True
            self.train_jump_time = self.rng.uniform(
                self.train_randomization.jump_time_min,
                self.train_randomization.jump_time_max
            )
            lo, hi = self.train_randomization.jump_range
            self.train_jump_factor = self.rng.uniform(lo, hi)

    def _apply_training_jump_if_needed(self):
        if not self.training_mode:
            return
        if not self.train_jump_enabled_this_ep:
            return
        if self.train_jump_applied:
            return
        if self.t < self.train_jump_time:
            return

        self._apply_param_relative_to_nominal(self.train_jump_target, self.train_jump_factor)
        self.train_jump_applied = True

    def _apply_eval_perturbation_if_needed(self):
        if self.training_mode:
            return
        if not self.eval_perturbation.enabled:
            return
        if self.perturb_applied:
            return
        if self.t < self.perturb_time:
            return

        factor = 1.0 + self.eval_perturbation.rel_change
        self._apply_param_relative_to_nominal(self.eval_perturbation.target, factor)
        self.perturb_applied = True

    def _enforce_current_control_mode_speed(self):
        """
        Apply the chosen current-control mechanical mode.
        """
        if self.control_mode == "locked":
            self.state.w = 0.0
            self.state.theta = 0.0
        elif self.control_mode == "sinus_speed":
            w, theta_mech = sinus_speed_profile_one_period(
                self.t,
                self.episode_length,
                self.sinus_w_amp
            )
            self.state.w = w
            self.state.theta = electrical_speed(self.motor, theta_mech)
        # free mode: do nothing, mechanical dynamics come from model

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.state = State(0.0, 0.0, 0.0, 0.0)

        self.R = self.R_nom
        self.Ld = self.Ld_nom
        self.Lq = self.Lq_nom
        self.lam = self.lam_nom
        self.Vdc = self.Vdc_nom

        self._setup_training_randomization()
        self._enforce_current_control_mode_speed()

        self.vd = 0.0
        self.vq = 0.0
        self.prev_vd = 0.0
        self.prev_vq = 0.0

        self.t = 0.0
        self.int_eq = 0.0

        self.profile_vals, self.profile_times = generate_iq_profile(self.rng)
        self.last_iq_ref = iq_ref_func(0.0, self.profile_vals, self.profile_times)

        self.perturb_applied = False
        if (not self.training_mode) and self.eval_perturbation.enabled:
            if self.eval_perturbation.random_time:
                self.perturb_time = self.rng.uniform(
                    self.eval_perturbation.t_min,
                    self.eval_perturbation.t_max
                )
            else:
                self.perturb_time = self.eval_perturbation.t_start
        else:
            self.perturb_time = None

        return self._obs(), {}

    def _obs(self):
        iq_ref = iq_ref_func(self.t, self.profile_vals, self.profile_times)

        e_q = (iq_ref - self.state.iq) / self.Imax
        e_d = (0.0 - self.state.id) / self.Imax

        return np.array([
            self.state.iq / self.Imax,
            self.state.id / self.Imax,
            iq_ref / self.Imax,
            e_q,
            e_d,
            np.clip(self.int_eq, -1.0, 1.0),
            self.vd / max(self.Vmax, 1e-12),
            self.vq / max(self.Vmax, 1e-12),
            np.clip(self.state.w / 500.0, -1.0, 1.0),
        ], dtype=np.float32)

    def step(self, action):
        reward = 0.0
        last_terms = None

        action = np.asarray(action, dtype=np.float64).reshape(-1)

        for _ in range(self.frame_skip):
            self._apply_training_jump_if_needed()
            self._apply_eval_perturbation_if_needed()

            vd_cmd = float(np.clip(action[0], -1.0, 1.0)) * self.Vmax
            vq_cmd = float(np.clip(action[1], -1.0, 1.0)) * self.Vmax

            alpha = self.dt / 5e-4
            alpha = float(np.clip(alpha, 0.0, 1.0))

            self.vd += alpha * (vd_cmd - self.vd)
            self.vq += alpha * (vq_cmd - self.vq)

            self.vd = float(np.clip(self.vd, -self.Vmax, self.Vmax))
            self.vq = float(np.clip(self.vq, -self.Vmax, self.Vmax))

            if self.control_mode == "free":
                self.state = rk4(
                    self.motor,
                    self.R,
                    self.Ld,
                    self.Lq,
                    self.lam,
                    self.state,
                    self.dt,
                    self.vd,
                    self.vq,
                    lock_rotor=False
                )
            else:
                # integrate electrical state with mechanical part frozen inside this step
                # then overwrite speed profile / locked condition
                frozen = rk4(
                    self.motor,
                    self.R,
                    self.Ld,
                    self.Lq,
                    self.lam,
                    self.state,
                    self.dt,
                    self.vd,
                    self.vq,
                    lock_rotor=True
                )
                self.state.id = frozen.id
                self.state.iq = frozen.iq
                self.t += self.dt
                self._enforce_current_control_mode_speed()
                # theta already handled by mode
                iq_ref = iq_ref_func(self.t, self.profile_vals, self.profile_times)
                self.last_iq_ref = iq_ref

                e_q = (iq_ref - self.state.iq) / self.Imax
                e_d = (0.0 - self.state.id) / self.Imax

                self.int_eq += e_q * self.dt
                self.int_eq = float(np.clip(self.int_eq, -1.0, 1.0))

                tracking_q = -18.0 * abs(e_q)
                tracking_d = -6.0 * abs(e_d)
                integral_pen = -0.12 * (self.int_eq ** 2)

                du_d = (self.vd - self.prev_vd) / max(self.Vmax, 1e-12)
                du_q = (self.vq - self.prev_vq) / max(self.Vmax, 1e-12)
                smooth_pen = -0.01 * (du_d ** 2 + du_q ** 2)

                id_pen = -0.02 * (self.state.id / self.Imax) ** 2
                tight_bonus = 0.08 if (abs(e_q) < 0.02 and abs(e_d) < 0.02) else 0.0

                reward_step_raw = (
                    tracking_q
                    + tracking_d
                    + integral_pen
                    + smooth_pen
                    + id_pen
                    + tight_bonus
                )
                reward_step = float(np.clip(reward_step_raw, -1.0, 0.5))
                reward += reward_step

                self.prev_vd = self.vd
                self.prev_vq = self.vq

                last_terms = {
                    "reward_step": reward_step,
                    "tracking_q": tracking_q,
                    "tracking_d": tracking_d,
                    "integral_pen": integral_pen,
                    "smooth_pen": smooth_pen,
                    "id_pen": id_pen,
                    "tight_bonus": tight_bonus,
                    "error_q": e_q,
                    "error_d": e_d,
                    "R": self.R,
                    "Ld": self.Ld,
                    "Lq": self.Lq,
                    "lam": self.lam,
                    "Vdc": self.Vdc,
                    "perturb_time": self.perturb_time if self.perturb_time is not None else -1.0,
                    "perturb_applied": self.perturb_applied,
                    "train_jump_time": self.train_jump_time if self.train_jump_time is not None else -1.0,
                    "train_jump_applied": self.train_jump_applied,
                }
                continue

            # free mode path
            self.t += self.dt

            iq_ref = iq_ref_func(self.t, self.profile_vals, self.profile_times)
            self.last_iq_ref = iq_ref

            e_q = (iq_ref - self.state.iq) / self.Imax
            e_d = (0.0 - self.state.id) / self.Imax

            self.int_eq += e_q * self.dt
            self.int_eq = float(np.clip(self.int_eq, -1.0, 1.0))

            tracking_q = -18.0 * abs(e_q)
            tracking_d = -6.0 * abs(e_d)
            integral_pen = -0.12 * (self.int_eq ** 2)

            du_d = (self.vd - self.prev_vd) / max(self.Vmax, 1e-12)
            du_q = (self.vq - self.prev_vq) / max(self.Vmax, 1e-12)
            smooth_pen = -0.01 * (du_d ** 2 + du_q ** 2)

            id_pen = -0.02 * (self.state.id / self.Imax) ** 2
            tight_bonus = 0.08 if (abs(e_q) < 0.02 and abs(e_d) < 0.02) else 0.0

            reward_step_raw = (
                tracking_q
                + tracking_d
                + integral_pen
                + smooth_pen
                + id_pen
                + tight_bonus
            )
            reward_step = float(np.clip(reward_step_raw, -1.0, 0.5))
            reward += reward_step

            self.prev_vd = self.vd
            self.prev_vq = self.vq

            last_terms = {
                "reward_step": reward_step,
                "tracking_q": tracking_q,
                "tracking_d": tracking_d,
                "integral_pen": integral_pen,
                "smooth_pen": smooth_pen,
                "id_pen": id_pen,
                "tight_bonus": tight_bonus,
                "error_q": e_q,
                "error_d": e_d,
                "R": self.R,
                "Ld": self.Ld,
                "Lq": self.Lq,
                "lam": self.lam,
                "Vdc": self.Vdc,
                "perturb_time": self.perturb_time if self.perturb_time is not None else -1.0,
                "perturb_applied": self.perturb_applied,
                "train_jump_time": self.train_jump_time if self.train_jump_time is not None else -1.0,
                "train_jump_applied": self.train_jump_applied,
            }

        reward /= self.frame_skip

        terminated = False
        truncated = self.t > self.episode_length

        return self._obs(), reward, terminated, truncated, last_terms


def make_env(control_mode="locked", base_seed=None, rank=0, sinus_w_amp=300.0):
    def _init():
        env_seed = None if base_seed is None else int(base_seed + rank)
        return PMSMCurrentEnv(
            control_mode=control_mode,
            training_mode=True,
            train_randomization=TrainRandomization(
                enabled=True,
                randomize_R=True,
                R_range=(0.85, 1.15),
                randomize_Ld=True,
                Ld_range=(0.9, 1.1),
                randomize_Lq=True,
                Lq_range=(0.9, 1.1),
                jump_enabled=True,
                jump_prob=0.7,
                jump_time_min=1.0,
                jump_time_max=4.0,
                jump_target="R",
                jump_range=(0.8, 1.35),
            ),
            eval_perturbation=EvalPerturbation(enabled=False),
            seed=env_seed,
            sinus_w_amp=sinus_w_amp,
        )
    return _init
