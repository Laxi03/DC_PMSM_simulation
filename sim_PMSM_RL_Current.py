#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import argparse
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

plt.style.use("seaborn-v0_8-darkgrid")


# ==========================================================
# MOTOR
# ==========================================================

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


# ==========================================================
# CALLBACK
# ==========================================================

class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ep_rewards = []
        self.current = None

    def _on_training_start(self):
        self.current = np.zeros(self.training_env.num_envs)

    def _on_step(self):
        r = self.locals["rewards"]
        d = self.locals["dones"]

        self.current += r

        for i, done in enumerate(d):
            if done:
                self.ep_rewards.append(self.current[i])
                self.current[i] = 0.0

        return True

    def _on_training_end(self):
        r = np.array(self.ep_rewards)

        plt.figure(figsize=(8, 5))
        plt.plot(r, alpha=0.3, label="Episode reward")

        if len(r) >= 20:
            smooth = np.convolve(r, np.ones(20) / 20, mode="valid")
            plt.plot(np.arange(19, len(r)), smooth, linewidth=3, label="Moving average")

        plt.title("Training Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()


# ==========================================================
# SEGMENT METRICS
# ==========================================================

def rise_time_segment(t_seg, y_seg, y0, yref):
    step = yref - y0
    amp = abs(step)

    if amp < 1e-12:
        return 0.0

    target = y0 + 0.9 * step

    if step > 0:
        idx = np.where(y_seg >= target)[0]
    else:
        idx = np.where(y_seg <= target)[0]

    if len(idx) == 0:
        return np.nan

    return t_seg[idx[0]] - t_seg[0]


def overshoot_percent_segment(y_seg, y0, yref):
    step = yref - y0
    amp = abs(step)

    if amp < 1e-12:
        return 0.0

    if step > 0:
        peak = np.max(y_seg)
        overshoot = max(0.0, peak - yref)
    else:
        valley = np.min(y_seg)
        overshoot = max(0.0, yref - valley)

    return 100.0 * overshoot / amp


def steady_state_error_segment(y_seg, yref):
    n_tail = max(5, int(0.1 * len(y_seg)))
    tail = y_seg[-n_tail:]
    return np.mean(yref - tail)


def segment_metrics(t, iq, iq_ref, vd, vq, Vmax):
    t = np.asarray(t)
    iq = np.asarray(iq)
    iq_ref = np.asarray(iq_ref)
    vd = np.asarray(vd)
    vq = np.asarray(vq)

    if len(t) < 2:
        return []

    dt = np.mean(np.diff(t))

    change_idx = [0]
    for k in range(1, len(iq_ref)):
        if abs(iq_ref[k] - iq_ref[k - 1]) > 1e-9:
            change_idx.append(k)
    change_idx.append(len(iq_ref))

    rows = []

    for s in range(len(change_idx) - 1):
        i0 = change_idx[s]
        i1 = change_idx[s + 1]

        if i1 - i0 < 5:
            continue

        t_seg = t[i0:i1]
        iq_seg = iq[i0:i1]
        iq_ref_seg = iq_ref[i0:i1]
        vd_seg = vd[i0:i1]
        vq_seg = vq[i0:i1]

        ref_now = iq_ref_seg[0]
        ref_prev = iq[i0 - 1] if i0 > 0 else iq_seg[0]

        e_seg = ref_now - iq_seg

        tr = rise_time_segment(t_seg, iq_seg, ref_prev, ref_now)
        mp = overshoot_percent_segment(iq_seg, ref_prev, ref_now)
        sse = steady_state_error_segment(iq_seg, ref_now)
        iae = np.trapezoid(np.abs(e_seg), t_seg)
        ise = np.trapezoid(e_seg ** 2, t_seg)

        u_mag = np.sqrt(vd_seg ** 2 + vq_seg ** 2)
        rms_u = np.sqrt(np.mean(u_mag ** 2))
        time_at_rail = np.sum(u_mag >= 0.98 * Vmax) * dt

        rows.append({
            "Segments": f"{t_seg[0]:.2f}-{t_seg[-1]:.2f}",
            "tr": tr,
            "Mp": mp,
            "SSE": sse,
            "IAE": iae,
            "ISE": ise,
            "RMSu": rms_u,
            "TimeRail": time_at_rail,
            "ref": ref_now
        })

    return rows


def print_segment_metrics_table(rows):
    if not rows:
        print("\nNincsenek kiírható szakaszmutatók.\n")
        return

    print("\nMetrics (PMSM RL current)\n")
    header = (
        f"{'Segments':>14} | "
        f"{'i_ref [A]':>9} | "
        f"{'t_r [s]':>9} | "
        f"{'M_p [%]':>9} | "
        f"{'SSE [A]':>10} | "
        f"{'IAE':>10} | "
        f"{'ISE':>10} | "
        f"{'RMS(u_dq) [V]':>14} | "
        f"{'Time@rail [s]':>14}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        tr_str = f"{r['tr']:.4f}" if np.isfinite(r["tr"]) else "nan"
        print(
            f"{r['Segments']:>14} | "
            f"{r['ref']:>9.3f} | "
            f"{tr_str:>9} | "
            f"{r['Mp']:>9.3f} | "
            f"{r['SSE']:>10.4f} | "
            f"{r['IAE']:>10.4f} | "
            f"{r['ISE']:>10.4f} | "
            f"{r['RMSu']:>14.4f} | "
            f"{r['TimeRail']:>14.4f}"
        )

    valid_tr = [r["tr"] for r in rows if np.isfinite(r["tr"])]

    avg_ref = np.mean([r["ref"] for r in rows])
    avg_tr = np.mean(valid_tr) if len(valid_tr) > 0 else np.nan
    avg_mp = np.mean([r["Mp"] for r in rows])
    avg_sse = np.mean([r["SSE"] for r in rows])
    avg_iae = np.mean([r["IAE"] for r in rows])
    avg_ise = np.mean([r["ISE"] for r in rows])
    avg_rmsu = np.mean([r["RMSu"] for r in rows])
    avg_timerail = np.mean([r["TimeRail"] for r in rows])

    print("-" * len(header))
    avg_tr_str = f"{avg_tr:.4f}" if np.isfinite(avg_tr) else "nan"
    print(
        f"{'Average':>14} | "
        f"{avg_ref:>9.3f} | "
        f"{avg_tr_str:>9} | "
        f"{avg_mp:>9.3f} | "
        f"{avg_sse:>10.4f} | "
        f"{avg_iae:>10.4f} | "
        f"{avg_ise:>10.4f} | "
        f"{avg_rmsu:>14.4f} | "
        f"{avg_timerail:>14.4f}"
    )
    print()


# ==========================================================
# TRAIN
# ==========================================================

MODEL_PATH = "RL_models/pmsm_rl_current"

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


def train(control_mode="locked", seed=None, sinus_w_amp=300.0):
    os.makedirs("RL_models", exist_ok=True)

    if seed is not None:
        np.random.seed(seed)

    env = SubprocVecEnv([
        make_env(control_mode=control_mode, base_seed=seed, rank=i, sinus_w_amp=sinus_w_amp)
        for i in range(8)
    ])

    noise = NormalActionNoise(
        mean=np.zeros(2),
        sigma=0.1 * np.ones(2)
    )

    model = TD3(
        "MlpPolicy",
        env,
        verbose=2,
        learning_rate=3e-4,
        buffer_size=200000,
        batch_size=128,
        gamma=0.99,
        tau=0.005,
        action_noise=noise,
        policy_kwargs=dict(net_arch=[128, 64]),
        seed=seed,
    )

    callback = RewardLogger()

    model.learn(1_000_000, callback=callback)
    model.save(MODEL_PATH)

    print(f"\nSaved model to: {MODEL_PATH}\n")


# ==========================================================
# EVAL
# ==========================================================

def evaluate(control_mode="locked", perturbation=None, seed=None, sinus_w_amp=300.0):
    env = PMSMCurrentEnv(
        control_mode=control_mode,
        training_mode=False,
        train_randomization=TrainRandomization(enabled=False),
        eval_perturbation=perturbation,
        seed=seed,
        sinus_w_amp=sinus_w_amp,
    )
    model = TD3.load(MODEL_PATH)

    obs, _ = env.reset(seed=seed)

    t = []
    iq = []
    iq_ref = []
    id_log = []
    vq_log = []
    vd_log = []
    w_log = []

    reward_log = []
    tracking_q_log = []
    tracking_d_log = []
    integral_log = []
    smooth_log = []
    id_pen_log = []
    bonus_log = []

    R_log = []
    Ld_log = []
    Lq_log = []
    lam_log = []
    Vdc_log = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, trunc, info = env.step(action)

        t.append(env.t)
        iq.append(env.state.iq)
        iq_ref.append(env.last_iq_ref)
        id_log.append(env.state.id)
        vq_log.append(env.vq)
        vd_log.append(env.vd)
        w_log.append(env.state.w)

        reward_log.append(info["reward_step"])
        tracking_q_log.append(info["tracking_q"])
        tracking_d_log.append(info["tracking_d"])
        integral_log.append(info["integral_pen"])
        smooth_log.append(info["smooth_pen"])
        id_pen_log.append(info["id_pen"])
        bonus_log.append(info["tight_bonus"])

        R_log.append(info["R"])
        Ld_log.append(info["Ld"])
        Lq_log.append(info["Lq"])
        lam_log.append(info["lam"])
        Vdc_log.append(info["Vdc"])

        if trunc:
            break

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 10,
        "lines.linewidth": 1.5
    })

    fig, ax = plt.subplots(5, 1, figsize=(9, 9.5), sharex=True)

    ax[0].plot(t, iq, label=r"$i_q$")
    ax[0].plot(t, iq_ref, "--", label=r"$i_{q,\mathrm{ref}}$")
    ax[0].plot(t, id_log, label=r"$i_d$")
    ax[0].axhline(0.0, linestyle=":", linewidth=1.0)
    ax[0].set_ylabel("Current [A]")
    ax[0].set_title("dq-axis Currents")
    ax[0].legend()

    ax[1].plot(t, w_log, label=r"$\omega$")
    ax[1].set_ylabel(r"$\omega$ [rad/s]")
    ax[1].set_title("Mechanical Speed")
    ax[1].legend()

    ax[2].plot(t, vq_log, label=r"$v_q$")
    ax[2].plot(t, vd_log, label=r"$v_d$")
    ax[2].set_ylabel("Voltage [V]")
    ax[2].set_title("Control Effort")
    ax[2].legend()

    ax[3].plot(t, np.array(R_log) / env.R_nom, label="R/R_nom")
    ax[3].plot(t, np.array(Ld_log) / env.Ld_nom, label="Ld/Ld_nom")
    ax[3].plot(t, np.array(Lq_log) / env.Lq_nom, label="Lq/Lq_nom")
    ax[3].plot(t, np.array(lam_log) / env.lam_nom, label="lam/lam_nom")
    ax[3].plot(t, np.array(Vdc_log) / env.Vdc_nom, label="Vdc/Vdc_nom")
    ax[3].set_ylabel("Normalized")
    ax[3].set_title("Runtime Parameters")
    ax[3].legend(ncol=3, fontsize=9)

    ax[4].plot(t, reward_log, label="reward")
    ax[4].set_ylabel("Reward")
    ax[4].set_xlabel("Time [s]")
    ax[4].set_title("Per-step Reward During Evaluation")
    ax[4].legend()

    for a in ax:
        a.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    if perturbation is not None and perturbation.enabled and env.perturb_time is not None:
        for a in ax:
            a.axvline(env.perturb_time, color="red", linestyle="--", alpha=0.8)

    plt.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots(1, 1, figsize=(9, 3.8))
    ax2.plot(t, tracking_q_log, label="tracking_q")
    ax2.plot(t, tracking_d_log, label="tracking_d")
    ax2.plot(t, integral_log, label="integral")
    ax2.plot(t, smooth_log, label="smooth")
    ax2.plot(t, id_pen_log, label="id_pen")
    ax2.plot(t, bonus_log, label="bonus")
    ax2.set_title("Reward Components During Evaluation")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Contribution")
    ax2.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    ax2.legend(ncol=3, fontsize=9)

    if perturbation is not None and perturbation.enabled and env.perturb_time is not None:
        ax2.axvline(env.perturb_time, color="red", linestyle="--", alpha=0.8)

    plt.tight_layout()
    plt.show()

    abs_err = np.abs(np.array(iq_ref) - np.array(iq))
    print(f"Mean abs iq error : {np.mean(abs_err):.4f} A")
    print(f"Max abs iq error  : {np.max(abs_err):.4f} A")
    print(f"Mean |id|         : {np.mean(np.abs(id_log)):.4f} A")
    u_mag = np.sqrt(np.array(vd_log) ** 2 + np.array(vq_log) ** 2)
    print(f"Mean |u_dq|       : {np.mean(u_mag):.4f} V")
    print(f"Mean step reward  : {np.mean(reward_log):.4f}")
    print(f"Final speed       : {w_log[-1]:.4f} rad/s")

    if perturbation is not None and perturbation.enabled:
        print(f"Perturbation target : {perturbation.target}")
        print(f"Relative change     : {perturbation.rel_change:+.3f}")
        print(f"Perturbation time   : {env.perturb_time:.4f} s")

    rows = segment_metrics(
        t=np.array(t),
        iq=np.array(iq),
        iq_ref=np.array(iq_ref),
        vd=np.array(vd_log),
        vq=np.array(vq_log),
        Vmax=max(Vdc_log) / math.sqrt(3.0)
    )
    print_segment_metrics_table(rows)


# ==========================================================
# CLI
# ==========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    # current-control mechanical mode
    parser.add_argument(
        "--control_mode",
        choices=["locked", "free", "sinus_speed"],
        default="locked"
    )
    parser.add_argument("--sinus_w_amp", type=float, default=300.0)

    # Evaluation-only perturbation options
    parser.add_argument("--perturb", action="store_true")
    parser.add_argument(
        "--perturb_target",
        type=str,
        default="R",
        choices=["R", "Ld", "Lq", "lam", "Vdc"]
    )
    parser.add_argument("--perturb_change", type=float, default=0.3)
    parser.add_argument("--perturb_random_time", action="store_true")
    parser.add_argument("--perturb_time", type=float, default=2.0)
    parser.add_argument("--perturb_tmin", type=float, default=1.0)
    parser.add_argument("--perturb_tmax", type=float, default=4.5)

    args = parser.parse_args()

    if args.train:
        train(
            control_mode=args.control_mode,
            seed=args.seed,
            sinus_w_amp=args.sinus_w_amp
        )

    if args.eval:
        perturbation = EvalPerturbation(
            enabled=args.perturb,
            target=args.perturb_target,
            rel_change=args.perturb_change,
            random_time=args.perturb_random_time,
            t_start=args.perturb_time,
            t_min=args.perturb_tmin,
            t_max=args.perturb_tmax
        )
        evaluate(
            control_mode=args.control_mode,
            perturbation=perturbation,
            seed=args.seed,
            sinus_w_amp=args.sinus_w_amp
        )


if __name__ == "__main__":
    main()