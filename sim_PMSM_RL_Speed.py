#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import argparse
import matplotlib.pyplot as plt
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

plt.style.use("seaborn-v0_8-darkgrid")


# ==========================================================
# MOTOR PARAMETERS
# ==========================================================

@dataclass
class PMSM:
    R_pp: float = 10.8
    L_pp: float = 1.61e-3
    kT: float = 22.4e-3
    J: float = 2e-5
    B: float = 5.5e-6
    pole_pairs: int = 4
    Vdc: float = 24

    def derived(self):
        R = self.R_pp / 2
        L = self.L_pp / 2
        lam = self.kT / (1.5 * self.pole_pairs)
        return R, L, lam


# ==========================================================
# STATE
# ==========================================================

@dataclass
class State:
    id: float
    iq: float
    w: float
    theta: float


# ==========================================================
# RANDOM SPEED PROFILE
# ==========================================================

def generate_speed_profile():
    n_segments = np.random.randint(6, 12)

    speeds = np.random.uniform(-350, 350, n_segments)
    durations = np.random.uniform(0.4, 1.2, n_segments)

    for i in range(1, len(speeds)):
        speeds[i] = 0.6 * speeds[i - 1] + 0.4 * speeds[i]

    times = np.cumsum(durations)
    return speeds, times


def speed_ref(t, speeds, times):
    for i in range(len(times)):
        if t < times[i]:
            return speeds[i]
    return speeds[-1]


# ==========================================================
# MODEL
# ==========================================================

def derivatives(motor, R, L, lam, x, vd, vq, Tload):
    p = motor.pole_pairs
    we = p * x.w

    did = (vd - R * x.id + we * L * x.iq) / L
    diq = (vq - R * x.iq - we * (L * x.id + lam)) / L

    Te = 1.5 * p * lam * x.iq
    dw = (Te - Tload - motor.B * x.w) / motor.J
    dth = we

    return State(did, diq, dw, dth)


def rk4(motor, R, L, lam, x, dt, vd, vq, Tload):
    def f(s):
        return derivatives(motor, R, L, lam, s, vd, vq, Tload)

    k1 = f(x)
    k2 = f(State(
        x.id + dt * k1.id / 2,
        x.iq + dt * k1.iq / 2,
        x.w + dt * k1.w / 2,
        x.theta + dt * k1.theta / 2
    ))
    k3 = f(State(
        x.id + dt * k2.id / 2,
        x.iq + dt * k2.iq / 2,
        x.w + dt * k2.w / 2,
        x.theta + dt * k2.theta / 2
    ))
    k4 = f(State(
        x.id + dt * k3.id,
        x.iq + dt * k3.iq,
        x.w + dt * k3.w,
        x.theta + dt * k3.theta
    ))

    return State(
        x.id + dt * (k1.id + 2 * k2.id + 2 * k3.id + k4.id) / 6,
        x.iq + dt * (k1.iq + 2 * k2.iq + 2 * k3.iq + k4.iq) / 6,
        x.w + dt * (k1.w + 2 * k2.w + 2 * k3.w + k4.w) / 6,
        x.theta + dt * (k1.theta + 2 * k2.theta + 2 * k3.theta + k4.theta) / 6
    )


# ==========================================================
# RL ENVIRONMENT
# ==========================================================

class PMSMEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.dt = 5e-5
        self.frame_skip = 20

        self.motor = PMSM()
        self.R, self.L, self.lam = self.motor.derived()

        self.state = State(0.0, 0.0, 0.0, 0.0)

        self.wmax = 700.0
        self.Imax = 2.0
        self.Vmax = self.motor.Vdc / math.sqrt(3)

        self.t = 0.0

        self.vd = 0.0
        self.vq = 0.0

        self.prev_vd = 0.0
        self.prev_vq = 0.0
        self.prev_error = 0.0
        self.error_integral = 0.0

        self.profile_speeds = None
        self.profile_times = None

        self.last_wref = 0.0

        self.action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.observation_space = spaces.Box(-1, 1, (8,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.motor = PMSM()
        self.R, self.L, self.lam = self.motor.derived()

        self.state = State(0.0, 0.0, 0.0, 0.0)

        self.vd = 0.0
        self.vq = 0.0

        self.prev_vd = 0.0
        self.prev_vq = 0.0
        self.prev_error = 0.0
        self.error_integral = 0.0

        self.t = 0.0

        self.profile_speeds, self.profile_times = generate_speed_profile()
        self.last_wref = speed_ref(0.0, self.profile_speeds, self.profile_times)

        return self._obs(), {}

    def _obs(self):
        wref = speed_ref(self.t, self.profile_speeds, self.profile_times)

        return np.array([
            self.state.id / self.Imax,
            self.state.iq / self.Imax,
            self.state.w / self.wmax,
            wref / self.wmax,
            (wref - self.state.w) / self.wmax,
            np.clip(self.error_integral, -1.0, 1.0),
            np.clip(self.prev_error, -1.0, 1.0),
            0.0
        ], dtype=np.float32)

    def step(self, action):
        vd_cmd = action[0] * self.Vmax
        vq_cmd = action[1] * self.Vmax

        reward = 0.0
        last_terms = None

        for _ in range(self.frame_skip):
            alpha = self.dt / 8e-4

            self.vd += alpha * (vd_cmd - self.vd)
            self.vq += alpha * (vq_cmd - self.vq)

            self.state = rk4(
                self.motor,
                self.R,
                self.L,
                self.lam,
                self.state,
                self.dt,
                self.vd,
                self.vq,
                0.002
            )

            self.t += self.dt

            wref = speed_ref(self.t, self.profile_speeds, self.profile_times)
            self.last_wref = wref

            e = (wref - self.state.w) / self.wmax
            de = (e - self.prev_error) / self.dt

            self.error_integral += e * self.dt
            self.error_integral = np.clip(self.error_integral, -0.5, 0.5)

            # --------------------------------------------------
            # REWARD TERMS
            # --------------------------------------------------
            tracking = -20.0 * (e ** 2)
            derivative_pen = -0.02 * (de ** 2)
            integral_pen = -0.1 * (self.error_integral ** 2)

            voltage_pen = -0.01 * (
                (self.vd / self.Vmax) ** 2 +
                (self.vq / self.Vmax) ** 2
            )

            current_pen = -0.01 * (
                (self.state.id / self.Imax) ** 2 +
                (self.state.iq / self.Imax) ** 2
            )

            du_d = (self.vd - self.prev_vd) / self.Vmax
            du_q = (self.vq - self.prev_vq) / self.Vmax
            smooth_pen = -0.01 * (du_d ** 2 + du_q ** 2)

            snap_bonus = 0.1 if abs(e) < 0.01 else 0.0

            reward_step_raw = (
                tracking +
                derivative_pen +
                integral_pen +
                voltage_pen +
                current_pen +
                smooth_pen +
                snap_bonus
            )

            reward_step = np.clip(reward_step_raw, -1.0, 1.0)
            reward += reward_step

            self.prev_error = e
            self.prev_vd = self.vd
            self.prev_vq = self.vq

            last_terms = {
                "reward_step": reward_step,
                "tracking": tracking,
                "derivative_pen": derivative_pen,
                "integral_pen": integral_pen,
                "voltage_pen": voltage_pen,
                "current_pen": current_pen,
                "smooth_pen": smooth_pen,
                "snap_bonus": snap_bonus,
                "error": e
            }

        terminated = False
        truncated = self.t > 8.0

        return self._obs(), reward, terminated, truncated, last_terms


# ==========================================================
# REWARD LOGGER CALLBACK
# ==========================================================

class RewardLoggerCallback(BaseCallback):

    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_rewards = None

    def _on_training_start(self):
        self.current_rewards = np.zeros(self.training_env.num_envs)

    def _on_step(self):
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        self.current_rewards += rewards

        for i, done in enumerate(dones):
            if done:
                self.episode_rewards.append(self.current_rewards[i])
                self.current_rewards[i] = 0.0

        return True

    def _on_training_end(self):
        r = np.array(self.episode_rewards)

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
# TRAIN
# ==========================================================

def make_env():
    def _init():
        return PMSMEnv()
    return _init


def train():
    env = SubprocVecEnv([make_env() for _ in range(8)])

    action_noise = NormalActionNoise(
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
        action_noise=action_noise,
        policy_kwargs=dict(net_arch=[128, 64])
    )

    callback = RewardLoggerCallback()

    model.learn(1_000_000, callback=callback)
    model.save("RL_models/pmsm_rl_tracking")


# ==========================================================
# EVALUATE
# ==========================================================

def evaluate():
    env = PMSMEnv()
    model = TD3.load("RL_models/pmsm_rl_tracking")

    obs, _ = env.reset()

    t = []
    w = []
    wref = []
    iq = []
    id_log = []
    vq_log = []
    vd_log = []

    reward_log = []
    tracking_log = []
    derivative_log = []
    integral_log = []
    voltage_log = []
    current_log = []
    smooth_log = []
    snap_log = []
    error_log = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, _, trunc, info = env.step(action)

        t.append(env.t)
        w.append(env.state.w)
        wref.append(env.last_wref)
        iq.append(env.state.iq)
        id_log.append(env.state.id)
        vq_log.append(env.vq)
        vd_log.append(env.vd)

        reward_log.append(info["reward_step"])
        tracking_log.append(info["tracking"])
        derivative_log.append(info["derivative_pen"])
        integral_log.append(info["integral_pen"])
        voltage_log.append(info["voltage_pen"])
        current_log.append(info["current_pen"])
        smooth_log.append(info["smooth_pen"])
        snap_log.append(info["snap_bonus"])
        error_log.append(info["error"])

        if trunc:
            break

    # ------------------------------------------------------
    # Publication-style evaluation figure
    # ------------------------------------------------------
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 10,
        "lines.linewidth": 1.5
    })

    fig, ax = plt.subplots(4, 1, figsize=(9, 8.5), sharex=True)

    ax[0].plot(t, w, label=r"$\omega$")
    ax[0].plot(t, wref, "--", label=r"$\omega_{\mathrm{ref}}$")
    ax[0].set_ylabel(r"$\omega$ [rad/s]")
    ax[0].set_title("Speed Tracking")
    ax[0].legend()

    ax[1].plot(t, iq, label=r"$i_q$")
    ax[1].plot(t, id_log, label=r"$i_d$")
    ax[1].axhline(0.0, linestyle=":", linewidth=1.0)
    ax[1].set_ylabel("Current [A]")
    ax[1].set_title("dq-axis Currents")
    ax[1].legend()

    ax[2].plot(t, vq_log, label=r"$v_q$")
    ax[2].plot(t, vd_log, label=r"$v_d$")
    ax[2].set_ylabel("Voltage [V]")
    ax[2].set_title("Control Effort")
    ax[2].legend()

    ax[3].plot(t, reward_log, label="reward")
    ax[3].set_ylabel("Reward")
    ax[3].set_xlabel("Time [s]")
    ax[3].set_title("Per-step Reward During Evaluation")
    ax[3].legend()

    for a in ax:
        a.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------
    # Reward component diagnostic plot
    # ------------------------------------------------------
    fig2, ax2 = plt.subplots(1, 1, figsize=(9, 3.8))
    ax2.plot(t, tracking_log, label="tracking")
    ax2.plot(t, derivative_log, label="derivative")
    ax2.plot(t, integral_log, label="integral")
    ax2.plot(t, voltage_log, label="voltage")
    ax2.plot(t, current_log, label="current")
    ax2.plot(t, smooth_log, label="smooth")
    ax2.plot(t, snap_log, label="snap")
    ax2.set_title("Reward Components During Evaluation")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Contribution")
    ax2.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    ax2.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------
    # Optional compact summary printed to terminal
    # ------------------------------------------------------
    abs_err = np.abs(np.array(wref) - np.array(w))
    print(f"Mean abs speed error: {np.mean(abs_err):.3f} rad/s")
    print(f"Max abs speed error : {np.max(abs_err):.3f} rad/s")
    print(f"Mean |iq|           : {np.mean(np.abs(iq)):.3f} A")
    print(f"Mean |vq|           : {np.mean(np.abs(vq_log)):.3f} V")
    print(f"Mean step reward    : {np.mean(reward_log):.3f}")


# ==========================================================
# CLI
# ==========================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--eval", action="store_true")

    args = p.parse_args()

    if args.train:
        train()

    if args.eval:
        evaluate()


if __name__ == "__main__":
    main()