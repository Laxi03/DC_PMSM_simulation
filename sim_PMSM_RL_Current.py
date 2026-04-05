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
# MOTOR
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


@dataclass
class State:
    id: float
    iq: float
    w: float
    theta: float


# ==========================================================
# REFERENCE
# ==========================================================

def generate_iq_profile():
    n = np.random.randint(8, 12)
    iq_vals = np.random.uniform(-1.0, 1.0, n)
    durations = np.random.uniform(0.4, 1.0, n)
    times = np.cumsum(durations)
    return iq_vals, times


def iq_ref_func(t, vals, times):
    for i in range(len(times)):
        if t < times[i]:
            return vals[i]
    return vals[-1]


# ==========================================================
# MODEL
# ==========================================================

def derivatives(motor, R, L, lam, x, vd, vq, lock_rotor):
    p = motor.pole_pairs
    we = p * x.w

    did = (vd - R * x.id + we * L * x.iq) / L
    diq = (vq - R * x.iq - we * (L * x.id + lam)) / L

    Te = 1.5 * p * lam * x.iq

    if lock_rotor:
        dw = 0.0
        dth = 0.0
    else:
        dw = (Te - motor.B * x.w) / motor.J
        dth = we

    return State(did, diq, dw, dth)


def rk4(motor, R, L, lam, x, dt, vd, vq, lock_rotor):

    def f(s):
        return derivatives(motor, R, L, lam, s, vd, vq, lock_rotor)

    k1 = f(x)
    k2 = f(State(x.id + dt * k1.id / 2, x.iq + dt * k1.iq / 2, x.w + dt * k1.w / 2, x.theta))
    k3 = f(State(x.id + dt * k2.id / 2, x.iq + dt * k2.iq / 2, x.w + dt * k2.w / 2, x.theta))
    k4 = f(State(x.id + dt * k3.id, x.iq + dt * k3.iq, x.w + dt * k3.w, x.theta))

    return State(
        x.id + dt * (k1.id + 2 * k2.id + 2 * k3.id + k4.id) / 6,
        x.iq + dt * (k1.iq + 2 * k2.iq + 2 * k3.iq + k4.iq) / 6,
        x.w + dt * (k1.w + 2 * k2.w + 2 * k3.w + k4.w) / 6,
        x.theta
    )


# ==========================================================
# ENVIRONMENT
# ==========================================================

class PMSMCurrentEnv(gym.Env):

    def __init__(self, lock_rotor=True):
        super().__init__()

        self.lock_rotor = lock_rotor

        self.dt = 5e-5
        self.frame_skip = 20

        self.motor = PMSM()
        self.R, self.L, self.lam = self.motor.derived()

        self.state = State(0.0, 0.0, 0.0, 0.0)

        self.Imax = 2.0
        self.Vmax = self.motor.Vdc / math.sqrt(3)

        self.vd = 0.0
        self.vq = 0.0
        self.prev_vq = 0.0

        self.t = 0.0
        self.int_e = 0.0

        self.profile_vals = None
        self.profile_times = None
        self.last_iq_ref = 0.0

        self.action_space = spaces.Box(-1, 1, (1,), dtype=np.float32)

        # iq, id, iq_ref, error, integral, vq
        self.observation_space = spaces.Box(-1, 1, (6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = State(0.0, 0.0, 0.0, 0.0)
        self.vd = 0.0
        self.vq = 0.0
        self.prev_vq = 0.0
        self.t = 0.0
        self.int_e = 0.0

        self.profile_vals, self.profile_times = generate_iq_profile()
        self.last_iq_ref = iq_ref_func(0.0, self.profile_vals, self.profile_times)

        return self._obs(), {}

    def _obs(self):
        iq_ref = iq_ref_func(self.t, self.profile_vals, self.profile_times)
        e = (iq_ref - self.state.iq) / self.Imax

        return np.array([
            self.state.iq / self.Imax,
            self.state.id / self.Imax,
            iq_ref / self.Imax,
            e,
            np.clip(self.int_e, -1.0, 1.0),
            self.vq / self.Vmax
        ], dtype=np.float32)

    def step(self, action):
        reward = 0.0
        last_terms = None

        for _ in range(self.frame_skip):
            we = self.motor.pole_pairs * self.state.w

            # decoupling
            self.vd = -we * self.L * self.state.iq
            vq_base = we * (self.L * self.state.id + self.lam)

            vq_cmd = vq_base + action[0] * 0.7 * self.Vmax

            alpha = self.dt / 5e-4
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
                self.lock_rotor
            )

            self.t += self.dt

            iq_ref = iq_ref_func(self.t, self.profile_vals, self.profile_times)
            self.last_iq_ref = iq_ref

            e = (iq_ref - self.state.iq) / self.Imax

            self.int_e += e * self.dt
            self.int_e = np.clip(self.int_e, -0.5, 0.5)

            # simplified reward
            tracking = -20.0 * (e ** 2)
            id_pen = -1.0 * (self.state.id / self.Imax) ** 2
            integral_pen = -0.05 * (self.int_e ** 2)

            du = (self.vq - self.prev_vq) / self.Vmax
            smooth = -0.005 * (du ** 2)

            snap_bonus = 0.05 if abs(e) < 0.01 else 0.0

            reward_step_raw = tracking + id_pen + integral_pen + smooth + snap_bonus
            reward_step = np.clip(reward_step_raw, -1.0, 0.5)

            reward += reward_step
            self.prev_vq = self.vq

            last_terms = {
                "reward_step": reward_step,
                "tracking": tracking,
                "id_pen": id_pen,
                "integral_pen": integral_pen,
                "smooth_pen": smooth,
                "snap_bonus": snap_bonus,
                "error": e
            }

        reward /= self.frame_skip

        terminated = False
        truncated = self.t > 6.0

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
# TRAIN
# ==========================================================

def make_env(lock_rotor=True):
    def _init():
        return PMSMCurrentEnv(lock_rotor=lock_rotor)
    return _init


def train(lock_rotor=True):
    env = SubprocVecEnv([make_env(lock_rotor=lock_rotor) for _ in range(8)])

    noise = NormalActionNoise(
        mean=np.zeros(1),
        sigma=0.1 * np.ones(1)
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
        policy_kwargs=dict(net_arch=[128, 64])
    )

    callback = RewardLogger()

    model.learn(1_000_000, callback=callback)
    model.save("RL_models/pmsm_rl_current")


# ==========================================================
# EVAL
# ==========================================================

def evaluate(lock_rotor=True):
    env = PMSMCurrentEnv(lock_rotor=lock_rotor)
    model = TD3.load("RL_models/pmsm_rl_current")

    obs, _ = env.reset()

    t = []
    iq = []
    iq_ref = []
    id_log = []
    vq_log = []
    vd_log = []
    w_log = []

    reward_log = []
    tracking_log = []
    id_pen_log = []
    integral_log = []
    smooth_log = []
    snap_log = []

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
        tracking_log.append(info["tracking"])
        id_pen_log.append(info["id_pen"])
        integral_log.append(info["integral_pen"])
        smooth_log.append(info["smooth_pen"])
        snap_log.append(info["snap_bonus"])

        if trunc:
            break

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 10,
        "lines.linewidth": 1.5
    })

    fig, ax = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True)

    ax[0].plot(t, iq, label=r"$i_q$")
    ax[0].plot(t, iq_ref, "--", label=r"$i_{q,\mathrm{ref}}$")
    ax[0].plot(t, id_log, label=r"$i_d$")
    ax[0].axhline(0.0, linestyle=":", linewidth=1.0)
    ax[0].set_ylabel("Current [A]")
    ax[0].set_title("dq-axis Currents")
    ax[0].legend()

    ax[1].plot(t, vq_log, label=r"$v_q$")
    ax[1].plot(t, vd_log, label=r"$v_d$")
    ax[1].set_ylabel("Voltage [V]")
    ax[1].set_title("Control Effort")
    ax[1].legend()

    ax[2].plot(t, reward_log, label="reward")
    ax[2].set_ylabel("Reward")
    ax[2].set_xlabel("Time [s]")
    ax[2].set_title("Per-step Reward During Evaluation")
    ax[2].legend()

    for a in ax:
        a.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots(1, 1, figsize=(9, 3.8))
    ax2.plot(t, tracking_log, label="tracking")
    ax2.plot(t, id_pen_log, label="id")
    ax2.plot(t, integral_log, label="integral")
    ax2.plot(t, smooth_log, label="smooth")
    ax2.plot(t, snap_log, label="snap")
    ax2.set_title("Reward Components During Evaluation")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Contribution")
    ax2.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    ax2.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.show()

    abs_err = np.abs(np.array(iq_ref) - np.array(iq))
    print(f"Mean abs iq error : {np.mean(abs_err):.4f} A")
    print(f"Max abs iq error  : {np.max(abs_err):.4f} A")
    print(f"Mean |id|         : {np.mean(np.abs(id_log)):.4f} A")
    print(f"Mean |vq|         : {np.mean(np.abs(vq_log)):.4f} V")
    print(f"Mean step reward  : {np.mean(reward_log):.4f}")
    print(f"Final speed       : {w_log[-1]:.4f} rad/s")


# ==========================================================
# CLI
# ==========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--free_rotor", action="store_true")

    args = parser.parse_args()

    lock_rotor = not args.free_rotor

    if args.train:
        train(lock_rotor=lock_rotor)

    if args.eval:
        evaluate(lock_rotor=lock_rotor)


if __name__ == "__main__":
    main()