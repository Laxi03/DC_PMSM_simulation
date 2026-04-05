#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
# DC MOTOR PARAMETERS
# ==========================================================

@dataclass
class AMax32Params:
    R: float = 3.99
    L: float = 0.556e-3
    kT: float = 35.2e-3
    Ke: float = 35.2e-3
    J: float = 45.3e-7
    B: float = 1.0e-6
    T_coulomb: float = 0.001506
    T_amb: float = 25.0
    n0_rpm: float = 6460.0

    def derived(self):
        omega0 = self.n0_rpm * 2.0 * math.pi / 60.0
        return {"omega0": omega0}


@dataclass
class MotorState:
    i: float
    omega: float
    theta: float
    Tw: float


# ==========================================================
# RANDOM CURRENT PROFILE
# ==========================================================

def generate_current_profile(duration: float = 6.0):
    n_segments = np.random.randint(8, 12)
    durations = np.random.uniform(0.35, 0.9, n_segments)
    t_points = np.cumsum(durations)
    t_points = t_points[t_points < duration]

    if len(t_points) == 0 or t_points[-1] < duration:
        t_points = np.concatenate(([0.0], t_points, [duration]))
    else:
        t_points = np.concatenate(([0.0], t_points))

    i_values = np.random.uniform(-1.0, 1.0, len(t_points))

    for k in range(1, len(i_values)):
        i_values[k] = 0.65 * i_values[k - 1] + 0.35 * i_values[k]

    return t_points, i_values


def current_ref(t: float, t_points: np.ndarray, i_values: np.ndarray) -> float:
    idx = np.searchsorted(t_points, t, side="right") - 1
    idx = int(np.clip(idx, 0, len(i_values) - 1))
    return float(i_values[idx])


# ==========================================================
# DC MOTOR MODEL
# ==========================================================

class AMax32Motor:
    def __init__(self, p=None):
        self.p = p or AMax32Params()
        self.d = self.p.derived()

    def derivatives(self, t: float, x: MotorState, u_v: float, load_torque: float, lock_rotor: bool) -> MotorState:
        p = self.p

        di = (u_v - p.R * x.i - p.Ke * x.omega) / p.L

        if lock_rotor:
            domega = 0.0
            dtheta = 0.0
        else:
            sign = 0.0
            if abs(x.omega) > 1e-6:
                sign = 1.0 if x.omega > 0 else -1.0

            T_fric = p.B * x.omega + p.T_coulomb * sign
            T_em = p.kT * x.i
            domega = (T_em - load_torque - T_fric) / p.J
            dtheta = x.omega

        dTw = 0.0

        return MotorState(di, domega, dtheta, dTw)

    @staticmethod
    def rk4_step(deriv, t: float, dt: float, x: MotorState) -> MotorState:
        k1 = deriv(t, x)
        k2 = deriv(
            t + 0.5 * dt,
            MotorState(
                x.i + 0.5 * dt * k1.i,
                x.omega + 0.5 * dt * k1.omega,
                x.theta + 0.5 * dt * k1.theta,
                x.Tw
            )
        )
        k3 = deriv(
            t + 0.5 * dt,
            MotorState(
                x.i + 0.5 * dt * k2.i,
                x.omega + 0.5 * dt * k2.omega,
                x.theta + 0.5 * dt * k2.theta,
                x.Tw
            )
        )
        k4 = deriv(
            t + dt,
            MotorState(
                x.i + dt * k3.i,
                x.omega + dt * k3.omega,
                x.theta + dt * k3.theta,
                x.Tw
            )
        )

        return MotorState(
            x.i + (dt / 6.0) * (k1.i + 2 * k2.i + 2 * k3.i + k4.i),
            x.omega + (dt / 6.0) * (k1.omega + 2 * k2.omega + 2 * k3.omega + k4.omega),
            x.theta + (dt / 6.0) * (k1.theta + 2 * k2.theta + 2 * k3.theta + k4.theta),
            x.Tw
        )


# ==========================================================
# RL ENVIRONMENT
# ==========================================================

class DCMotorCurrentEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        dt=1e-4,
        frame_skip=20,
        Vdc=24.0,
        ep_len_s=6.0,
        tau_act=7e-4,
        load_torque=0.0,
        lock_rotor=True
    ):
        super().__init__()

        self.dt = dt
        self.frame_skip = frame_skip
        self.ep_len_s = ep_len_s
        self.max_rl_steps = max(1, int(self.ep_len_s / (self.dt * self.frame_skip)))

        self.Vdc = Vdc
        self.tau_act = tau_act
        self.base_load_torque = load_torque
        self.lock_rotor = lock_rotor

        self.p = AMax32Params()
        self.motor = AMax32Motor(self.p)

        self.state = MotorState(0.0, 0.0, 0.0, self.p.T_amb)

        self.t = 0.0
        self.rl_step_count = 0

        self.u_applied = 0.0
        self.prev_u = 0.0
        self.prev_error = 0.0
        self.error_integral = 0.0

        self.ref_t = None
        self.ref_i = None
        self.last_iref = 0.0

        self.i_abs_max = 2.0
        self.omega_abs_max = 900.0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # [i_n, omega_n, u_n, iref_n, error_n, int_e]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = MotorState(0.0, 0.0, 0.0, self.p.T_amb)
        self.t = 0.0
        self.rl_step_count = 0

        self.u_applied = 0.0
        self.prev_u = 0.0
        self.prev_error = 0.0
        self.error_integral = 0.0

        self.ref_t, self.ref_i = generate_current_profile(self.ep_len_s)
        self.last_iref = current_ref(0.0, self.ref_t, self.ref_i)

        return self._obs(), {}

    def _obs(self):
        iref = current_ref(self.t, self.ref_t, self.ref_i)
        e = (iref - self.state.i) / self.i_abs_max

        return np.array([
            np.clip(self.state.i / self.i_abs_max, -1.0, 1.0),
            np.clip(self.state.omega / self.omega_abs_max, -1.0, 1.0),
            np.clip(self.prev_u / self.Vdc, -1.0, 1.0),
            np.clip(iref / self.i_abs_max, -1.0, 1.0),
            np.clip(e, -1.0, 1.0),
            np.clip(self.error_integral, -1.0, 1.0)
        ], dtype=np.float32)

    def step(self, action):
        duty = float(np.clip(action[0], -1.0, 1.0))
        u_cmd = self.Vdc * duty

        reward = 0.0
        last_terms = None

        for _ in range(self.frame_skip):
            alpha = self.dt / max(self.tau_act, 1e-6)
            alpha = min(alpha, 1.0)
            self.u_applied += alpha * (u_cmd - self.u_applied)

            self.state = AMax32Motor.rk4_step(
                lambda tau, s: self.motor.derivatives(
                    tau,
                    s,
                    self.u_applied,
                    self.base_load_torque,
                    self.lock_rotor
                ),
                self.t,
                self.dt,
                self.state
            )
            self.t += self.dt

            self.state.i = float(np.clip(self.state.i, -1.2 * self.i_abs_max, 1.2 * self.i_abs_max))
            self.state.omega = float(np.clip(self.state.omega, -1.2 * self.omega_abs_max, 1.2 * self.omega_abs_max))

            iref = current_ref(self.t, self.ref_t, self.ref_i)
            self.last_iref = iref

            e = (iref - self.state.i) / self.i_abs_max
            de = np.clip((e - self.prev_error) / self.dt, -80.0, 80.0)

            self.error_integral += e * self.dt
            self.error_integral = float(np.clip(self.error_integral, -0.35, 0.35))

            u_norm = self.u_applied / self.Vdc
            i_norm = self.state.i / self.i_abs_max
            du = (self.u_applied - self.prev_u) / self.Vdc

            tracking = -20.0 * (e ** 2)
            derivative_pen = -0.00005 * (de ** 2)
            integral_pen = -0.08 * (self.error_integral ** 2)
            effort_pen = -0.001 * (u_norm ** 2)
            current_pen = -0.002 * (i_norm ** 2)
            smooth_pen = -0.002 * (du ** 2)
            snap_bonus = 0.10 if abs(e) < 0.02 else 0.0

            reward_step_raw = (
                tracking +
                derivative_pen +
                integral_pen +
                effort_pen +
                current_pen +
                smooth_pen +
                snap_bonus
            )

            reward_step = float(np.clip(reward_step_raw, -1.0, 0.3))
            reward += reward_step

            self.prev_error = e
            self.prev_u = self.u_applied

            last_terms = {
                "reward_step": reward_step,
                "tracking": tracking,
                "derivative_pen": derivative_pen,
                "integral_pen": integral_pen,
                "effort_pen": effort_pen,
                "current_pen": current_pen,
                "smooth_pen": smooth_pen,
                "snap_bonus": snap_bonus,
                "error": e
            }

        self.rl_step_count += 1
        terminated = False
        truncated = self.rl_step_count >= self.max_rl_steps

        return self._obs(), reward, terminated, truncated, last_terms


# ==========================================================
# CALLBACK
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
        return DCMotorCurrentEnv(
            dt=1e-4,
            frame_skip=20,
            Vdc=24.0,
            ep_len_s=6.0,
            tau_act=7e-4,
            load_torque=0.0,
            lock_rotor=True
        )
    return _init


def train():
    env = SubprocVecEnv([make_env() for _ in range(8)])

    action_noise = NormalActionNoise(
        mean=np.zeros(1),
        sigma=0.08 * np.ones(1)
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
        train_freq=(1, "step"),
        gradient_steps=1,
        action_noise=action_noise,
        policy_kwargs=dict(net_arch=[128, 64])
    )

    callback = RewardLoggerCallback()

    model.learn(1_000_000, callback=callback)
    model.save("RL_models/dc_rl_current")


# ==========================================================
# EVALUATION
# ==========================================================

def evaluate():
    env = DCMotorCurrentEnv(
        dt=1e-4,
        frame_skip=20,
        Vdc=24.0,
        ep_len_s=6.0,
        tau_act=7e-4,
        load_torque=0.0,
        lock_rotor=True
    )
    model = TD3.load("RL_models/dc_rl_current.zip")

    obs, _ = env.reset()

    t = []
    u_log = []
    i_log = []
    iref_log = []
    w_log = []
    torque_log = []

    reward_log = []
    tracking_log = []
    derivative_log = []
    integral_log = []
    effort_log = []
    current_log = []
    smooth_log = []
    snap_log = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, trunc, info = env.step(action)

        t.append(env.t)
        u_log.append(env.prev_u)
        i_log.append(env.state.i)
        iref_log.append(env.last_iref)
        w_log.append(env.state.omega)
        torque_log.append(env.p.kT * env.state.i)

        reward_log.append(info["reward_step"])
        tracking_log.append(info["tracking"])
        derivative_log.append(info["derivative_pen"])
        integral_log.append(info["integral_pen"])
        effort_log.append(info["effort_pen"])
        current_log.append(info["current_pen"])
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

    fig, ax = plt.subplots(4, 1, figsize=(9, 8.5), sharex=True)

    ax[0].plot(t, i_log, label=r"$i$")
    ax[0].plot(t, iref_log, "--", label=r"$i_{\mathrm{ref}}$")
    ax[0].set_ylabel("Current [A]")
    ax[0].set_title("Current Tracking")
    ax[0].legend()

    ax[1].plot(t, w_log, label=r"$\omega$")
    ax[1].set_ylabel(r"$\omega$ [rad/s]")
    ax[1].set_title("Mechanical Response During Current Control")
    ax[1].legend()

    ax[2].plot(t, u_log, label=r"$u$")
    ax[2].set_ylabel("Voltage [V]")
    ax[2].set_title("Control Effort")
    ax[2].legend()

    ax[3].plot(t, torque_log, label=r"$T_{\mathrm{em}}$")
    ax[3].set_ylabel(r"$T_{\mathrm{em}}$ [Nm]")
    ax[3].set_xlabel("Time [s]")
    ax[3].set_title("Electromagnetic Torque")
    ax[3].legend()

    for a in ax:
        a.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots(1, 1, figsize=(9, 3.8))
    ax2.plot(t, tracking_log, label="tracking")
    ax2.plot(t, derivative_log, label="derivative")
    ax2.plot(t, integral_log, label="integral")
    ax2.plot(t, effort_log, label="effort")
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

    fig3, ax3 = plt.subplots(1, 1, figsize=(9, 3.8))
    ax3.plot(t, reward_log, label="reward")
    ax3.set_title("Per-step Reward During Evaluation")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Reward")
    ax3.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    ax3.legend()
    plt.tight_layout()
    plt.show()

    abs_err = np.abs(np.array(iref_log) - np.array(i_log))
    print(f"Mean abs current error : {np.mean(abs_err):.4f} A")
    print(f"Max abs current error  : {np.max(abs_err):.4f} A")
    print(f"Mean |u|               : {np.mean(np.abs(u_log)):.4f} V")
    print(f"Final speed            : {w_log[-1]:.4f} rad/s")
    print(f"Mean step reward       : {np.mean(reward_log):.4f}")


# ==========================================================
# CLI
# ==========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args()

    if args.train:
        train()

    if args.eval:
        evaluate()


if __name__ == "__main__":
    main()