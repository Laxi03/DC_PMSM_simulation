#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
motor_rl.py — DC motor RL-only (TD3, Gymnasium + Stable-Baselines3)

Robust setup (no extra u cap):
- Soft-safety on states (clip + gentle penalties), no early termination
- Env never returns NaN/Inf (sanitized obs/reward, mid-step guard)
- Randomized step references (incl. negatives) + domain randomization
- Huber tracking + (phase-aware) authority penalties + terminal bonus
- Actuator low-pass (prevents chatter)
- TD3 with action noise and conservative hyperparams (+ resume training)
- Live evaluation: slowed playback, autoscaling axes, clear labels
"""

from typing import Tuple, Optional
from dataclasses import dataclass
import os, math, argparse, warnings, time
import numpy as np
from datetime import datetime

# ---------- Backend ----------
def _choose_backend(headless: bool):
    import matplotlib
    if headless:
        matplotlib.use("Agg")
    else:
        try:
            matplotlib.use("MacOSX")
        except Exception:
            matplotlib.use("TkAgg")
    return matplotlib

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPORT_ROOT = "exports"
RUN_DIR = os.path.join(EXPORT_ROOT, RUN_TIMESTAMP)
os.makedirs(RUN_DIR, exist_ok=True)

# =========================
# Motor model
# =========================
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
    n_nom_rpm: float = 5060.0
    def derived(self):
        rpm_to_radps = lambda rpm: rpm * 2.0 * math.pi / 60.0
        return {"omega0": rpm_to_radps(self.n0_rpm)}

@dataclass
class MotorState:
    i: float
    omega: float
    theta: float
    Tw: float

def _clip(v, lo, hi): return min(max(v, lo), hi)

class AMax32Motor:
    def __init__(self, p: Optional[AMax32Params] = None):
        self.p = p or AMax32Params()
        self.d = self.p.derived()
    def derivatives(self, t, x: MotorState, u_v: float, load_torque: float) -> MotorState:
        p = self.p
        di = (u_v - p.R * x.i - p.Ke * x.omega) / p.L
        sign = 0.0 if abs(x.omega) < 1e-6 else (1.0 if x.omega > 0 else -1.0)
        T_fric = p.B * x.omega + p.T_coulomb * sign
        T_em = p.kT * x.i
        domega = (T_em - load_torque - T_fric) / p.J
        dtheta = x.omega
        dTw = 0.0
        return MotorState(di, domega, dtheta, dTw)
    @staticmethod
    def rk4_step(deriv, t, dt, x: MotorState) -> MotorState:
        k1 = deriv(t, x)
        k2 = deriv(t+0.5*dt, MotorState(x.i+0.5*dt*k1.i, x.omega+0.5*dt*k1.omega, x.theta, x.Tw))
        k3 = deriv(t+0.5*dt, MotorState(x.i+0.5*dt*k2.i, x.omega+0.5*dt*k2.omega, x.theta, x.Tw))
        k4 = deriv(t+dt, MotorState(x.i+dt*k3.i, x.omega+dt*k3.omega, x.theta, x.Tw))
        return MotorState(
            x.i+(dt/6)*(k1.i+2*k2.i+2*k3.i+k4.i),
            x.omega+(dt/6)*(k1.omega+2*k2.omega+2*k3.omega+k4.omega),
            x.theta, x.Tw
        )

# =========================
# References
# =========================
def generate_random_steps(duration: float,
                          min_val=-600.0, max_val=600.0,
                          min_seg=0.3, max_seg=1.2,
                          skew_high_prob=0.5, high_lo=300.0) -> Tuple[np.ndarray, np.ndarray]:
    """Random step sequence, optionally skewed to spend more time at high ω."""
    t_points = [0.0]
    while t_points[-1] < duration:
        t_points.append(min(duration, t_points[-1] + float(np.random.uniform(min_seg, max_seg))))
    t_points = np.unique(np.array(t_points, dtype=float))
    if t_points[-1] < duration:
        t_points = np.append(t_points, duration)
    vals = []
    for _ in range(len(t_points)):
        if np.random.rand() < skew_high_prob:
            vals.append(np.random.uniform(high_lo, max_val))
        else:
            vals.append(np.random.uniform(min_val, high_lo))
    w_values = np.array(vals, dtype=float)
    return t_points, w_values

def random_speed_ref(t: float, t_points: np.ndarray, w_values: np.ndarray) -> float:
    idx = np.searchsorted(t_points, t, side="right") - 1
    idx = int(_clip(idx, 0, len(w_values)-1))
    return float(w_values[idx])

def speed_ref_profile(t: float, kind: str, w_ref: float, rand_seq=None) -> float:
    if kind == "const":
        return float(w_ref)
    if kind == "step_seq_motor":
        if t < 1.0: return 200.0
        if t < 2.0: return 325.0
        if t < 3.0: return 50.0
        if t < 4.0: return 150.0
        return 523.60
    if kind == "rand_steps" and rand_seq is not None:
        t_points, w_values = rand_seq
        return random_speed_ref(t, t_points, w_values)
    return 0.0

# =========================
# Gym environment
# =========================
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    gym = None
    warnings.warn("Gymnasium not available. Install: pip install gymnasium")

def huber(x: float, k: float = 0.05) -> float:
    ax = abs(x)
    return (0.5 * (ax**2) / k) if ax <= k else (ax - 0.5 * k)

class MotorEnv(gym.Env):  # type: ignore[misc]
    metadata = {"render_modes": []}
    def __init__(self,
                 task="speed",
                 dt=1e-4,
                 frame_skip=20,            # faster control by default (≈500 Hz)
                 Vdc=24.0,
                 ulim=1.0,
                 ref_kind_speed="rand_steps",
                 w_ref=160.0,
                 # reward weights (normalized units)
                 rw_e=4.0, rw_u=0.01, rw_de=0.02, rw_du=0.005, rw_i=0.01,
                 ep_len_s=8.0,
                 # phase profile for domain rand / noise
                 phase="b",
                 # state safety (no voltage cap)
                 i_abs_max=6.0,
                 omega_abs_max=900.0,
                 # noise
                 obs_noise=0.002,
                 load_noise=0.06,
                 # training disturbances
                 load_step_prob=0.4,
                 base_load=0.02,
                 # actuator lag
                 tau_act=1e-3):
        assert task == "speed"
        self.task, self.dt, self.frame_skip = task, dt, frame_skip
        self.Vdc_nom = Vdc
        self.ulim = ulim
        self.ref_kind_speed, self.w_ref_base = ref_kind_speed, w_ref

        self.rw_e, self.rw_u, self.rw_de, self.rw_du, self.rw_i = rw_e, rw_u, rw_de, rw_du, rw_i
        self.ep_len_s = float(ep_len_s)
        self.max_rl_steps = max(1, int(self.ep_len_s / (self.dt * self.frame_skip)))

        self.phase = phase
        self.base_load = base_load
        self.load_step_prob = load_step_prob
        self.load_noise = float(load_noise)
        self.obs_noise = float(obs_noise)

        # state soft-clamps (no hard cap on voltage)
        self.i_abs_max = float(i_abs_max)
        self.omega_abs_max = float(omega_abs_max)
        self.i_env_max = 1.2 * self.i_abs_max
        self.omega_env_max = 1.2 * self.omega_abs_max

        # actuator low-pass
        self.tau_act = float(tau_act)
        self.u_applied = 0.0

        # authority scaling (phase-dependent; set in reset)
        self.authority_gain = 1.5

        self.p = AMax32Params()
        self.motor = AMax32Motor(self.p)
        self.state = MotorState(0.0, 0.0, 0.0, self.p.T_amb)

        self.t = 0.0
        self.prev_u = 0.0
        self.prev_e_norm = 0.0
        self.prev_duty = 0.0
        self.rl_step_count = 0

        self.rand_seq: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # obs = [omega_n, i_n, u_n, e_norm, headroom, sin(t)]
        obs_high = np.ones(6, np.float32)
        self.observation_space = spaces.Box(-obs_high, high=obs_high, dtype=np.float32)

        # runtime-randomized parameters (set in reset)
        self.Vdc = self.Vdc_nom
        self.load_torque = self.base_load
        self.load_bump = 0.0

    # ------------- helpers
    def _current_wref(self) -> float:
        return speed_ref_profile(self.t, self.ref_kind_speed, self.w_ref_base, self.rand_seq)

    def _finite(self) -> bool:
        s = self.state
        return np.isfinite(s.i) and np.isfinite(s.omega)

    def _headroom(self) -> float:
        """0..1: how much volt headroom remains after back-EMF."""
        Vemf = abs(self.motor.p.Ke * self.state.omega)
        return float(np.clip((self.Vdc - Vemf) / max(self.Vdc, 1e-6), 0.0, 1.0))

    def _authority_multiplier(self) -> float:
        """Penalty scaling; Phase B sets authority_gain=0 to avoid discouraging high-ω control."""
        return 1.0 + self.authority_gain * (1.0 - self._headroom())

    # ------------- Gym API
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # --- PHASE curriculum ---
        if self.phase.lower() == "a":
            scale = lambda x, pct: float(x * np.random.uniform(1 - pct, 1 + pct))
            self.p.R = scale(self.p.R, 0.10); self.p.L = scale(self.p.L, 0.10)
            self.p.J = scale(self.p.J, 0.10); self.p.B = scale(self.p.B, 0.15)
            self.p.kT = scale(self.p.kT, 0.08); self.p.Ke = scale(self.p.Ke, 0.08)
            self.Vdc  = scale(self.Vdc_nom, 0.05)
            self.base_load = scale(self.base_load, 0.20)
            self.load_noise = 0.06; self.obs_noise = 0.002
            self.authority_gain = 1.5  # gentle scaling
            self.rand_seq = generate_random_steps(self.ep_len_s, -500.0, 500.0, 0.4, 1.1, skew_high_prob=0.4) \
                            if self.ref_kind_speed == "rand_steps" else None
        else:
            scale = lambda x, pct: float(x * np.random.uniform(1 - pct, 1 + pct))
            self.p.R = scale(self.p.R, 0.20); self.p.L = scale(self.p.L, 0.20)
            self.p.J = scale(self.p.J, 0.25); self.p.B = scale(self.p.B, 0.30)
            self.p.kT = scale(self.p.kT, 0.15); self.p.Ke = scale(self.p.Ke, 0.15)
            self.Vdc  = scale(self.Vdc_nom, 0.10)
            self.base_load = scale(self.base_load, 0.35)
            self.load_noise = 0.10; self.obs_noise = 0.003
            self.authority_gain = 0.0  # **disable** to allow strong actuation at high ω
            self.rand_seq = generate_random_steps(self.ep_len_s, -650.0, 650.0, 0.3, 1.0, skew_high_prob=0.6) \
                            if self.ref_kind_speed == "rand_steps" else None

        # potential load bump mid-episode
        self.load_bump = 0.0
        if np.random.rand() < self.load_step_prob:
            bump_mag = float(np.random.uniform(-0.015, 0.03))
            bump_t = float(np.random.uniform(0.7, self.ep_len_s - 0.7))
            self.load_bump = (bump_t, bump_mag)

        self.motor = AMax32Motor(self.p)
        self.state = MotorState(0.0, float(np.random.uniform(-10, 10)), 0.0, self.p.T_amb)
        self.t = 0.0
        self.prev_u = 0.0
        self.prev_duty = 0.0
        self.u_applied = 0.0  # actuator state
        wref0 = self._current_wref()
        self.prev_e_norm = float(np.clip((wref0 - self.state.omega) / 900.0, -1.0, 1.0))
        self.rl_step_count = 0
        return self._obs(), {}

    def _obs(self):
        # soft clamp internal state
        self.state.i = float(np.clip(self.state.i, -self.i_env_max, self.i_env_max))
        self.state.omega = float(np.clip(self.state.omega, -self.omega_env_max, self.omega_env_max))

        omega_n = float(np.clip(self.state.omega / 900.0, -1.0, 1.0))
        i_n     = float(np.clip(self.state.i     / 6.0,   -1.0, 1.0))
        u_n     = float(np.clip(self.prev_u      / self.Vdc, -1.0, 1.0))
        e_norm  = float(np.clip((self._current_wref() - self.state.omega) / 900.0, -1.0, 1.0))
        headroom = self._headroom()
        s = float(np.sin(self.t))
        obs = np.array([omega_n, i_n, u_n, e_norm, headroom, s], dtype=np.float32)
        if self.obs_noise > 0.0:
            obs = np.clip(obs + np.random.normal(0.0, self.obs_noise, size=obs.shape).astype(np.float32), -1.0, 1.0)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return obs

    def step(self, action):
        # guard numerical issues
        if not self._finite():
            obs, _ = self.reset()
            return obs, 0.0, False, True, {}

        duty = float(np.clip(action[0], -self.ulim, self.ulim))
        u_cmd = self.Vdc * duty  # <-- no extra clamp on u
        rew_sum = 0.0

        for _ in range(self.frame_skip):
            # actuator low-pass: u_applied <- u_cmd
            alpha = self.dt / max(self.tau_act, 1e-6)
            self.u_applied += alpha * (u_cmd - self.u_applied)
            u_eff = self.u_applied

            # load: base + noise + optional bump
            Tl = self.base_load * (1.0 + np.random.uniform(-self.load_noise, self.load_noise))
            if self.load_bump and self.t >= self.load_bump[0]:
                Tl += self.load_bump[1]

            # integrate
            self.state = AMax32Motor.rk4_step(
                lambda tau, s: self.motor.derivatives(tau, s, u_eff, Tl),
                self.t, self.dt, self.state
            )
            self.t += self.dt

            if not self._finite():
                obs, _ = self.reset()
                return obs, -0.2, False, True, {}

            # normalized vars
            wref   = self._current_wref()
            e_norm = float(np.clip((wref - self.state.omega) / 900.0, -1.0, 1.0))
            u_norm = float(np.clip(u_eff / self.Vdc, -1.0, 1.0))
            de_norm = e_norm - self.prev_e_norm
            du_norm = duty - self.prev_duty
            i_norm  = float(np.clip(self.state.i / 6.0, -1.0, 1.0))

            # authority-aware penalties (phase dependent)
            m = self._authority_multiplier()

            r = - ( self.rw_e * huber(e_norm, k=0.05)
                    + m * self.rw_u  * (u_norm**2)
                    + m * self.rw_du * abs(du_norm)
                    + self.rw_de * abs(de_norm)
                    + self.rw_i  * (i_norm**2) )

            # soft-safety nudges near state clamps
            if abs(self.state.i) > 0.9 * self.i_abs_max:          r -= 0.05
            if abs(self.state.omega) > 0.9 * self.omega_abs_max:   r -= 0.05

            r = float(np.clip(r, -2.0, 0.3))
            rew_sum += r

            self.prev_e_norm = e_norm
            self.prev_duty = duty
            self.prev_u = float(u_eff)

        # terminal bonus for good tracking
        if (self.rl_step_count + 1) >= self.max_rl_steps and abs(self.prev_e_norm) < 0.03:
            rew_sum += 0.2

        rew_sum = float(np.clip(np.nan_to_num(rew_sum, nan=0.0, posinf=0.0, neginf=-1.0), -10.0, 10.0))

        self.rl_step_count += 1
        truncated = self.rl_step_count >= self.max_rl_steps
        return self._obs(), rew_sum, False, truncated, {}

# =========================
# Plotting helpers
# =========================
def save_plots(data, title, run_dir, live=False):
    import matplotlib.pyplot as plt
    t = data["t"]
    fig, axs = plt.subplots(4,1,figsize=(11,10),sharex=True)
    axs[0].plot(t,data["u"], label="u")
    axs[0].set_ylabel("u [V]"); axs[0].legend(); axs[0].grid(True)
    axs[1].plot(t,data["i"],label="i"); axs[1].plot(t,data["i_ref"],"--",label="i_ref (n/a)")
    axs[1].set_ylabel("i [A]"); axs[1].legend(); axs[1].grid(True)
    axs[2].plot(t,data["omega"],label="ω"); axs[2].plot(t,data["omega_ref"],"--",label="ω_ref")
    axs[2].set_ylabel("ω [rad/s]"); axs[2].legend(); axs[2].grid(True)
    axs[3].plot(t,data["T_em"],label="T_em"); axs[3].plot(t,data["T_load"],"--",label="T_load")
    axs[3].set_ylabel("Torque [Nm]"); axs[3].legend(); axs[3].grid(True)
    axs[3].set_xlabel("t [s]"); fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir,"plot.png"),dpi=300)
    fig.savefig(os.path.join(run_dir,"plot.pdf"))
    if live: plt.show()
    plt.close(fig)

# =========================
# RL (TD3) — Train / Evaluate
# =========================
def make_env(args):
    return MotorEnv(
        task=args.task,
        dt=args.dt,
        frame_skip=args.frame_skip,
        Vdc=24.0,
        ulim=1.0,
        ref_kind_speed=args.ref_profile,
        w_ref=args.w_ref,
        rw_e=args.rw_e, rw_u=args.rw_u, rw_de=args.rw_de, rw_du=args.rw_du, rw_i=args.rw_i,
        ep_len_s=args.ep_len_s,
        phase=args.phase,
        i_abs_max=args.i_abs_max,
        omega_abs_max=args.omega_abs_max,
        tau_act=args.tau_act,
    )

def train_rl(args):
    try:
        from stable_baselines3 import TD3
        from stable_baselines3.common.noise import NormalActionNoise
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.callbacks import EvalCallback
        try:
            from stable_baselines3.common.callbacks import ProgressBarCallback
            have_pbar = True
        except Exception:
            have_pbar = False
    except Exception:
        print("Stable-Baselines3 not available. Install: pip install 'stable-baselines3[extra]' torch gymnasium")
        return

    env = Monitor(make_env(args))
    eval_env = Monitor(make_env(args))

    print(f"[TRAIN] Export dir: {RUN_DIR}")
    print(f"[TRAIN] TD3 total_timesteps={args.total_timesteps}  frame_skip={args.frame_skip}  ep_len_s={args.ep_len_s}  phase={args.phase.upper()}")

    action_noise = NormalActionNoise(mean=np.zeros(1), sigma=0.10 * np.ones(1))

    # Fresh or resume
    model = None
    if args.resume_from:
        path = os.path.expanduser(args.resume_from)
        if not path.endswith(".zip"): path += ".zip"
        if os.path.exists(path):
            print(f"[TRAIN] Resuming from {path}")
            model = TD3.load(path, device="auto")
            model.set_env(env)
    if model is None:
        model = TD3(
            "MlpPolicy",
            env,
            verbose=2,
            device="auto",
            seed=args.seed,
            learning_rate=3e-4,
            buffer_size=300_000,
            batch_size=128,
            learning_starts=5000,
            train_freq=(1, "step"),
            gradient_steps=1,
            gamma=0.99,
            tau=0.005,
            policy_kwargs=dict(net_arch=[128, 128]),
            action_noise=action_noise,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
        )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=RUN_DIR,
        log_path=RUN_DIR,
        eval_freq=max(2000 // max(1, args.frame_skip), 500),
        n_eval_episodes=1,
        deterministic=True,
        verbose=1
    )

    callbacks = [eval_cb]
    if have_pbar:
        callbacks.append(ProgressBarCallback())

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
    path = os.path.join(RUN_DIR, "td3_motor.zip")
    model.save(path)
    print(f"[RL] Saved model → {path}")

def eval_rl(args):
    """Live streaming plot with autoscale (& slowed playback) or static plots."""
    try:
        from stable_baselines3 import TD3
    except Exception:
        print("Stable-Baselines3 not available. Install: pip install stable-baselines3 torch gymnasium")
        return

    if not args.model_path:
        print("--model_path missing.")
        return
    mpath = os.path.expanduser(args.model_path)
    if not mpath.endswith(".zip"): mpath += ".zip"
    if not os.path.exists(mpath):
        raise FileNotFoundError(f"Model file not found: {mpath}")

    env = make_env(args)
    model = TD3.load(mpath, device="auto")

    obs, _ = env.reset(seed=args.seed)
    steps = int(args.t_end / (args.dt * args.frame_skip))

    ts, us, is_, ws, wrefs, Tems, Tls = [], [], [], [], [], [], []

    if args.live:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, axs = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
        (l_u,)   = axs[0].plot([], [], lw=1.2, label="u")
        axs[0].set_ylabel("u [V]"); axs[0].legend(); axs[0].grid(True)
        (l_i,)   = axs[1].plot([], [], lw=1.2, label="i")
        (l_iref,) = axs[1].plot([], [], "--", lw=1.0, label="i_ref (n/a)")
        axs[1].set_ylabel("i [A]"); axs[1].legend(); axs[1].grid(True)
        (l_w,)   = axs[2].plot([], [], lw=1.2, label="ω")
        (l_wref,) = axs[2].plot([], [], "--", lw=1.0, label="ω_ref")
        axs[2].set_ylabel("ω [rad/s]"); axs[2].legend(); axs[2].grid(True)
        (l_tem,) = axs[3].plot([], [], lw=1.2, label="T_em")
        (l_tl,)  = axs[3].plot([], [], "--", lw=1.0, label="T_load")
        axs[3].set_ylabel("Torque [Nm]"); axs[3].legend(); axs[3].grid(True)
        axs[3].set_xlabel("t [s]")
        fig.suptitle(f"TD3 Evaluation — task={args.task} | ref={args.ref_profile}")
        wall_dt = (env.frame_skip * env.dt) / max(args.vis_speed, 1e-6)

    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)

        ts.append(env.t); us.append(env.prev_u); is_.append(env.state.i); ws.append(env.state.omega)
        wrefs.append(speed_ref_profile(env.t, args.ref_profile, args.w_ref, env.rand_seq))
        Tems.append(env.p.kT * env.state.i); Tls.append(env.base_load)

        if args.live:
            import matplotlib.pyplot as plt
            l_u.set_data(ts, us)
            l_i.set_data(ts, is_)
            l_iref.set_data(ts, [0.0]*len(ts))
            l_w.set_data(ts, ws)
            l_wref.set_data(ts, wrefs)
            l_tem.set_data(ts, Tems)
            l_tl.set_data(ts, Tls)
            for ax in axs:
                ax.relim(); ax.autoscale_view(True, True, True)
            plt.pause(0.001)
            if wall_dt > 0: time.sleep(wall_dt)
        if term or trunc: break

    if args.live:
        import matplotlib.pyplot as plt
        fig.canvas.draw(); fig.canvas.flush_events()
        fig.savefig(os.path.join(RUN_DIR, "eval_live_snapshot.png"), dpi=200)
        plt.ioff(); plt.show()
        print(f"[EVAL] Live snapshot saved to {RUN_DIR}/eval_live_snapshot.png")
    else:
        data = {
            "t": np.array(ts),
            "u": np.array(us),
            "i": np.array(is_),
            "omega": np.array(ws),
            "omega_ref": np.array(wrefs),
            "i_ref": np.zeros_like(wrefs),
            "T_em": np.array(Tems),
            "T_load": np.array(Tls)
        }
        save_plots(data, f"TD3 Evaluation — task={args.task} | ref={args.ref_profile}", RUN_DIR, live=False)
        print(f"[EVAL] Plots saved under {RUN_DIR}")

# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["a","b"], default="b", help="Curriculum: a (easy) or b (robust)")
    parser.add_argument("--task", choices=["speed"], default="speed")
    parser.add_argument("--ref_profile", choices=["const","step_seq_motor","rand_steps"], default="rand_steps")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval_rl", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None, help="Resume training from saved .zip")
    parser.add_argument("--t_end", type=float, default=8.0)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--frame_skip", type=int, default=20)
    parser.add_argument("--ep_len_s", type=float, default=8.0)
    parser.add_argument("--total_timesteps", type=int, default=300_000)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--vis_speed", type=float, default=1.0, help="1.0=realtime, 0.5=2x slower")
    parser.add_argument("--w_ref", type=float, default=160.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau_act", type=float, default=1e-3, help="Actuator time constant [s]")
    # reward weights
    parser.add_argument("--rw_e", type=float, default=4.0)
    parser.add_argument("--rw_u", type=float, default=0.01)
    parser.add_argument("--rw_de", type=float, default=0.02)
    parser.add_argument("--rw_du", type=float, default=0.005)
    parser.add_argument("--rw_i", type=float, default=0.01)
    # state safety clamps (no voltage cap)
    parser.add_argument("--i_abs_max", type=float, default=6.0)
    parser.add_argument("--omega_abs_max", type=float, default=900.0)
    args = parser.parse_args()

    _choose_backend(headless=not args.live)

    if args.train:   train_rl(args)
    if args.eval_rl: eval_rl(args)

if __name__ == "__main__":
    main()