#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
motor_rl.py — DC motor RL control (TD3, Gymnasium + Stable-Baselines3)

Includes:
- warmup_zero_s: first seconds of each episode track ω_ref=0 (train & eval)
- HOLD latch with hysteresis (band_in / band_out) to discourage dithering
- Anti-chatter: |Δu| penalty + u^2 penalty near target
- Positive band bonus when close to target (stronger when latched)
- Hard duty rate limit max_du (+ tighter max_du_hold while latched)
- Actuator low-pass
- Domain randomization curriculum (phase=a/b)
- TimeLimit wrapper
- Live plotting during evaluation (--live)
- Reward saturation (configurable) to keep per-step rewards bounded
- AFTER TRAINING: plot episodic rewards and save training parameters to RUN_DIR
- LOW-SPEED HELPERS:
    * e_int (leaky integral of error) in observation
    * reference-aware u^2 penalty (weaker when |ω_ref| small)
    * low-speed duty “floor” encouragement when latched
"""

from typing import Tuple, Optional, List
from dataclasses import dataclass
import os, math, argparse, warnings, time, csv, json
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
                          min_val=-676.46, max_val=676.46,
                          min_seg=0.8, max_seg=1.5,
                          skew_high_prob=0.5, high_lo=300.0) -> Tuple[np.ndarray, np.ndarray]:
    """Random step sequence."""
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
        if t < 0.5: return 0.0          # warm-up zero
        if t < 1.0: return 100.0
        if t < 2.0: return 325.0
        if t < 3.0: return 50.0
        if t < 4.0: return -150.0
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
    from gymnasium.wrappers import TimeLimit
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
                 frame_skip=20,
                 Vdc=24.0,
                 ulim=1.0,
                 ref_kind_speed="rand_steps",
                 w_ref=160.0,
                 # reward weights (normalized units)
                 rw_e=4.0, rw_u=0.01, rw_de=0.02, rw_du=0.005, rw_i=0.01,
                 ep_len_s=8.0,
                 # phase profile for domain rand / noise
                 phase="b",
                 # state safety
                 i_abs_max=6.0,
                 omega_abs_max=900.0,
                 # noise
                 obs_noise=0.000,
                 load_noise=0.00,
                 # disturbances
                 load_step_prob=0.4,
                 base_load=0.02,
                 # actuator lag
                 tau_act=1e-3,
                 # warm-up / HOLD shaping
                 warmup_zero_s=0.5,
                 e_small=0.04,
                 hold_w=0.003,
                 hold_u_w=0.01,
                 # duty rate limits
                 max_du=0.02,
                 max_du_hold=0.005,
                 # band bonus + hysteresis
                 r_band=0.02,
                 r_band_latched=0.035,
                 band_in=0.03,
                 band_out=0.05,
                 # reward saturation (per-step)
                 reward_lo=-1.0,
                 reward_hi=0.3,
                 # LOW-SPEED helpers
                 eint_tau=0.30,
                 w_low_radps=120.0,
                 u_min_hold_low=0.05,
                 k_under_u_low=0.02,
                 **_):
        assert task == "speed"
        self.task, self.dt, self.frame_skip = task, dt, frame_skip
        self.Vdc_nom = Vdc
        self.ulim = ulim
        self.ref_kind_speed, self.w_ref_base = ref_kind_speed, w_ref

        # reward weights
        self.rw_e, self.rw_u, self.rw_de, self.rw_du, self.rw_i = rw_e, rw_u, rw_de, rw_du, rw_i
        self.ep_len_s = float(ep_len_s)
        self.max_rl_steps = max(1, int(self.ep_len_s / (self.dt * self.frame_skip)))

        # curriculum / domain randomization
        self.phase = phase
        self.base_load = base_load
        self.load_step_prob = load_step_prob
        self.load_noise = float(load_noise)
        self.obs_noise = float(obs_noise)

        # state soft-clamps
        self.i_abs_max = float(i_abs_max)
        self.omega_abs_max = float(omega_abs_max)
        self.i_env_max = 1.2 * self.i_abs_max
        self.omega_env_max = 1.2 * self.omega_abs_max

        # actuator
        self.tau_act = float(tau_act)
        self.u_applied = 0.0

        # warm-up + HOLD + rate limit
        self.warmup_zero_s = float(warmup_zero_s)
        self.e_small = float(e_small)
        self.hold_w = float(hold_w)
        self.hold_u_w = float(hold_u_w)
        self.max_du = float(max_du)
        self.max_du_hold = float(max_du_hold)

        # band & hysteresis
        self.r_band = float(r_band)
        self.r_band_latched = float(r_band_latched)
        self.band_in = float(band_in)
        self.band_out = float(band_out)
        self.hold_latched = False

        # reward saturation
        self.reward_lo = float(reward_lo)
        self.reward_hi = float(reward_hi)

        # LOW-SPEED helpers
        self.eint_tau = float(eint_tau)
        self.w_low_radps = float(w_low_radps)
        self.u_min_hold_low = float(u_min_hold_low)  # in duty fraction (0..1)
        self.k_under_u_low = float(k_under_u_low)

        # internals
        self.p = AMax32Params()
        self.motor = AMax32Motor(self.p)
        self.state = MotorState(0.0, 0.0, 0.0, self.p.T_amb)
        self.t = 0.0
        self.prev_u = 0.0
        self.prev_e_norm = 0.0
        self.prev_duty = 0.0
        self.rl_step_count = 0
        self.e_int = 0.0  # leaky integral of normalized error

        # observation/action spaces (7 dims with e_int)
        self.rand_seq: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        obs_high = np.ones(7, np.float32)  # [omega_n, i_n, u_n, e_norm, headroom, sin(t), e_int]
        self.observation_space = spaces.Box(-obs_high, high=obs_high, dtype=np.float32)

        # runtime-randomized parameters (set in reset)
        self.Vdc = self.Vdc_nom
        self.load_torque = self.base_load
        self.load_bump = 0.0

    # ------------- helpers
    def _current_wref(self) -> float:
        if self.t < self.warmup_zero_s:
            return 0.0
        return speed_ref_profile(self.t, self.ref_kind_speed, self.w_ref_base, self.rand_seq)

    def _finite(self) -> bool:
        s = self.state
        return np.isfinite(s.i) and np.isfinite(s.omega)

    # ------------- Gym API
    def reset(self, *, seed=None, options=None):
        self.hold_latched = False
        self.e_int = 0.0
        if seed is not None:
            np.random.seed(seed)

        # PHASE domain randomization
        scale = lambda x, pct: float(x * np.random.uniform(1 - pct, 1 + pct))
        if self.phase.lower() == "a":
            self.p.R = scale(self.p.R, 0.10); self.p.L = scale(self.p.L, 0.10)
            self.p.J = scale(self.p.J, 0.10); self.p.B = scale(self.p.B, 0.15)
            self.p.kT = scale(self.p.kT, 0.08); self.p.Ke = scale(self.p.Ke, 0.08)
            self.Vdc  = scale(self.Vdc_nom, 0.05)
            self.base_load = scale(self.base_load, 0.20)
            self.load_noise = 0.06; self.obs_noise = 0.002
            self.rand_seq = generate_random_steps(self.ep_len_s, -500.0, 500.0, 0.4, 1.1, skew_high_prob=0.4) \
                            if self.ref_kind_speed == "rand_steps" else None
        else:
            self.p.R = scale(self.p.R, 0.20); self.p.L = scale(self.p.L, 0.20)
            self.p.J = scale(self.p.J, 0.25); self.p.B = scale(self.p.B, 0.30)
            self.p.kT = scale(self.p.kT, 0.15); self.p.Ke = scale(self.p.Ke, 0.15)
            self.Vdc  = scale(self.Vdc_nom, 0.10)
            self.base_load = scale(self.base_load, 0.35)
            self.load_noise = 0.10; self.obs_noise = 0.003
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
        self.state.i = float(np.clip(self.state.i, -1.2*self.i_abs_max, 1.2*self.i_abs_max))
        self.state.omega = float(np.clip(self.state.omega, -1.2*self.omega_abs_max, 1.2*self.omega_abs_max))

        omega_n = float(np.clip(self.state.omega / 900.0, -1.0, 1.0))
        i_n     = float(np.clip(self.state.i     / 6.0,   -1.0, 1.0))
        u_n     = float(np.clip(self.prev_u      / self.Vdc, -1.0, 1.0))
        e_norm  = float(np.clip((self._current_wref() - self.state.omega) / 900.0, -1.0, 1.0))
        headroom = float(np.clip((self.Vdc - abs(self.motor.p.Ke * self.state.omega)) / max(self.Vdc, 1e-6), 0.0, 1.0))
        s = float(np.sin(self.t))
        obs = np.array([omega_n, i_n, u_n, e_norm, headroom, s, float(np.clip(self.e_int, -1.0, 1.0))], dtype=np.float32)
        if self.obs_noise > 0.0:
            obs = np.clip(obs + np.random.normal(0.0, self.obs_noise, size=obs.shape).astype(np.float32), -1.0, 1.0)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return obs

    def step(self, action):
        # guard numerical issues
        if not self._finite():
            obs, _ = self.reset()
            return obs, 0.0, False, True, {}

        # raw agent duty within [-ulim, ulim]
        raw_duty = float(np.clip(action[0], -self.ulim, self.ulim))

        # hysteresis latch on error band (based on previous e_norm)
        abs_e = abs(self.prev_e_norm)
        if not self.hold_latched and abs_e < self.band_in:
            self.hold_latched = True
        elif self.hold_latched and abs_e > self.band_out:
            self.hold_latched = False

        # duty rate limit (tighter when latched)
        du_limit = self.max_du_hold if self.hold_latched else self.max_du
        duty = float(np.clip(raw_duty, self.prev_duty - du_limit, self.prev_duty + du_limit))

        u_cmd = self.Vdc * duty
        rew_sum = 0.0

        for _ in range(self.frame_skip):
            # actuator low-pass: u_applied <- u_cmd
            alpha_lp = self.dt / max(self.tau_act, 1e-6)
            self.u_applied += alpha_lp * (u_cmd - self.u_applied)
            u_eff = self.u_applied

            # load: base + noise + optional bump
            Tl = self.base_load * (1.0 + np.random.uniform(-self.load_noise, self.load_noise))
            if self.load_bump and self.t >= self.load_bump[0]:
                Tl += self.load_bump[1]

            # integrate plant
            self.state = AMax32Motor.rk4_step(
                lambda tau, s: self.motor.derivatives(tau, s, u_eff, Tl),
                self.t, self.dt, self.state
            )
            self.t += self.dt

            if not self._finite():
                obs, _ = self.reset()
                return obs, self.reward_lo, False, True, {}

            # normalized vars
            wref   = self._current_wref()
            e_norm = float(np.clip((wref - self.state.omega) / 900.0, -1.0, 1.0))
            u_norm = float(np.clip(u_eff / self.Vdc, -1.0, 1.0))
            du_norm = duty - self.prev_duty

            # --- leaky integral of error (normalized)
            if self.eint_tau > 1e-6:
                alpha = self.dt / self.eint_tau
                self.e_int = (1.0 - alpha)*self.e_int + alpha*e_norm
                self.e_int = float(np.clip(self.e_int, -1.0, 1.0))

            # -------- REWARD (with saturation) --------
            r = - ( self.rw_e * huber(e_norm, k=0.05) )

            # reference-aware u^2 penalty scaling: weak at tiny |ω_ref|
            ref_mag = float(np.clip(abs(wref) / max(self.w_low_radps, 1e-6), 0.0, 1.0))
            hold_u_w_eff = self.hold_u_w * (0.2 + 0.8*ref_mag)

            if abs(e_norm) < self.e_small:
                # bonus for being in band (stronger when latched)
                r += (self.r_band_latched if self.hold_latched else self.r_band)
                factor = (2.0 if self.hold_latched else 1.0)
                r -= factor*self.hold_w     * abs(du_norm)
                r -= factor*hold_u_w_eff    * (u_norm**2)

                # low-speed “push” to overcome stiction when latched
                if abs(wref) > 1e-6 and abs(wref) <= self.w_low_radps and self.hold_latched:
                    shortfall = max(0.0, self.u_min_hold_low - abs(duty))
                    r -= self.k_under_u_low * shortfall

            # soft-safety nudges near clamps
            if abs(self.state.i) > 0.9 * self.i_abs_max:          r -= 0.05
            if abs(self.state.omega) > 0.9 * self.omega_abs_max:   r -= 0.05

            # per-step reward saturation
            r = float(np.clip(r, self.reward_lo, self.reward_hi))
            rew_sum += r

            # update prev signals at fast timescale
            self.prev_e_norm = e_norm
            self.prev_u = float(u_eff)
            self.prev_duty = duty

        # mild terminal encouragement if ending close
        if (self.rl_step_count + 1) >= self.max_rl_steps and abs(self.prev_e_norm) < 0.03:
            rew_sum += min(0.2, self.reward_hi)

        # clip whole-step sum (numerical guard)
        rew_sum = float(np.clip(np.nan_to_num(rew_sum, nan=0.0, posinf=0.0, neginf=self.reward_lo),
                                10*self.reward_lo, 10*self.reward_hi))

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
    axs[0].plot(t,data["u"], label="u"); axs[0].set_ylabel("u [V]"); axs[0].legend(); axs[0].grid(True)
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

# ---------- Training rewards plotting ----------
def _read_monitor_rewards(monitor_csv: str) -> List[float]:
    rewards: List[float] = []
    if not os.path.exists(monitor_csv):
        return rewards
    with open(monitor_csv, "r", newline="") as f:
        reader = csv.DictReader((row for row in f if not row.startswith("#")))
        for row in reader:
            try:
                rewards.append(float(row["r"]))
            except Exception:
                pass
    return rewards

def plot_training_rewards(monitor_csv: str, run_dir: str, ma_window: int = 50):
    import matplotlib.pyplot as plt
    rewards = _read_monitor_rewards(monitor_csv)
    if not rewards:
        print(f"[PLOT] No rewards found in {monitor_csv}")
        return
    ep = np.arange(1, len(rewards)+1)
    ma = np.convolve(rewards, np.ones(ma_window)/ma_window, mode="valid") if len(rewards) >= ma_window else None

    plt.figure(figsize=(10,5))
    plt.plot(ep, rewards, linewidth=1.0, label="Episode reward")
    if ma is not None:
        plt.plot(np.arange(ma_window, len(rewards)+1), ma, linewidth=2.0, label=f"MA({ma_window})")
    plt.grid(True); plt.xlabel("Episode"); plt.ylabel("Reward"); plt.title("Training rewards per episode"); plt.legend()
    out_png = os.path.join(run_dir, "rewards_per_episode.png")
    out_pdf = os.path.join(run_dir, "rewards_per_episode.pdf")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.savefig(out_pdf); plt.close()
    print(f"[PLOT] Saved episodic reward curves → {out_png} / .pdf")

def save_training_params(args, run_dir: str, model_path: str):
    d = vars(args).copy()
    d["run_dir"] = run_dir
    d["model_path"] = model_path
    txt = os.path.join(run_dir, "train_params.txt")
    jsn = os.path.join(run_dir, "train_params.json")
    with open(txt, "w") as f:
        f.write("=== Training Parameters ===\n")
        for k in sorted(d.keys()):
            f.write(f"{k}: {d[k]}\n")
    with open(jsn, "w") as f:
        json.dump(d, f, indent=2)
    print(f"[INFO] Saved training parameters → {txt} / train_params.json")

# =========================
# RL (TD3) — Train / Evaluate
# =========================
def make_env(args):
    env = MotorEnv(
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
        warmup_zero_s=args.warmup_zero_s,
        e_small=args.e_small,
        hold_w=args.hold_w,
        hold_u_w=args.hold_u_w,
        max_du=args.max_du,
        max_du_hold=args.max_du_hold,
        r_band=args.r_band,
        r_band_latched=args.r_band_latched,
        band_in=args.band_in,
        band_out=args.band_out,
        reward_lo=args.reward_lo,
        reward_hi=args.reward_hi,
        # low-speed helpers
        eint_tau=args.eint_tau,
        w_low_radps=args.w_low_radps,
        u_min_hold_low=args.u_min_hold_low,
        k_under_u_low=args.k_under_u_low,
    )
    try:
        from gymnasium.wrappers import TimeLimit
        return TimeLimit(env, max_episode_steps=env.max_rl_steps)
    except Exception:
        return env

def train_rl(args):
    try:
        from stable_baselines3 import TD3
        from stable_baselines3.common.noise import NormalActionNoise
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback
        have_pbar = True
    except Exception:
        try:
            from stable_baselines3 import TD3
            from stable_baselines3.common.noise import NormalActionNoise
            from stable_baselines3.common.monitor import Monitor
            from stable_baselines3.common.callbacks import EvalCallback
            have_pbar = False
        except Exception:
            print("Stable-Baselines3 not available. Install: pip install 'stable-baselines3[extra]' torch gymnasium")
            return

    # write monitor.csv inside RUN_DIR so we can plot later
    monitor_path = os.path.join(RUN_DIR, "monitor.csv")
    env = Monitor(make_env(args), filename=monitor_path)
    eval_env = Monitor(make_env(args), filename=None)

    print(f"[TRAIN] Export dir: {RUN_DIR}")
    print(f"[TRAIN] TD3 total_timesteps={args.total_timesteps}  frame_skip={args.frame_skip}  ep_len_s={args.ep_len_s}  phase={args.phase.upper()}  warmup_zero_s={args.warmup_zero_s}")

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
            learning_rate=1e-3,
            buffer_size=300_000,
            batch_size=128,
            learning_starts=10_000,
            train_freq=(1, "step"),
            gradient_steps=1,
            gamma=0.99,
            tau=0.005,
            policy_kwargs=dict(net_arch=[128, 64]),
            action_noise=action_noise,
            policy_delay=2,
            target_policy_noise=0.0,
            target_noise_clip=0.0,
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
    if 'have_pbar' in locals() and have_pbar:
        callbacks.append(ProgressBarCallback())

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
    model_path = os.path.join(RUN_DIR, "td3_motor.zip")
    model.save(model_path)
    print(f"[RL] Saved model → {model_path}")

    # ---- post-training artifacts ----
    plot_training_rewards(monitor_path, RUN_DIR)
    save_training_params(args, RUN_DIR, model_path)

def eval_rl(args):
    """Live plotting evaluation for a trained TD3 model (or static if --live not set)."""
    try:
        from stable_baselines3 import TD3
    except Exception:
        print("Stable-Baselines3 not available. Install: pip install 'stable-baselines3[extra]' torch gymnasium")
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
        for ax in axs: ax.grid(True)
        (l_u,)   = axs[0].plot([], [], lw=1.2, label="u"); axs[0].set_ylabel("u [V]"); axs[0].legend()
        (l_i,)   = axs[1].plot([], [], lw=1.2, label="i")
        (l_iref,) = axs[1].plot([], [], "--", lw=1.0, label="i_ref (n/a)"); axs[1].set_ylabel("i [A]"); axs[1].legend()
        (l_w,)   = axs[2].plot([], [], lw=1.2, label="ω")
        (l_wref,) = axs[2].plot([], [], "--", lw=1.0, label="ω_ref"); axs[2].set_ylabel("ω [rad/s]"); axs[2].legend()
        (l_tem,) = axs[3].plot([], [], lw=1.2, label="T_em")
        (l_tl,)  = axs[3].plot([], [], "--", lw=1.0, label="T_load"); axs[3].set_ylabel("Torque [Nm]"); axs[3].legend()
        axs[3].set_xlabel("t [s]")
        fig.suptitle(f"TD3 Evaluation — task={args.task} | ref={args.ref_profile}")
        wall_dt = (env.unwrapped.frame_skip * env.unwrapped.dt) / max(args.vis_speed, 1e-6)

    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)

        ue = env.unwrapped
        ts.append(ue.t); us.append(ue.prev_u); is_.append(ue.state.i); ws.append(ue.state.omega)
        wrefs.append(speed_ref_profile(ue.t, ue.ref_kind_speed, ue.w_ref_base, ue.rand_seq))
        Tems.append(ue.p.kT * ue.state.i); Tls.append(ue.base_load)

        if args.live:
            import matplotlib.pyplot as plt
            l_u.set_data(ts, us); l_i.set_data(ts, is_); l_iref.set_data(ts, [0.0]*len(ts))
            l_w.set_data(ts, ws); l_wref.set_data(ts, wrefs); l_tem.set_data(ts, Tems); l_tl.set_data(ts, Tls)
            for ax in axs: ax.relim(); ax.autoscale_view(True, True, True)
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
    parser.add_argument("--ep_len_s", type=float, default=5.0)
    parser.add_argument("--total_timesteps", type=int, default=300_000)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--vis_speed", type=float, default=1.0, help="1.0=realtime, 0.5=2x slower")
    parser.add_argument("--w_ref", type=float, default=160.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau_act", type=float, default=3e-3, help="Actuator time constant [s]")
    # reward weights
    parser.add_argument("--rw_e", type=float, default=4.0)
    parser.add_argument("--rw_u", type=float, default=0.01)
    parser.add_argument("--rw_de", type=float, default=0.02)
    parser.add_argument("--rw_du", type=float, default=0.005)
    parser.add_argument("--rw_i", type=float, default=0.01)
    # state safety clamps
    parser.add_argument("--i_abs_max", type=float, default=6.0)
    parser.add_argument("--omega_abs_max", type=float, default=900.0)
    # warmup + HOLD controls
    parser.add_argument("--warmup_zero_s", type=float, default=0.5)
    parser.add_argument("--e_small", type=float, default=0.04)
    parser.add_argument("--hold_w", type=float, default=0.003)
    parser.add_argument("--hold_u_w", type=float, default=0.01)
    parser.add_argument("--max_du", type=float, default=0.02)
    parser.add_argument("--max_du_hold", type=float, default=0.005)
    parser.add_argument("--r_band", type=float, default=0.02)
    parser.add_argument("--r_band_latched", type=float, default=0.035)
    parser.add_argument("--band_in", type=float, default=0.03, help="Latch when |e| < band_in")
    parser.add_argument("--band_out", type=float, default=0.05, help="Unlatch when |e| > band_out")
    # reward saturation
    parser.add_argument("--reward_lo", type=float, default=-1.0, help="per-step reward lower clip")
    parser.add_argument("--reward_hi", type=float, default=0.3,  help="per-step reward upper clip")
    # low-speed helpers
    parser.add_argument("--eint_tau", type=float, default=0.30,
                        help="Leaky error integral time constant [s] (adds e_int to obs).")
    parser.add_argument("--w_low_radps", type=float, default=120.0,
                        help="Below this |ω_ref| treat as low-speed regime.")
    parser.add_argument("--u_min_hold_low", type=float, default=0.05,
                        help="Min |duty| encouraged when latched near low refs.")
    parser.add_argument("--k_under_u_low", type=float, default=0.02,
                        help="Penalty weight for shortfall below low-speed duty floor.")
    args = parser.parse_args()

    _choose_backend(headless=not args.live)

    if args.train:   train_rl(args)
    if args.eval_rl: eval_rl(args)

if __name__ == "__main__":
    main()