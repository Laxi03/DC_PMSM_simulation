"""PMSM speed-tracking environment from the TD3 study."""
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
# REFERENCE
# ==========================================================

def generate_speed_profile(rng: np.random.Generator):
    n_segments = rng.integers(6, 12)

    speeds = rng.uniform(-350.0, 350.0, n_segments)
    durations = rng.uniform(0.4, 1.2, n_segments)

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
# TRAIN-TIME RANDOMIZATION / EVAL PERTURBATION
# ==========================================================

@dataclass
class TrainRandomization:
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
    jump_time_max: float = 6.0
    jump_target: str = "R"      # R | Ld | Lq | lam | Vdc
    jump_range: tuple = (0.8, 1.35)


@dataclass
class EvalPerturbation:
    enabled: bool = False
    target: str = "R"           # R | Ld | Lq | lam | Vdc
    rel_change: float = 0.3
    random_time: bool = True
    t_start: float = 2.0
    t_min: float = 1.0
    t_max: float = 6.0


# ==========================================================
# SEGMENT METRICS
# ==========================================================

def find_reference_segments(t, ref, tol=1e-9):
    segments = []

    if len(ref) < 2:
        return segments

    start_idx = 0

    for k in range(1, len(ref)):
        if abs(ref[k] - ref[k - 1]) > tol:
            end_idx = k - 1
            if end_idx > start_idx:
                segments.append((start_idx, end_idx))
            start_idx = k

    if start_idx < len(ref) - 1:
        segments.append((start_idx, len(ref) - 1))

    return segments


def compute_rise_time_speed(t_seg, w_seg, w0, w1):
    delta = w1 - w0

    if abs(delta) < 1e-12:
        return np.nan

    y10 = w0 + 0.1 * delta
    y90 = w0 + 0.9 * delta

    idx10 = None
    idx90 = None

    if delta > 0:
        for i in range(len(w_seg)):
            if idx10 is None and w_seg[i] >= y10:
                idx10 = i
            if idx90 is None and w_seg[i] >= y90:
                idx90 = i
                break
    else:
        for i in range(len(w_seg)):
            if idx10 is None and w_seg[i] <= y10:
                idx10 = i
            if idx90 is None and w_seg[i] <= y90:
                idx90 = i
                break

    if idx10 is None or idx90 is None:
        return np.nan

    return t_seg[idx90] - t_seg[idx10]


def compute_overshoot_speed(w_seg, w0, w1):
    delta = w1 - w0

    if abs(delta) < 1e-12:
        return 0.0

    if delta > 0:
        peak = np.max(w_seg)
        mp = max(0.0, (peak - w1) / abs(delta) * 100.0)
    else:
        valley = np.min(w_seg)
        mp = max(0.0, (w1 - valley) / abs(delta) * 100.0)

    return mp


def compute_segment_metrics_speed(t, w, wref, vd, vq, Vmax):
    rows = []

    segments = find_reference_segments(t, wref)

    for seg_id, (i0, i1) in enumerate(segments, start=1):
        if i1 - i0 < 3:
            continue

        t_seg = t[i0:i1 + 1]
        w_seg = w[i0:i1 + 1]
        wref_seg = wref[i0:i1 + 1]
        vd_seg = vd[i0:i1 + 1]
        vq_seg = vq[i0:i1 + 1]

        w1 = wref_seg[0]
        w0 = wref[i0 - 1] if i0 > 0 else w[0]

        e_seg = wref_seg - w_seg
        dt_seg = np.diff(t_seg)

        tr = compute_rise_time_speed(t_seg, w_seg, w0, w1)
        mp = compute_overshoot_speed(w_seg, w0, w1)

        sse = np.mean(e_seg[-max(1, len(e_seg) // 5):])
        iae = np.sum(np.abs(e_seg[:-1]) * dt_seg)
        ise = np.sum((e_seg[:-1] ** 2) * dt_seg)

        vmag = np.sqrt(vd_seg ** 2 + vq_seg ** 2)
        rms_u = np.sqrt(np.mean(vmag ** 2))
        time_at_rail = np.sum((vmag[:-1] >= 0.98 * Vmax) * dt_seg)

        rows.append({
            "segment": f"{seg_id}: {t_seg[0]:.2f}-{t_seg[-1]:.2f}",
            "ref": w1,
            "tr": tr,
            "Mp": mp,
            "SSE": sse,
            "IAE": iae,
            "ISE": ise,
            "RMSu": rms_u,
            "TimeRail": time_at_rail
        })

    return rows


def print_segment_metrics_table_speed(rows):
    if not rows:
        print("\nNincsenek kiírható szakaszmutatók.\n")
        return

    print("\nSzakaszonkénti mutatók (PMSM RL speed)\n")

    header = (
        f"{'Szegmens':>14} | "
        f"{'w_ref [rad/s]':>13} | "
        f"{'t_r [s]':>9} | "
        f"{'M_p [%]':>9} | "
        f"{'SSE [rad/s]':>12} | "
        f"{'IAE':>10} | "
        f"{'ISE':>12} | "
        f"{'RMS(u_dq) [V]':>14} | "
        f"{'Time@rail [s]':>14}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        tr_str = f"{r['tr']:.4f}" if np.isfinite(r["tr"]) else "nan"

        print(
            f"{r['segment']:>14} | "
            f"{r['ref']:>13.3f} | "
            f"{tr_str:>9} | "
            f"{r['Mp']:>9.3f} | "
            f"{r['SSE']:>12.4f} | "
            f"{r['IAE']:>10.4f} | "
            f"{r['ISE']:>12.4f} | "
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
        f"{'ÁTLAG':>14} | "
        f"{avg_ref:>13.3f} | "
        f"{avg_tr_str:>9} | "
        f"{avg_mp:>9.3f} | "
        f"{avg_sse:>12.4f} | "
        f"{avg_iae:>10.4f} | "
        f"{avg_ise:>12.4f} | "
        f"{avg_rmsu:>14.4f} | "
        f"{avg_timerail:>14.4f}"
    )
    print()


# ==========================================================
# MODEL
# ==========================================================

def electrical_speed(motor: PMSM, w_mech: float) -> float:
    return motor.pole_pairs * w_mech


def derivatives(motor, R, Ld, Lq, lam, x, vd, vq, Tload):
    p = motor.pole_pairs
    we = electrical_speed(motor, x.w)

    did = (vd - R * x.id + we * Lq * x.iq) / Ld
    diq = (vq - R * x.iq - we * (Ld * x.id + lam)) / Lq

    Te = 1.5 * p * (lam * x.iq + (Ld - Lq) * x.id * x.iq)
    dw = (Te - Tload - motor.B * x.w) / motor.J
    dth = we

    return State(did, diq, dw, dth)


def rk4(motor, R, Ld, Lq, lam, x, dt, vd, vq, Tload):
    def f(s):
        return derivatives(motor, R, Ld, Lq, lam, s, vd, vq, Tload)

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
# RL ENVIRONMENT
# ==========================================================

class PMSMSpeedEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        training_mode=False,
        train_randomization=None,
        eval_perturbation=None,
        seed=None,
        Tload=0.002,
        Kp_w_pi=0.03,
        Ki_w_pi=0.03,
        base_speed=500.0,
    ):
        super().__init__()

        self.training_mode = training_mode
        self.train_randomization = (
            train_randomization if train_randomization is not None else TrainRandomization()
        )
        self.eval_perturbation = (
            eval_perturbation if eval_perturbation is not None else EvalPerturbation()
        )

        self.dt = 5e-5
        self.frame_skip = 20
        self.episode_length = 8.0

        self.motor = PMSM()
        self.R_nom, self.Ld_nom, self.Lq_nom, self.lam_nom = self.motor.derived()
        self.Vdc_nom = self.motor.Vdc

        self.R = self.R_nom
        self.Ld = self.Ld_nom
        self.Lq = self.Lq_nom
        self.lam = self.lam_nom
        self.Vdc = self.Vdc_nom

        self.state = State(0.0, 0.0, 0.0, 0.0)

        self.wmax = 700.0
        self.Imax = 2.0
        self.Tload = Tload
        self.base_speed = base_speed

        self.Kp_w_pi = Kp_w_pi
        self.Ki_w_pi = Ki_w_pi

        self.t = 0.0

        self.vd = 0.0
        self.vq = 0.0
        self.prev_vd = 0.0
        self.prev_vq = 0.0

        self.error_integral = 0.0
        self.prev_error_w = 0.0

        # internal PI speed-loop generated current references
        self.int_w_pi = 0.0
        self.last_id_ref_pi = 0.0
        self.last_iq_ref_pi = 0.0

        self.profile_speeds = None
        self.profile_times = None
        self.last_wref = 0.0

        self.perturb_time = None
        self.perturb_applied = False

        self.train_jump_enabled_this_ep = False
        self.train_jump_time = None
        self.train_jump_factor = 1.0
        self.train_jump_applied = False
        self.train_jump_target = "R"

        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

        # [id, iq, w, w_ref, e_w, prev_e_w, int_e_w, id_ref_pi, iq_ref_pi, vd, vq]
        self.observation_space = spaces.Box(-1.0, 1.0, (11,), dtype=np.float32)

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

    def _compute_internal_current_references(self, wref):
        ew = wref - self.state.w
        iq_unsat = self.Kp_w_pi * ew + self.Ki_w_pi * self.int_w_pi

        iq_dyn = self.Vmax / max(self.lam * self.motor.pole_pairs, 1e-6)
        iq_ref = float(np.clip(iq_unsat, -min(self.Imax, iq_dyn), min(self.Imax, iq_dyn)))

        aw_gain = 5.0
        self.int_w_pi += (ew + aw_gain * (iq_ref - iq_unsat)) * self.dt
        self.int_w_pi = float(np.clip(self.int_w_pi, -5.0, 5.0))

        id_ref = -0.3 if abs(self.state.w) > self.base_speed else 0.0

        self.last_iq_ref_pi = iq_ref
        self.last_id_ref_pi = id_ref

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.motor = PMSM()
        self.R_nom, self.Ld_nom, self.Lq_nom, self.lam_nom = self.motor.derived()

        self.R = self.R_nom
        self.Ld = self.Ld_nom
        self.Lq = self.Lq_nom
        self.lam = self.lam_nom
        self.Vdc = self.Vdc_nom

        self._setup_training_randomization()

        self.state = State(0.0, 0.0, 0.0, 0.0)

        self.vd = 0.0
        self.vq = 0.0
        self.prev_vd = 0.0
        self.prev_vq = 0.0

        self.error_integral = 0.0
        self.prev_error_w = 0.0

        self.int_w_pi = 0.0
        self.last_id_ref_pi = 0.0
        self.last_iq_ref_pi = 0.0

        self.t = 0.0

        self.profile_speeds, self.profile_times = generate_speed_profile(self.rng)
        self.last_wref = speed_ref(0.0, self.profile_speeds, self.profile_times)
        self._compute_internal_current_references(self.last_wref)

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
        wref = speed_ref(self.t, self.profile_speeds, self.profile_times)
        e_w = (wref - self.state.w) / self.wmax

        return np.array([
            self.state.id / self.Imax,
            self.state.iq / self.Imax,
            self.state.w / self.wmax,
            wref / self.wmax,
            e_w,
            np.clip(self.prev_error_w, -1.0, 1.0),
            np.clip(self.error_integral, -1.0, 1.0),
            self.last_id_ref_pi / self.Imax,
            self.last_iq_ref_pi / self.Imax,
            self.vd / max(self.Vmax, 1e-12),
            self.vq / max(self.Vmax, 1e-12),
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

            alpha = self.dt / 8e-4
            alpha = float(np.clip(alpha, 0.0, 1.0))

            self.vd += alpha * (vd_cmd - self.vd)
            self.vq += alpha * (vq_cmd - self.vq)

            self.vd = float(np.clip(self.vd, -self.Vmax, self.Vmax))
            self.vq = float(np.clip(self.vq, -self.Vmax, self.Vmax))

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
                self.Tload
            )

            self.t += self.dt

            wref = speed_ref(self.t, self.profile_speeds, self.profile_times)
            self.last_wref = wref

            self._compute_internal_current_references(wref)

            e_w = (wref - self.state.w) / self.wmax
            e_d = (0.0 - self.state.id) / self.Imax
            e_iq_ref = (self.last_iq_ref_pi - self.state.iq) / self.Imax
            e_id_ref = (self.last_id_ref_pi - self.state.id) / self.Imax

            self.error_integral += e_w * self.dt
            self.error_integral = float(np.clip(self.error_integral, -1.0, 1.0))

            # Main objective: speed tracking
            tracking_w = -10.0 * abs(e_w)
            large_error_pen = -12.0 * (e_w ** 2)

            # Keep id small
            tracking_d = -2.0 * abs(e_d)

            # Mild shaping toward internal PI current references
            current_ref_tracking = -1.5 * abs(e_iq_ref) - 0.8 * abs(e_id_ref)

            # Small integral penalty
            integral_pen = -0.05 * (self.error_integral ** 2)

            # Softer control penalties
            voltage_pen = -0.003 * (
                (self.vd / max(self.Vmax, 1e-12)) ** 2 +
                (self.vq / max(self.Vmax, 1e-12)) ** 2
            )

            current_pen = -0.01 * (
                (self.state.id / self.Imax) ** 2 +
                (self.state.iq / self.Imax) ** 2
            )

            du_d = (self.vd - self.prev_vd) / max(self.Vmax, 1e-12)
            du_q = (self.vq - self.prev_vq) / max(self.Vmax, 1e-12)
            smooth_pen = -0.01 * (du_d ** 2 + du_q ** 2)

            settle_bonus = 0.08 if abs(e_w) < 0.02 else 0.0
            tight_bonus = 0.05 if (abs(e_w) < 0.01 and abs(e_d) < 0.02) else 0.0

            reward_step_raw = (
                tracking_w
                + large_error_pen
                + tracking_d
                + current_ref_tracking
                + integral_pen
                + voltage_pen
                + current_pen
                + smooth_pen
                + settle_bonus
                + tight_bonus
            )

            reward_step = float(np.clip(reward_step_raw, -1.0, 1.0))
            reward += reward_step

            self.prev_error_w = e_w
            self.prev_vd = self.vd
            self.prev_vq = self.vq

            last_terms = {
                "reward_step": reward_step,
                "tracking_w": tracking_w,
                "large_error_pen": large_error_pen,
                "tracking_d": tracking_d,
                "current_ref_tracking": current_ref_tracking,
                "integral_pen": integral_pen,
                "voltage_pen": voltage_pen,
                "current_pen": current_pen,
                "smooth_pen": smooth_pen,
                "settle_bonus": settle_bonus,
                "tight_bonus": tight_bonus,
                "error_w": e_w,
                "error_d": e_d,
                "id_ref_pi": self.last_id_ref_pi,
                "iq_ref_pi": self.last_iq_ref_pi,
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


def make_env(base_seed=None, rank=0, Tload=0.002, Kp_w_pi=0.03, Ki_w_pi=0.03, base_speed=500.0):
    def _init():
        env_seed = None if base_seed is None else int(base_seed + rank)
        return PMSMSpeedEnv(
            training_mode=True,
            train_randomization=TrainRandomization(
                enabled=True,
                randomize_R=True,
                R_range=(0.85, 1.15),
                randomize_Ld=False,
                randomize_Lq=False,
                jump_enabled=True,
                jump_prob=0.7,
                jump_time_min=1.0,
                jump_time_max=6.0,
                jump_target="R",
                jump_range=(0.8, 1.35),
            ),
            eval_perturbation=EvalPerturbation(enabled=False),
            seed=env_seed,
            Tload=Tload,
            Kp_w_pi=Kp_w_pi,
            Ki_w_pi=Ki_w_pi,
            base_speed=base_speed,
        )
    return _init
