#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import argparse
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid")


# ----------------------------------------------------------
# Motor parameters
# ----------------------------------------------------------

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


# ----------------------------------------------------------
# Eval perturbation
# ----------------------------------------------------------

@dataclass
class EvalPerturbation:
    enabled: bool = False
    target: str = "R"          # R | Ld | Lq | lam | Vdc
    rel_change: float = 0.3    # +30% -> 0.3 ; -20% -> -0.2
    random_time: bool = False
    t_start: float = 2.0
    t_min: float = 1.0
    t_max: float = 4.0


# ----------------------------------------------------------
# Random profile generator
# ----------------------------------------------------------

def generate_profile(rng: np.random.Generator, val_min, val_max, dur_min, dur_max):
    n = rng.integers(6, 12)

    values = rng.uniform(val_min, val_max, n)
    durations = rng.uniform(dur_min, dur_max, n)

    for i in range(1, len(values)):
        values[i] = 0.6 * values[i - 1] + 0.4 * values[i]

    times = np.cumsum(durations)
    return values, times


def profile_eval(t, values, times):
    for i in range(len(times)):
        if t < times[i]:
            return values[i]
    return values[-1]


def sinus_speed_profile(t, t_end, amp):
    """
    One-period sinusoidal mechanical speed profile on [0, t_end]:
        w(t) = amp * sin(2*pi*t/t_end)
    """
    if t_end <= 0:
        return 0.0
    return amp * math.sin(2.0 * math.pi * t / t_end)


# ----------------------------------------------------------
# PMSM model
# ----------------------------------------------------------

def derivatives(motor, R, Ld, Lq, lam, x, vd, vq, Tload, mech_mode, prescribed_w=None):
    p = motor.pole_pairs

    if mech_mode == "sinus_speed":
        w_mech = 0.0 if prescribed_w is None else prescribed_w
    else:
        w_mech = x.w

    we = p * w_mech

    did = (vd - R * x.id + we * Lq * x.iq) / Ld
    diq = (vq - R * x.iq - we * (Ld * x.id + lam)) / Lq

    Te = 1.5 * p * (lam * x.iq + (Ld - Lq) * x.id * x.iq)

    if mech_mode == "locked":
        dw = 0.0
    elif mech_mode == "sinus_speed":
        dw = 0.0
    else:
        dw = (Te - Tload - motor.B * x.w) / motor.J

    dth = we

    return State(did, diq, dw, dth)


def rk4(motor, R, Ld, Lq, lam, x, dt, vd, vq, Tload, mech_mode, prescribed_w=None):
    def f(s):
        return derivatives(motor, R, Ld, Lq, lam, s, vd, vq, Tload, mech_mode, prescribed_w)

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

    x_next = State(
        x.id + dt * (k1.id + 2 * k2.id + 2 * k3.id + k4.id) / 6.0,
        x.iq + dt * (k1.iq + 2 * k2.iq + 2 * k3.iq + k4.iq) / 6.0,
        x.w + dt * (k1.w + 2 * k2.w + 2 * k3.w + k4.w) / 6.0,
        x.theta + dt * (k1.theta + 2 * k2.theta + 2 * k3.theta + k4.theta) / 6.0
    )

    if mech_mode == "locked":
        x_next.w = 0.0
    elif mech_mode == "sinus_speed":
        x_next.w = 0.0 if prescribed_w is None else prescribed_w

    return x_next


# ----------------------------------------------------------
# Metrics
# ----------------------------------------------------------

def segment_indices_from_reference(ref, min_change=1e-9):
    change_idx = [0]
    for k in range(1, len(ref)):
        if abs(ref[k] - ref[k - 1]) > min_change:
            change_idx.append(k)
    change_idx.append(len(ref))
    return change_idx


def safe_rms(x):
    x = np.asarray(x)
    if len(x) == 0:
        return np.nan
    return np.sqrt(np.mean(x**2))


def compute_rise_time(t, y, r, tol_low=0.1, tol_high=0.9):
    if len(t) < 2:
        return np.nan

    y0 = y[0]
    dy = r - y0

    if abs(dy) < 1e-9:
        return np.nan

    y10 = y0 + tol_low * dy
    y90 = y0 + tol_high * dy

    i10 = None
    i90 = None

    if dy > 0:
        for i in range(len(y)):
            if i10 is None and y[i] >= y10:
                i10 = i
            if i90 is None and y[i] >= y90:
                i90 = i
                break
    else:
        for i in range(len(y)):
            if i10 is None and y[i] <= y10:
                i10 = i
            if i90 is None and y[i] <= y90:
                i90 = i
                break

    if i10 is None or i90 is None:
        return np.nan

    return t[i90] - t[i10]


def compute_overshoot_percent(y, r):
    if len(y) == 0:
        return np.nan

    y0 = y[0]
    step_amp = r - y0

    if abs(step_amp) < 1e-9:
        return 0.0

    if step_amp > 0:
        peak = np.max(y)
        overshoot = max(0.0, peak - r)
    else:
        valley = np.min(y)
        overshoot = max(0.0, r - valley)

    return 100.0 * overshoot / abs(step_amp)


def compute_sse(y, r, tail_ratio=0.2):
    if len(y) == 0:
        return np.nan
    n_tail = max(1, int(len(y) * tail_ratio))
    return np.mean(y[-n_tail:] - r)


def compute_iae(t, e):
    if len(t) < 2:
        return np.nan
    return np.trapezoid(np.abs(e), t)


def compute_ise(t, e):
    if len(t) < 2:
        return np.nan
    return np.trapezoid(e**2, t)


def compute_time_at_rail(t, u_mag, Vmax, rail_ratio=0.95):
    if len(t) < 2:
        return np.nan
    dt = np.mean(np.diff(t))
    return np.sum(u_mag >= rail_ratio * Vmax) * dt


def print_segment_metrics(data, args):
    t = data["t"]
    vd = data["vd"]
    vq = data["vq"]
    u_mag = np.sqrt(vd**2 + vq**2)
    Vmax = np.max(data["Vdc"]) / math.sqrt(3.0) * 0.95

    if args.mode == "speed":
        ref = data["w_ref"]
        y = data["w"]
        label = "w_ref [rad/s]"
        title = "Szakaszonkénti mutatók (PMSM PI speed)"
    else:
        ref = data["iq_ref"]
        y = data["iq"]
        label = "iq_ref [A]"
        title = "Szakaszonkénti mutatók (PMSM PI current)"

    idx = segment_indices_from_reference(ref)
    rows = []

    for s in range(len(idx) - 1):
        i0 = idx[s]
        i1 = idx[s + 1]

        if i1 - i0 < 5:
            continue

        ts = t[i0:i1] - t[i0]
        ys = y[i0:i1]
        rs = ref[i0:i1]
        us = u_mag[i0:i1]

        r_const = float(np.mean(rs))
        e = ys - r_const

        tr = compute_rise_time(ts, ys, r_const)
        mp = compute_overshoot_percent(ys, r_const)
        sse = compute_sse(ys, r_const)
        iae = compute_iae(ts, e)
        ise = compute_ise(ts, e)
        rms_u = safe_rms(us)
        rail_time = compute_time_at_rail(ts, us, Vmax)

        rows.append({
            "segment": f"{s+1}: {t[i0]:.2f}-{t[i1-1]:.2f}",
            "ref": r_const,
            "tr": tr,
            "mp": mp,
            "sse": sse,
            "iae": iae,
            "ise": ise,
            "rms_u": rms_u,
            "rail_time": rail_time
        })

    print()
    print(title)
    print()
    print(
        f"{'Szegmens':>14} | "
        f"{label:>12} | "
        f"{'t_r [s]':>9} | "
        f"{'M_p [%]':>9} | "
        f"{'SSE':>12} | "
        f"{'IAE':>12} | "
        f"{'ISE':>12} | "
        f"{'RMS(u_dq) [V]':>14} | "
        f"{'Time@rail [s]':>14}"
    )
    print("-" * 130)

    for r in rows:
        tr_str = f"{r['tr']:.4f}" if np.isfinite(r["tr"]) else "nan"
        print(
            f"{r['segment']:>14} | "
            f"{r['ref']:12.3f} | "
            f"{tr_str:>9} | "
            f"{r['mp']:9.3f} | "
            f"{r['sse']:12.4f} | "
            f"{r['iae']:12.4f} | "
            f"{r['ise']:12.4f} | "
            f"{r['rms_u']:14.4f} | "
            f"{r['rail_time']:14.4f}"
        )

    if len(rows) > 0:
        avg_ref = np.mean([r["ref"] for r in rows])
        avg_tr = np.nanmean([r["tr"] for r in rows])
        avg_mp = np.nanmean([r["mp"] for r in rows])
        avg_sse = np.nanmean([r["sse"] for r in rows])
        avg_iae = np.nanmean([r["iae"] for r in rows])
        avg_ise = np.nanmean([r["ise"] for r in rows])
        avg_rms = np.nanmean([r["rms_u"] for r in rows])
        avg_rail = np.nanmean([r["rail_time"] for r in rows])

        print("-" * 130)
        print(
            f"{'ÁTLAG':>14} | "
            f"{avg_ref:12.3f} | "
            f"{avg_tr:9.4f} | "
            f"{avg_mp:9.3f} | "
            f"{avg_sse:12.4f} | "
            f"{avg_iae:12.4f} | "
            f"{avg_ise:12.4f} | "
            f"{avg_rms:14.4f} | "
            f"{avg_rail:14.4f}"
        )

    print()


def print_global_metrics(data, args):
    t = data["t"]
    vd = data["vd"]
    vq = data["vq"]
    u_mag = np.sqrt(vd**2 + vq**2)
    Vmax = np.max(data["Vdc"]) / math.sqrt(3.0) * 0.95

    if args.mode == "speed":
        ref = data["w_ref"]
        y = data["w"]
        err_name = "speed"
        unit = "rad/s"
    else:
        ref = data["iq_ref"]
        y = data["iq"]
        err_name = "current"
        unit = "A"

    e = y - ref

    print(f"Mean abs {err_name} error : {np.mean(np.abs(e)):.4f} {unit}")
    print(f"Max abs {err_name} error  : {np.max(np.abs(e)):.4f} {unit}")
    print(f"IAE                      : {compute_iae(t, e):.4f}")
    print(f"ISE                      : {compute_ise(t, e):.4f}")
    print(f"RMS(u_dq)                : {safe_rms(u_mag):.4f} V")
    print(f"Time@rail                : {compute_time_at_rail(t, u_mag, Vmax):.4f} s")
    print(f"Mean |id|                : {np.mean(np.abs(data['id'])):.4f} A")
    if args.mode == "speed":
        print(f"Mean |iq|                : {np.mean(np.abs(data['iq'])):.4f} A")
    print(f"Final speed              : {data['w'][-1]:.4f} rad/s")

    if args.perturb:
        print(f"Perturbation target      : {args.perturb_target}")
        print(f"Relative change          : {args.perturb_change:+.4f}")
        print(f"Perturbation time        : {data['perturb_time']:.4f} s")


# ----------------------------------------------------------
# Simulation
# ----------------------------------------------------------

def simulate(args):
    rng = np.random.default_rng(args.seed)

    motor = PMSM(
        Vdc=args.Vdc,
        Ld_pp=args.Ld_pp,
        Lq_pp=args.Lq_pp
    )
    R_nom, Ld_nom, Lq_nom, lam_nom = motor.derived()
    Vdc_nom = args.Vdc

    # actual runtime plant parameters
    R = R_nom
    Ld = Ld_nom
    Lq = Lq_nom
    lam = lam_nom
    Vdc = Vdc_nom

    x = State(0.0, 0.0, 0.0, 0.0)

    # PI gains are fixed user settings.
    # They are NOT recomputed after perturbation.
    int_id = 0.0
    int_iq = 0.0
    int_w = 0.0

    tau_inv = 80e-6

    vd = 0.0
    vq = 0.0

    if args.random_ref:
        w_vals, w_times = generate_profile(rng, -300.0, 300.0, 0.5, 1.2)
        iq_vals, iq_times = generate_profile(rng, -1.5, 1.5, 0.4, 1.0)
    else:
        w_vals, w_times = None, None
        iq_vals, iq_times = None, None

    if args.perturb:
        if args.perturb_random_time:
            perturb_time = rng.uniform(args.perturb_tmin, args.perturb_tmax)
        else:
            perturb_time = args.perturb_time
    else:
        perturb_time = -1.0

    perturb_applied = False

    log = {
        k: [] for k in [
            "t", "w", "w_ref", "id", "iq", "iq_ref", "vd", "vq", "Te",
            "R", "Ld", "Lq", "lam", "Vdc"
        ]
    }

    prev_iq_ref = 0.0
    w_ref_prev = 0.0

    n_steps = int(args.t_end / args.dt)

    for k in range(n_steps):
        t = k * args.dt

        # ----------------------------------------------
        # Online perturbation
        # ----------------------------------------------
        if args.perturb and (not perturb_applied) and (t >= perturb_time):
            factor = 1.0 + args.perturb_change

            if args.perturb_target == "R":
                R = R_nom * factor
            elif args.perturb_target == "Ld":
                Ld = Ld_nom * factor
            elif args.perturb_target == "Lq":
                Lq = Lq_nom * factor
            elif args.perturb_target == "lam":
                lam = lam_nom * factor
            elif args.perturb_target == "Vdc":
                Vdc = Vdc_nom * factor

            perturb_applied = True

        Vmax = Vdc / math.sqrt(3.0) * 0.95

        # ----------------------------------------------
        # References
        # ----------------------------------------------
        if args.random_ref:
            if args.mode == "speed":
                w_ref = profile_eval(t, w_vals, w_times)
            else:
                iq_ref = profile_eval(t, iq_vals, iq_times)
                w_ref = 0.0
        else:
            if args.mode == "speed":
                w_ref = args.w_ref
            else:
                iq_ref = args.iq_ref
                w_ref = 0.0

        if args.mode == "speed":
            dw_max = 500.0
            w_ref = np.clip(
                w_ref,
                w_ref_prev - dw_max * args.dt,
                w_ref_prev + dw_max * args.dt
            )
            w_ref_prev = w_ref

        # ----------------------------------------------
        # Mechanical mode
        # ----------------------------------------------
        if args.mode == "current":
            mech_mode = args.current_control_mode
        else:
            mech_mode = "locked" if args.lock_rotor else "dynamic"

        prescribed_w = None
        if args.mode == "current" and mech_mode == "sinus_speed":
            prescribed_w = sinus_speed_profile(t, args.t_end, args.sinus_w_amp)
            x.w = prescribed_w

        # ----------------------------------------------
        # Speed controller
        # ----------------------------------------------
        if args.mode == "speed":
            ew = w_ref - x.w
            iq_unsat = args.Kp_w * ew + args.Ki_w * int_w

            # fixed PI gains, but active plant parameters in limit calculations
            iq_dyn = Vmax / max(lam * motor.pole_pairs, 1e-6)
            iq_ref = np.clip(
                iq_unsat,
                -min(args.Imax, iq_dyn),
                min(args.Imax, iq_dyn)
            )

            aw_gain = 5.0
            int_w += (ew + aw_gain * (iq_ref - iq_unsat)) * args.dt
        else:
            int_w = 0.0

        # ----------------------------------------------
        # Current slew limit
        # ----------------------------------------------
        step = args.di_dt * args.dt
        iq_ref = np.clip(iq_ref, prev_iq_ref - step, prev_iq_ref + step)
        prev_iq_ref = iq_ref

        # ----------------------------------------------
        # Flux weakening
        # ----------------------------------------------
        id_ref = -0.3 if abs(x.w) > args.base_speed else 0.0

        # ----------------------------------------------
        # Current controller
        # ----------------------------------------------
        eid = id_ref - x.id
        eiq = iq_ref - x.iq

        int_id += eid * args.dt
        int_iq += eiq * args.dt

        we = motor.pole_pairs * x.w

        # PI gains stay unchanged.
        # Decoupling/feedforward and active plant-related terms use current perturbed params.
        vd_u = args.Kp_id * eid + args.Ki_id * int_id - we * Lq * x.iq
        vq_u = args.Kp_iq * eiq + args.Ki_iq * int_iq + we * (Ld * x.id + lam)

        mag = math.hypot(vd_u, vq_u)

        if mag > Vmax:
            scale = Vmax / mag
            vd_cmd = vd_u * scale
            vq_cmd = vq_u * scale
            int_id *= 0.98
            int_iq *= 0.98
        else:
            vd_cmd = vd_u
            vq_cmd = vq_u

        # ----------------------------------------------
        # Inverter dynamics
        # ----------------------------------------------
        vd += args.dt * (vd_cmd - vd) / tau_inv
        vq += args.dt * (vq_cmd - vq) / tau_inv

        vd = np.clip(vd, -Vmax, Vmax)
        vq = np.clip(vq, -Vmax, Vmax)

        # ----------------------------------------------
        # Motor
        # ----------------------------------------------
        x = rk4(
            motor, R, Ld, Lq, lam, x, args.dt, vd, vq, args.Tload,
            mech_mode=mech_mode,
            prescribed_w=prescribed_w
        )

        Te = 1.5 * motor.pole_pairs * (lam * x.iq + (Ld - Lq) * x.id * x.iq)

        # ----------------------------------------------
        # Logging
        # ----------------------------------------------
        log["t"].append(t)
        log["w"].append(x.w)
        log["w_ref"].append(w_ref)
        log["id"].append(x.id)
        log["iq"].append(x.iq)
        log["iq_ref"].append(iq_ref)
        log["vd"].append(vd)
        log["vq"].append(vq)
        log["Te"].append(Te)

        log["R"].append(R)
        log["Ld"].append(Ld)
        log["Lq"].append(Lq)
        log["lam"].append(lam)
        log["Vdc"].append(Vdc)

    for key in log:
        log[key] = np.array(log[key])

    log["perturb_time"] = perturb_time
    return log


# ----------------------------------------------------------
# Plot
# ----------------------------------------------------------

def plot(data, args):
    t = data["t"]
    perturb_time = data["perturb_time"]

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 10,
        "lines.linewidth": 1.5
    })

    Rn = data["R"] / data["R"][0]
    Ldn = data["Ld"] / data["Ld"][0]
    Lqn = data["Lq"] / data["Lq"][0]
    lamn = data["lam"] / data["lam"][0]
    Vdcn = data["Vdc"] / data["Vdc"][0]

    if args.mode == "speed":
        fig, ax = plt.subplots(4, 1, figsize=(9, 8), sharex=True)

        ax[0].plot(t, data["w"], label=r"$\omega$")
        ax[0].plot(t, data["w_ref"], "--", label=r"$\omega_{\mathrm{ref}}$")
        ax[0].set_ylabel(r"$\omega$ [rad/s]")
        ax[0].set_title("Speed Tracking")
        ax[0].legend()

        ax[1].plot(t, data["iq"], label=r"$i_q$")
        ax[1].plot(t, data["iq_ref"], "--", label=r"$i_{q,\mathrm{ref}}$")
        ax[1].plot(t, data["id"], label=r"$i_d$")
        ax[1].axhline(0.0, linestyle=":", linewidth=1.0)
        ax[1].set_ylabel(r"Current [A]")
        ax[1].set_title("dq-axis Currents")
        ax[1].legend()

        ax[2].plot(t, data["vq"], label=r"$v_q$")
        ax[2].plot(t, data["vd"], label=r"$v_d$")
        ax[2].set_ylabel(r"Voltage [V]")
        ax[2].set_title("Control Inputs")
        ax[2].legend()

        ax[3].plot(t, Rn, label="R/R_nom")
        ax[3].plot(t, Ldn, label="Ld/Ld_nom")
        ax[3].plot(t, Lqn, label="Lq/Lq_nom")
        ax[3].plot(t, lamn, label="lam/lam_nom")
        ax[3].plot(t, Vdcn, label="Vdc/Vdc_nom")
        ax[3].set_ylabel("Normalized")
        ax[3].set_xlabel("Time [s]")
        ax[3].set_title("Runtime Parameters")
        ax[3].legend(ncol=5, fontsize=9)

    else:
        fig, ax = plt.subplots(4, 1, figsize=(9, 8), sharex=True)

        ax[0].plot(t, data["iq"], label=r"$i_q$")
        ax[0].plot(t, data["iq_ref"], "--", label=r"$i_{q,\mathrm{ref}}$")
        ax[0].plot(t, data["id"], label=r"$i_d$")
        ax[0].axhline(0.0, linestyle=":", linewidth=1.0)
        ax[0].set_ylabel(r"Current [A]")
        ax[0].set_title("dq-axis Currents")
        ax[0].legend()

        ax[1].plot(t, data["w"], label=r"$\omega$")
        if args.current_control_mode == "sinus_speed":
            ax[1].plot(t, [sinus_speed_profile(tt, args.t_end, args.sinus_w_amp) for tt in t],
                       "--", label=r"$\omega_{\mathrm{sin}}$")
        ax[1].set_ylabel(r"$\omega$ [rad/s]")
        ax[1].set_title("Mechanical Response")
        ax[1].legend()

        ax[2].plot(t, data["vq"], label=r"$v_q$")
        ax[2].plot(t, data["vd"], label=r"$v_d$")
        ax[2].set_ylabel(r"Voltage [V]")
        ax[2].set_title("Control Inputs")
        ax[2].legend()

        ax[3].plot(t, Rn, label="R/R_nom")
        ax[3].plot(t, Ldn, label="Ld/Ld_nom")
        ax[3].plot(t, Lqn, label="Lq/Lq_nom")
        ax[3].plot(t, lamn, label="lam/lam_nom")
        ax[3].plot(t, Vdcn, label="Vdc/Vdc_nom")
        ax[3].set_ylabel("Normalized")
        ax[3].set_xlabel("Time [s]")
        ax[3].set_title("Runtime Parameters")
        ax[3].legend(ncol=5, fontsize=9)

    for a in ax:
        a.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
        if args.perturb and perturb_time >= 0:
            a.axvline(perturb_time, color="red", linestyle="--", alpha=0.8)

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--mode", choices=["speed", "current"], default="speed")
    p.add_argument("--random_ref", "--rand_ref", action="store_true")
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--dt", type=float, default=5e-5)
    p.add_argument("--t_end", type=float, default=5.0)

    p.add_argument("--Vdc", type=float, default=24.0)
    p.add_argument("--Ld_pp", type=float, default=1.61e-3)
    p.add_argument("--Lq_pp", type=float, default=1.61e-3)

    p.add_argument("--w_ref", type=float, default=300.0)
    p.add_argument("--iq_ref", type=float, default=0.5)
    p.add_argument("--Tload", type=float, default=0.002)

    p.add_argument("--Imax", type=float, default=1.6)
    p.add_argument("--di_dt", type=float, default=5.0)

    p.add_argument("--base_speed", type=float, default=500.0)

    p.add_argument("--lock_rotor", action="store_true")

    # current control mechanical mode
    p.add_argument("--current_control_mode", choices=["locked", "free", "sinus_speed"], default="locked")
    p.add_argument("--sinus_w_amp", type=float, default=300.0)

    # PI gains: fixed, not recalculated during perturbation
    p.add_argument("--Kp_id", type=float, default=1.2)
    p.add_argument("--Ki_id", type=float, default=6000.0)

    p.add_argument("--Kp_iq", type=float, default=1.2)
    p.add_argument("--Ki_iq", type=float, default=6000.0)

    p.add_argument("--Kp_w", type=float, default=0.03)
    p.add_argument("--Ki_w", type=float, default=0.03)

    # perturbation
    p.add_argument("--perturb", action="store_true")
    p.add_argument("--perturb_target", type=str, default="R", choices=["R", "Ld", "Lq", "lam", "Vdc"])
    p.add_argument("--perturb_change", type=float, default=0.3)
    p.add_argument("--perturb_random_time", action="store_true")
    p.add_argument("--perturb_time", type=float, default=2.0)
    p.add_argument("--perturb_tmin", type=float, default=1.0)
    p.add_argument("--perturb_tmax", type=float, default=4.0)

    return p


def main():
    args = build_parser().parse_args()
    # backward compatibility
    if args.mode == "current" and args.lock_rotor:
        args.current_control_mode = "locked"

    data = simulate(args)
    plot(data, args)
    print_global_metrics(data, args)
    print_segment_metrics(data, args)


if __name__ == "__main__":
    main()
