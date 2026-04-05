#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import argparse
import matplotlib.pyplot as plt
from dataclasses import dataclass

plt.style.use("seaborn-v0_8-darkgrid")


# ----------------------------------------------------------
# Motor parameters
# ----------------------------------------------------------

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


# ----------------------------------------------------------
# Random profile generator
# ----------------------------------------------------------

def generate_profile(val_min, val_max, dur_min, dur_max):
    n = np.random.randint(6, 12)

    values = np.random.uniform(val_min, val_max, n)
    durations = np.random.uniform(dur_min, dur_max, n)

    for i in range(1, len(values)):
        values[i] = 0.6 * values[i - 1] + 0.4 * values[i]

    times = np.cumsum(durations)
    return values, times


def profile_eval(t, values, times):
    for i in range(len(times)):
        if t < times[i]:
            return values[i]
    return values[-1]


# ----------------------------------------------------------
# PMSM model
# ----------------------------------------------------------

def derivatives(motor, R, L, lam, x, vd, vq, Tload, lock_rotor):
    p = motor.pole_pairs
    we = p * x.w

    did = (vd - R * x.id + we * L * x.iq) / L
    diq = (vq - R * x.iq - we * (L * x.id + lam)) / L

    Te = 1.5 * p * lam * x.iq

    if lock_rotor:
        dw = 0.0
    else:
        dw = (Te - Tload - motor.B * x.w) / motor.J

    dth = we

    return State(did, diq, dw, dth)


def rk4(motor, R, L, lam, x, dt, vd, vq, Tload, lock_rotor):
    def f(s):
        return derivatives(motor, R, L, lam, s, vd, vq, Tload, lock_rotor)

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


# ----------------------------------------------------------
# Simulation
# ----------------------------------------------------------

def simulate(args):
    motor = PMSM(Vdc=args.Vdc)
    R, L, lam = motor.derived()

    x = State(0.0, 0.0, 0.0, 0.0)

    int_id = 0.0
    int_iq = 0.0
    int_w = 0.0

    Vmax = args.Vdc / math.sqrt(3) * 0.95
    tau_inv = 80e-6

    vd = 0.0
    vq = 0.0

    if args.random_ref:
        w_vals, w_times = generate_profile(-300, 300, 0.5, 1.2)
        iq_vals, iq_times = generate_profile(-1.5, 1.5, 0.4, 1.0)
    else:
        w_vals, w_times = None, None
        iq_vals, iq_times = None, None

    log = {k: [] for k in ["t", "w", "w_ref", "id", "iq", "iq_ref", "vd", "vq", "Te"]}

    prev_iq_ref = 0.0
    w_ref_prev = 0.0

    n_steps = int(args.t_end / args.dt)

    for k in range(n_steps):
        t = k * args.dt

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
                iq_ref = 0.0
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
        # Speed controller
        # ----------------------------------------------

        if args.mode == "speed":
            ew = w_ref - x.w
            iq_unsat = args.Kp_w * ew + args.Ki_w * int_w

            iq_dyn = Vmax / (lam * motor.pole_pairs + 1e-6)
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

        vd_u = args.Kp_id * eid + args.Ki_id * int_id - we * L * x.iq
        vq_u = args.Kp_iq * eiq + args.Ki_iq * int_iq + we * (L * x.id + lam)

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

        # ----------------------------------------------
        # Motor
        # ----------------------------------------------

        x = rk4(motor, R, L, lam, x, args.dt, vd, vq, args.Tload, args.lock_rotor)

        Te = 1.5 * motor.pole_pairs * lam * x.iq

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

    for key in log:
        log[key] = np.array(log[key])

    return log


# ----------------------------------------------------------
# Plot
# ----------------------------------------------------------

def plot(data, args):
    t = data["t"]

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 10,
        "lines.linewidth": 1.5
    })

    if args.mode == "speed":
        fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

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
        ax[2].set_xlabel("Time [s]")
        ax[2].set_title("Control Inputs")
        ax[2].legend()

    else:
        fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

        ax[0].plot(t, data["iq"], label=r"$i_q$")
        ax[0].plot(t, data["iq_ref"], "--", label=r"$i_{q,\mathrm{ref}}$")
        ax[0].plot(t, data["id"], label=r"$i_d$")
        ax[0].axhline(0.0, linestyle=":", linewidth=1.0)
        ax[0].set_ylabel(r"Current [A]")
        ax[0].set_title("dq-axis Currents")
        ax[0].legend()

        ax[1].plot(t, data["w"], label=r"$\omega$")
        ax[1].set_ylabel(r"$\omega$ [rad/s]")
        ax[1].set_title("Mechanical Response")
        ax[1].legend()

        ax[2].plot(t, data["vq"], label=r"$v_q$")
        ax[2].plot(t, data["vd"], label=r"$v_d$")
        ax[2].set_ylabel(r"Voltage [V]")
        ax[2].set_xlabel("Time [s]")
        ax[2].set_title("Control Inputs")
        ax[2].legend()

    for a in ax:
        a.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--mode", choices=["speed", "current"], default="speed")
    p.add_argument("--random_ref", "--rand_ref", action="store_true")

    p.add_argument("--dt", type=float, default=5e-5)
    p.add_argument("--t_end", type=float, default=5.0)

    p.add_argument("--Vdc", type=float, default=24.0)
    p.add_argument("--w_ref", type=float, default=300.0)
    p.add_argument("--Tload", type=float, default=0.002)

    p.add_argument("--Imax", type=float, default=1.6)
    p.add_argument("--di_dt", type=float, default=5.0)

    p.add_argument("--base_speed", type=float, default=500.0)

    p.add_argument("--lock_rotor", action="store_true")

    p.add_argument("--Kp_id", type=float, default=1.2)
    p.add_argument("--Ki_id", type=float, default=6000.0)

    p.add_argument("--Kp_iq", type=float, default=1.2)
    p.add_argument("--Ki_iq", type=float, default=6000.0)

    p.add_argument("--Kp_w", type=float, default=0.03)
    p.add_argument("--Ki_w", type=float, default=0.03)

    args = p.parse_args()

    data = simulate(args)
    plot(data, args)


if __name__ == "__main__":
    main()