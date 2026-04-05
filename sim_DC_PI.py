#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import argparse
import matplotlib.pyplot as plt
from dataclasses import dataclass

plt.style.use("seaborn-v0_8-darkgrid")


# ==========================================================
# MOTOR PARAMETERS
# ==========================================================

@dataclass
class DCMotorParams:
    R: float = 3.99
    L: float = 0.556e-3
    kT: float = 35.2e-3
    Ke: float = 35.2e-3
    J: float = 45.3e-7
    B: float = 1.0e-6
    T_coulomb: float = 0.001506


@dataclass
class State:
    i: float
    omega: float
    theta: float


# ==========================================================
# RANDOM REFERENCES
# ==========================================================

def generate_random_profile(val_min, val_max, dur_min, dur_max, n_min=6, n_max=12):
    n = np.random.randint(n_min, n_max)
    values = np.random.uniform(val_min, val_max, n)
    durations = np.random.uniform(dur_min, dur_max, n)
    times = np.cumsum(durations)
    return values, times


def profile_eval(t, values, times):
    for i in range(len(times)):
        if t < times[i]:
            return values[i]
    return values[-1]


# ==========================================================
# MODEL
# ==========================================================

def derivatives(motor, x, u, Tload, lock_rotor=False):
    sign = 0.0 if abs(x.omega) < 1e-6 else (1.0 if x.omega > 0 else -1.0)

    di = (u - motor.R * x.i - motor.Ke * x.omega) / motor.L

    T_em = motor.kT * x.i
    T_fric = motor.B * x.omega + motor.T_coulomb * sign

    if lock_rotor:
        domega = 0.0
        dtheta = 0.0
    else:
        domega = (T_em - Tload - T_fric) / motor.J
        dtheta = x.omega

    return State(di, domega, dtheta)


def rk4_step(motor, x, dt, u, Tload, lock_rotor=False):
    def f(s):
        return derivatives(motor, s, u, Tload, lock_rotor)

    k1 = f(x)
    k2 = f(State(
        x.i + 0.5 * dt * k1.i,
        x.omega + 0.5 * dt * k1.omega,
        x.theta + 0.5 * dt * k1.theta
    ))
    k3 = f(State(
        x.i + 0.5 * dt * k2.i,
        x.omega + 0.5 * dt * k2.omega,
        x.theta + 0.5 * dt * k2.theta
    ))
    k4 = f(State(
        x.i + dt * k3.i,
        x.omega + dt * k3.omega,
        x.theta + dt * k3.theta
    ))

    return State(
        x.i + dt * (k1.i + 2*k2.i + 2*k3.i + k4.i) / 6.0,
        x.omega + dt * (k1.omega + 2*k2.omega + 2*k3.omega + k4.omega) / 6.0,
        x.theta + dt * (k1.theta + 2*k2.theta + 2*k3.theta + k4.theta) / 6.0
    )


# ==========================================================
# SIMULATION
# ==========================================================

def simulate(args):
    motor = DCMotorParams()
    x = State(0.0, 0.0, 0.0)

    u = 0.0
    int_speed = 0.0
    int_current = 0.0

    speed_vals = None
    speed_times = None
    current_vals = None
    current_times = None

    if args.random_ref:
        if args.mode == "speed":
            speed_vals, speed_times = generate_random_profile(
                args.w_ref_min, args.w_ref_max,
                args.seg_min, args.seg_max
            )
        elif args.mode == "current":
            current_vals, current_times = generate_random_profile(
                args.i_ref_min, args.i_ref_max,
                args.seg_min, args.seg_max
            )

    log = {
        "t": [],
        "u": [],
        "i": [],
        "omega": [],
        "theta": [],
        "T_em": [],
        "Tload": [],
        "omega_ref": [],
        "i_ref": []
    }

    n_steps = int(args.t_end / args.dt)

    for k in range(n_steps):
        t = k * args.dt

        if args.mode == "speed":
            if args.random_ref:
                omega_ref = profile_eval(t, speed_vals, speed_times)
            else:
                omega_ref = args.w_ref

            e_speed = omega_ref - x.omega
            int_speed += e_speed * args.dt
            int_speed = np.clip(int_speed, -args.int_speed_limit, args.int_speed_limit)

            i_ref_unsat = args.Kp_w * e_speed + args.Ki_w * int_speed
            i_ref = np.clip(i_ref_unsat, -args.i_limit, args.i_limit)

            e_current = i_ref - x.i
            int_current += e_current * args.dt
            int_current = np.clip(int_current, -args.int_current_limit, args.int_current_limit)

            u_ff = motor.R * i_ref + motor.Ke * x.omega
            u_cmd = args.Kp_i * e_current + args.Ki_i * int_current + u_ff

        elif args.mode == "current":
            omega_ref = 0.0

            if args.random_ref:
                i_ref = profile_eval(t, current_vals, current_times)
            else:
                i_ref = args.i_ref

            e_current = i_ref - x.i
            int_current += e_current * args.dt
            int_current = np.clip(int_current, -args.int_current_limit, args.int_current_limit)

            u_ff = motor.R * i_ref + motor.Ke * x.omega
            u_cmd = args.Kp_i * e_current + args.Ki_i * int_current + u_ff

        else:
            raise ValueError("Unsupported mode")

        u_cmd = np.clip(u_cmd, -args.Vdc, args.Vdc)

        alpha = min(args.dt / max(args.tau_act, 1e-9), 1.0)
        u += alpha * (u_cmd - u)

        x = rk4_step(
            motor=motor,
            x=x,
            dt=args.dt,
            u=u,
            Tload=args.Tload,
            lock_rotor=args.lock_rotor
        )

        T_em = motor.kT * x.i

        log["t"].append(t)
        log["u"].append(u)
        log["i"].append(x.i)
        log["omega"].append(x.omega)
        log["theta"].append(x.theta)
        log["T_em"].append(T_em)
        log["Tload"].append(args.Tload)
        log["omega_ref"].append(omega_ref)
        log["i_ref"].append(i_ref)

    for key in log:
        log[key] = np.array(log[key])

    return log


# ==========================================================
# METRICS
# ==========================================================

def print_metrics(data, args):
    if args.mode == "speed":
        e = data["omega_ref"] - data["omega"]
        print(f"Mean abs speed error : {np.mean(np.abs(e)):.3f} rad/s")
        print(f"Max abs speed error  : {np.max(np.abs(e)):.3f} rad/s")
        print(f"Mean |i|             : {np.mean(np.abs(data['i'])):.3f} A")
        print(f"Mean |u|             : {np.mean(np.abs(data['u'])):.3f} V")
    else:
        e = data["i_ref"] - data["i"]
        print(f"Mean abs current error : {np.mean(np.abs(e)):.4f} A")
        print(f"Max abs current error  : {np.max(np.abs(e)):.4f} A")
        print(f"Mean |u|               : {np.mean(np.abs(data['u'])):.3f} V")
        print(f"Final speed            : {data['omega'][-1]:.3f} rad/s")


# ==========================================================
# PLOTTING
# ==========================================================

def plot_results(data, args):
    t = data["t"]

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "lines.linewidth": 1.8
    })

    if args.mode == "speed":
        fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)

        ax[0].plot(t, data["omega"], label=r"$\omega$")
        ax[0].plot(t, data["omega_ref"], "--", label=r"$\omega_{\mathrm{ref}}$")
        ax[0].set_ylabel(r"$\omega$ [rad/s]")
        ax[0].set_title("Speed Tracking")
        ax[0].legend()

        ax[1].plot(t, data["i"], label=r"$i$")
        ax[1].set_ylabel(r"$i$ [A]")
        ax[1].set_title("Armature Current")
        ax[1].legend()

        ax[2].plot(t, data["u"], label=r"$u$")
        ax[2].set_ylabel(r"$u$ [V]")
        ax[2].set_title("Control Effort")
        ax[2].legend()

        ax[3].plot(t, data["T_em"], label=r"$T_{\mathrm{em}}$")
        ax[3].set_ylabel(r"$T_{\mathrm{em}}$ [Nm]")
        ax[3].set_xlabel("Time [s]")
        ax[3].set_title("Electromagnetic Torque")
        ax[3].legend()

    elif args.mode == "current":
        fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)

        ax[0].plot(t, data["i"], label=r"$i$")
        ax[0].plot(t, data["i_ref"], "--", label=r"$i_{\mathrm{ref}}$")
        ax[0].set_ylabel(r"$i$ [A]")
        ax[0].set_title("Current Tracking")
        ax[0].legend()

        ax[1].plot(t, data["omega"], label=r"$\omega$")
        ax[1].set_ylabel(r"$\omega$ [rad/s]")
        ax[1].set_title("Mechanical Response During Current Control")
        ax[1].legend()

        ax[2].plot(t, data["u"], label=r"$u$")
        ax[2].set_ylabel(r"$u$ [V]")
        ax[2].set_title("Control Effort")
        ax[2].legend()

        ax[3].plot(t, data["T_em"], label=r"$T_{\mathrm{em}}$")
        ax[3].set_ylabel(r"$T_{\mathrm{em}}$ [Nm]")
        ax[3].set_xlabel("Time [s]")
        ax[3].set_title("Electromagnetic Torque")
        ax[3].legend()

    for a in ax:
        a.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)

    plt.tight_layout()
    plt.show()


# ==========================================================
# CLI
# ==========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["speed", "current"], default="speed")
    parser.add_argument("--random_ref", action="store_true")
    parser.add_argument("--lock_rotor", action="store_true")

    parser.add_argument("--t_end", type=float, default=6.0)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--Vdc", type=float, default=24.0)
    parser.add_argument("--Tload", type=float, default=0.002)
    parser.add_argument("--tau_act", type=float, default=5e-4)

    parser.add_argument("--w_ref", type=float, default=150.0)
    parser.add_argument("--i_ref", type=float, default=0.5)

    parser.add_argument("--w_ref_min", type=float, default=-250.0)
    parser.add_argument("--w_ref_max", type=float, default=250.0)
    parser.add_argument("--i_ref_min", type=float, default=-1.5)
    parser.add_argument("--i_ref_max", type=float, default=1.5)
    parser.add_argument("--seg_min", type=float, default=0.5)
    parser.add_argument("--seg_max", type=float, default=1.2)

    parser.add_argument("--Kp_w", type=float, default=0.015)
    parser.add_argument("--Ki_w", type=float, default=0.8)

    parser.add_argument("--Kp_i", type=float, default=3.0)
    parser.add_argument("--Ki_i", type=float, default=400.0)

    parser.add_argument("--i_limit", type=float, default=2.0)
    parser.add_argument("--int_speed_limit", type=float, default=1000.0)
    parser.add_argument("--int_current_limit", type=float, default=10.0)

    args = parser.parse_args()

    data = simulate(args)
    print_metrics(data, args)
    plot_results(data, args)


if __name__ == "__main__":
    main()