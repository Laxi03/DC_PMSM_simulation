#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Callable, Dict
from dataclasses import dataclass
import numpy as np, math, matplotlib
matplotlib.use("TkAgg")  # interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse, os
from datetime import datetime

# ---------------------------
# Run-scoped export folder
# ---------------------------
EXPORT_ROOT = "exports"
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(EXPORT_ROOT, RUN_TIMESTAMP)
os.makedirs(RUN_DIR, exist_ok=True)

# ---------------------------
# Helpers
# ---------------------------
def _movavg(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < k: return x
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="valid")

def _clip(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)

def _set_line_safe(line, x: np.ndarray, y: np.ndarray, enabled: bool) -> None:
    """Ensure we never give Matplotlib mismatched x/y shapes."""
    if enabled and len(x) and len(y) and len(x) == len(y):
        line.set_data(x, y)
    else:
        line.set_data([], [])

def export_subplot_csvs(label: str, t: np.ndarray, y: dict, run_dir: str = RUN_DIR) -> None:
    os.makedirs(run_dir, exist_ok=True)
    np.savetxt(os.path.join(run_dir, f"{label}_u.csv"),
               np.column_stack([t, y['u']]), delimiter=",", comments="", header="t,u")
    np.savetxt(os.path.join(run_dir, f"{label}_i.csv"),
               np.column_stack([t, y['i']]), delimiter=",", comments="", header="t,i")
    np.savetxt(os.path.join(run_dir, f"{label}_omega.csv"),
               np.column_stack([t, y['omega']]), delimiter=",", comments="", header="t,omega")

    if 'i_ref' in y and y['i_ref'] is not None:
        np.savetxt(os.path.join(run_dir, f"{label}_i_ref.csv"),
                   np.column_stack([t, y['i_ref']]), delimiter=",", comments="", header="t,i_ref")
    if 'omega_ref' in y and y['omega_ref'] is not None:
        np.savetxt(os.path.join(run_dir, f"{label}_omega_ref.csv"),
                   np.column_stack([t, y['omega_ref']]), delimiter=",", comments="", header="t,omega_ref")

    np.savetxt(os.path.join(run_dir, f"{label}_torque.csv"),
               np.column_stack([t, y['T_em'], y['T_load']]), delimiter=",", comments="", header="t,T_em,T_load")
    print(f"[CSV] Exported to: {run_dir}")

# ---------------------------
# Motor parameters
# ---------------------------
@dataclass
class AMax32Params:
    # Electrical
    R: float = 3.99           # Ohm
    L: float = 0.556e-3       # H
    kT: float = 35.2e-3       # N·m/A
    Ke: float = 35.2e-3       # V·s/rad
    # Mechanical
    J: float = 45.3e-7        # kg·m^2
    B: float = 1.0e-6         # N·m·s/rad (viscous)
    T_coulomb: float = 0.001506
    # Thermal (coarse)
    R_th_ha: float = 7.5
    R_th_wh: float = 2.1
    tau_w: float = 17.8
    tau_m: float = 521.0
    T_amb: float = 25.0
    T_max: float = 125.0
    # Datasheet (rpm kept only to derive baseline ω once)
    V_nom: float = 16.0
    n0_rpm: float = 6460.0
    n_nom_rpm: float = 5060.0
    def derived(self) -> Dict[str, float]:
        rpm_to_radps = lambda rpm: rpm * 2.0 * math.pi / 60.0
        return {
            'omega0': rpm_to_radps(self.n0_rpm),
            'omega_nom': rpm_to_radps(self.n_nom_rpm),
            'C_th_w': self.tau_w / self.R_th_wh
        }

@dataclass
class MotorState:
    i: float
    omega: float
    theta: float
    Tw: float

class AMax32Motor:
    def __init__(self, p: AMax32Params | None = None):
        self.p = p or AMax32Params()
        self.d = self.p.derived()
    def derivatives(self, t: float, x: MotorState, u_v: float, load_torque: float) -> MotorState:
        p = self.p
        di = (u_v - p.R * x.i - p.Ke * x.omega) / p.L
        sign = 0.0 if abs(x.omega) < 1e-6 else (-1.0 if x.omega < 0 else 1.0)
        T_fric = p.B * x.omega + p.T_coulomb * sign
        T_em = p.kT * x.i
        domega = (T_em - load_torque - T_fric) / p.J
        dtheta = x.omega
        P_cu = p.R * x.i * x.i
        P_me = abs(T_fric * x.omega)
        P_loss = P_cu + 0.2 * P_me
        Rth_total = p.R_th_wh + p.R_th_ha
        Cth_w = self.d['C_th_w']
        dTw = (P_loss - (x.Tw - p.T_amb) / Rth_total) / Cth_w
        return MotorState(di, domega, dtheta, dTw)

    @staticmethod
    def rk4_step(deriv: Callable[[float, MotorState], MotorState],
                 t: float, dt: float, x: MotorState) -> MotorState:
        k1 = deriv(t, x)
        k2 = deriv(t+0.5*dt, MotorState(x.i+0.5*dt*k1.i, x.omega+0.5*dt*k1.omega,
                                        x.theta+0.5*dt*k1.theta, x.Tw+0.5*dt*k1.Tw))
        k3 = deriv(t+0.5*dt, MotorState(x.i+0.5*dt*k2.i, x.omega+0.5*dt+k2.omega if False else x.omega+0.5*dt*k2.omega,
                                        x.theta+0.5*dt*k2.theta, x.Tw+0.5*dt*k2.Tw))
        k4 = deriv(t+dt, MotorState(x.i+dt*k3.i, x.omega+dt*k3.omega,
                                    x.theta+dt*k3.theta, x.Tw+dt*k3.Tw))
        return MotorState(
            x.i+(dt/6)*(k1.i+2*k2.i+2*k3.i+k4.i),
            x.omega+(dt/6)*(k1.omega+2*k2.omega+2*k3.omega+k4.omega),
            x.theta+(dt/6)*(k1.theta+2*k2.theta+2*k3.theta+k4.theta),
            x.Tw+(dt/6)*(k1.Tw+2*k2.Tw+2*k3.Tw+k4.Tw)
        )

# ---------------------------
# PWM helper
# ---------------------------
def pwm_voltage(Vdc=24.0, duty=0.5, freq=5_000.0):
    T = 1.0 / freq
    duty = _clip(float(duty), -1.0, 1.0)
    def u(t):
        phase = (t % T) / T
        if phase < abs(duty):
            return Vdc if duty >= 0.0 else -Vdc
        return 0.0
    return u

# ---------------------------
# Simulation + Live Plot
# ---------------------------
_ANIM = None

def run_live(t_end=2.0, dt=1e-4, V=24.0, Tload=0.02,
             steps_per_frame=200, scroll_window=1.0,
             pwm=False, smooth_ms=0.0, smooth_cycles=0.0,
             downsample=8, pwm_freq=5000.0, pwm_duty=0.4,
             speed_ctrl=False, w_ref=160.0, ref_profile='const',
             kps=0.002, kis=0.4, ulim=24.0, avg_pwm=False,
             current_ctrl=False, i_ref=0.5, kpc=2.0, kic=200.0,
             i_ref_profile='const'):

    """w_ref is in rad/s (no rpm anywhere)."""
    global _ANIM
    params = AMax32Params()
    motor = AMax32Motor(params)

    # Speed reference (rad/s)
    def speed_ref_omega_fun(t: float) -> float:
        if ref_profile == 'const':
            return float(w_ref)
        # Example step sequence in rad/s (edit as desired)
        if t < 1.0: return 262.0     # ~2500 rpm
        if t < 2.0: return 366.52    # ~3500 rpm
        if t < 3.0: return -52.36    # ~-500 rpm
        if t < 4.0: return 157.08    # ~1500 rpm
        return 523.60                # ~5000 rpm

    # Current reference
    def current_ref_fun(t: float) -> float:
        if i_ref_profile == 'const':
            return i_ref
        # Conservative steps (adjust freely)
        if t < 0.25: return 0.20
        if t < 0.50: return 0.30
        if t < 0.75: return -0.60
        if t < 1.00: return 0.25
        return 0.0

    # Controller integrator & helpers
    ui = 0.0
    last_i_ref = None
    ui_limit = 1.5            # clamp for integrator (in duty "units")
    duty_eps  = 1e-6

    # Sources
    u_fun = pwm_voltage(Vdc=V, duty=pwm_duty, freq=pwm_freq) if pwm else (lambda t: V)
    load_fun = (lambda t: Tload)

    # Buffers
    x = MotorState(0.0, 0.0, 0.0, params.T_amb)
    t_buf, u_buf, i_buf, omega_buf, Tem_buf, Tl_buf = [], [], [], [], [], []
    iref_buf, omega_ref_buf = [], []
    exported_once = False

    # Figure
    fig, axs = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    ax_u, ax_i, ax_w, ax_T = axs
    for ax in axs: ax.grid(True)
    ax_u.set_ylabel('u [V]'); ax_i.set_ylabel('i [A]')
    ax_w.set_ylabel('ω [rad/s]'); ax_T.set_ylabel('Torque [N·m]')
    ax_T.set_xlabel('t [s]')

    line_u, = ax_u.plot([], [])
    line_i, = ax_i.plot([], [])
    line_i_ref, = ax_i.plot([], [], linestyle='--', label='i_ref')
    line_w, = ax_w.plot([], [])
    line_w_ref, = ax_w.plot([], [], linestyle='--', label='ω_ref')
    line_Tem, = ax_T.plot([], [], label='T_em')
    line_Tl,  = ax_T.plot([], [], linestyle='--', label='T_load')

    ax_i.legend()
    ax_w.legend()
    ax_T.legend()
    fig.suptitle('A-max 32 DC Motor — LIVE (ω in rad/s)')
    fig.tight_layout()
    t_now = 0.0

    def _export_once():
        nonlocal exported_once
        if exported_once or not t_buf: return
        y = {
            'u': np.asarray(u_buf),
            'i': np.asarray(i_buf),
            'omega': np.asarray(omega_buf),
            'i_ref': np.asarray(iref_buf) if current_ctrl else None,
            'omega_ref': np.asarray(omega_ref_buf) if speed_ctrl else None,
            'T_em': np.asarray(Tem_buf),
            'T_load': np.asarray(Tl_buf),
        }
        export_subplot_csvs("live", np.asarray(t_buf), y, run_dir=RUN_DIR)
        exported_once = True

    fig.canvas.mpl_connect('close_event', lambda _evt: _export_once())

    def init():
        for ax in axs: ax.set_xlim(0, scroll_window)
        for ln in (line_u, line_i, line_i_ref, line_w, line_w_ref, line_Tem, line_Tl):
            ln.set_data([], [])
        return line_u, line_i, line_i_ref, line_w, line_w_ref, line_Tem, line_Tl

    def update(_frame):
        nonlocal x, t_now, ui, last_i_ref
        if t_now >= t_end:
            _export_once()
            if _ANIM is not None and _ANIM.event_source is not None:
                _ANIM.event_source.stop()
            return line_u, line_i, line_i_ref, line_w, line_w_ref, line_Tem, line_Tl

        # integrate several RK4 steps between frames
        for _ in range(steps_per_frame):
            if current_ctrl:
                i_ref_now = current_ref_fun(t_now)

                # small “bumpless” reset on big/sign-changing ref steps
                if last_i_ref is None: last_i_ref = i_ref_now
                if np.sign(last_i_ref) != np.sign(i_ref_now) or abs(i_ref_now - last_i_ref) > 0.5:
                    ui = 0.0
                last_i_ref = i_ref_now

                # PI on current with feed-forward & anti-windup
                e_i = i_ref_now - x.i
                ui += kic * dt * e_i
                ui = _clip(ui, -ui_limit, ui_limit)

                if pwm:
                    Veff = V if abs(V) > 1e-9 else 1.0
                    u_ff = (params.R * i_ref_now + params.Ke * x.omega) / Veff  # volts->duty
                    duty_pre = (kpc * e_i + ui) + u_ff
                    duty_lim = min(1.0, abs(ulim))
                    duty_sat = _clip(duty_pre, -duty_lim, duty_lim)
                    if avg_pwm:
                        u = V * duty_sat
                    else:
                        Tsw = 1.0 / max(pwm_freq, 1e-12)
                        phase = t_now % Tsw
                        u = (V if duty_sat >= 0 else -V) if phase < abs(duty_sat) * Tsw else 0.0
                    if abs(duty_pre) > duty_lim + duty_eps:
                        ui += (duty_sat - duty_pre)
                else:
                    v_ff = params.R * i_ref_now + params.Ke * x.omega
                    v_pre = (kpc * e_i + ui) + v_ff
                    v_sat = _clip(v_pre, -abs(ulim), +abs(ulim))
                    u = v_sat
                    if abs(v_pre - v_sat) > 1e-12:
                        ui += (v_sat - v_pre)

            elif speed_ctrl:
                wref = float(speed_ref_omega_fun(t_now))
                e = wref - x.omega
                ui += kis * dt * e
                u_cmd = kps * e + ui
                if pwm:
                    duty = _clip(u_cmd, -1.0, 1.0)
                    if avg_pwm:
                        u = V * duty
                    else:
                        Tsw = 1.0 / max(pwm_freq, 1e-12)
                        phase = t_now % Tsw
                        u = (V if duty >= 0 else -V) if phase < abs(duty) * Tsw else 0.0
                else:
                    u = _clip(u_cmd, -abs(ulim), +abs(ulim))
            else:
                u = u_fun(t_now)

            Tl = load_fun(t_now)

            # references for logging (NaN when controller off)
            omega_ref_now = speed_ctrl and speed_ref_omega_fun(t_now) or np.nan
            i_ref_now_log = current_ctrl and current_ref_fun(t_now) or np.nan

            # log BEFORE step
            t_buf.append(t_now); u_buf.append(u)
            i_buf.append(x.i);   omega_buf.append(x.omega)
            Tem_buf.append(params.kT * x.i); Tl_buf.append(Tl)
            iref_buf.append(i_ref_now_log); omega_ref_buf.append(omega_ref_now)

            x = AMax32Motor.rk4_step(lambda tau, s: motor.derivatives(tau, s, u, Tl),
                                     t_now, dt, x)
            t_now += dt
            if t_now >= t_end: break

        # tail for window
        tmin = max(0.0, t_now - scroll_window)
        def tail(xs, ys):
            k = 0
            while k < len(xs) and xs[k] < tmin: k += 1
            return xs[k:], ys[k:]

        tx, ux = tail(t_buf, u_buf)
        _, ix = tail(t_buf, i_buf)
        _, wx = tail(t_buf, omega_buf)
        _, Temx = tail(t_buf, Tem_buf)
        _, Tlx = tail(t_buf, Tl_buf)
        _, irefx = tail(t_buf, iref_buf)
        _, wrefx = tail(t_buf, omega_ref_buf)

        # smoothing (screen-only)
        k_ms = int((smooth_ms / 1000.0) / max(dt, 1e-12)) if smooth_ms > 0 else 1
        k_cy = int((smooth_cycles / max(pwm_freq, 1e-12)) / max(dt, 1e-12)) if (smooth_cycles > 0 and pwm) else 1
        k = max(1, k_cy if k_cy > 1 else k_ms)

        tx_arr = np.asarray(tx); ux_arr = np.asarray(ux); ix_arr = np.asarray(ix)
        wx_arr = np.asarray(wx); Temx_arr = np.asarray(Temx); Tlx_arr = np.asarray(Tlx)
        iref_arr = np.asarray(irefx); wref_arr = np.asarray(wrefx)

        if k > 1 and len(tx_arr) >= k:
            tx_eff   = tx_arr[k-1:]
            ux_eff   = _movavg(ux_arr,   k)
            ix_eff   = _movavg(ix_arr,   k)
            wx_eff   = _movavg(wx_arr,   k)
            Temx_eff = _movavg(Temx_arr, k)
            Tlx_eff  = _movavg(Tlx_arr,  k)
            iref_eff = _movavg(iref_arr, k)
            wref_eff = _movavg(wref_arr, k)
        else:
            tx_eff, ux_eff, ix_eff, wx_eff, Temx_eff, Tlx_eff, iref_eff, wref_eff = \
                tx_arr, ux_arr, ix_arr, wx_arr, Temx_arr, Tlx_arr, iref_arr, wref_arr

        # optional downsample for drawing
        step_draw = max(1, int(downsample))
        tx_eff   = tx_eff[::step_draw]
        ux_eff   = ux_eff[::step_draw]
        ix_eff   = ix_eff[::step_draw]
        wx_eff   = wx_eff[::step_draw]
        Temx_eff = Temx_eff[::step_draw]
        Tlx_eff  = Tlx_eff[::step_draw]
        iref_eff = iref_eff[::step_draw]
        wref_eff = wref_eff[::step_draw]

        # update artists (SAFE)
        _set_line_safe(line_u,      tx_eff, ux_eff, True)
        _set_line_safe(line_i,      tx_eff, ix_eff, True)
        _set_line_safe(line_i_ref,  tx_eff, iref_eff, current_ctrl)
        _set_line_safe(line_w,      tx_eff, wx_eff, True)
        _set_line_safe(line_w_ref,  tx_eff, wref_eff, speed_ctrl)
        _set_line_safe(line_Tem,    tx_eff, Temx_eff, True)
        _set_line_safe(line_Tl,     tx_eff, Tlx_eff, True)

        # scroll & auto-scale Y
        for ax in axs:
            ax.set_xlim(max(0, t_now - scroll_window), max(scroll_window, t_now))
            ax.relim(); ax.autoscale_view(scalex=False, scaley=True)
        ax_u.margins(y=0.10); ax_i.margins(y=0.20); ax_w.margins(y=0.10); ax_T.margins(y=0.15)

        return line_u, line_i, line_i_ref, line_w, line_w_ref, line_Tem, line_Tl

    # ---- Seed once so we have data before animation starts ----
    def _seed_one_frame():
        update(0)
    _seed_one_frame()
    for ax in axs:
        ax.set_xlim(0.0, max(scroll_window, t_end))

    # Animation (finite)
    interval_ms = 50
    n_frames = max(1, int(t_end / (steps_per_frame * dt)))
    _ANIM = FuncAnimation(fig, update, init_func=init, frames=n_frames,
                          blit=False, interval=interval_ms, cache_frame_data=False, repeat=False)

    plt.show()
    _export_once()

    # -------- Full run export (downsample ~2 ms) --------
    if t_buf:
        t_arr = np.asarray(t_buf, dtype=float)
        step = max(1, int(0.002 / dt))
        t_ds = t_arr[::step]
        u_arr = np.asarray(u_buf); i_arr = np.asarray(i_buf)
        w_arr = np.asarray(omega_buf); Tem_arr = np.asarray(Tem_buf); Tl_arr = np.asarray(Tl_buf)
        iref_arr = np.asarray(iref_buf); wref_arr = np.asarray(omega_ref_buf)

        y_full = {
            'u':        u_arr[::step],
            'i':        i_arr[::step],
            'omega':    w_arr[::step],
            'i_ref':    iref_arr[::step] if current_ctrl else None,
            'omega_ref':wref_arr[::step] if speed_ctrl else None,
            'T_em':     Tem_arr[::step],
            'T_load':   Tl_arr[::step],
        }
        export_subplot_csvs("live_full", t_ds, y_full, run_dir=RUN_DIR)

        # Static full-run figure
        fig2, axs2 = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
        axs2[0].plot(t_ds, y_full['u']);     axs2[0].set_ylabel('u [V]');    axs2[0].grid(True)
        axs2[1].plot(t_ds, y_full['i'], label='i')
        if current_ctrl and y_full['i_ref'] is not None:
            axs2[1].plot(t_ds, y_full['i_ref'], linestyle='--', label='i_ref')
            axs2[1].legend()
        axs2[1].set_ylabel('i [A]'); axs2[1].grid(True)
        axs2[2].plot(t_ds, y_full['omega'], label='ω')
        if speed_ctrl and y_full['omega_ref'] is not None:
            axs2[2].plot(t_ds, y_full['omega_ref'], linestyle='--', label='ω_ref')
            axs2[2].legend()
        axs2[2].set_ylabel('ω [rad/s]');  axs2[2].grid(True)
        axs2[3].plot(t_ds, y_full['T_em'], label='T_em')
        axs2[3].plot(t_ds, y_full['T_load'], label='T_load', linestyle='--')
        axs2[3].set_ylabel('Torque [N·m]'); axs2[3].set_xlabel('t [s]'); axs2[3].legend(); axs2[3].grid(True)
        fig2.suptitle("A-max 32 DC Motor — Full Run (ω in rad/s)")
        fig2.tight_layout()
        fig2.savefig(os.path.join(RUN_DIR, "full_run.png"), dpi=300)
        fig2.savefig(os.path.join(RUN_DIR, "full_run.pdf"))
        plt.close(fig2)
        print(f"[PLOT] Full run plot saved in {RUN_DIR}")

# ---------------------------
# CLI (LIVE ONLY)
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A-max 32 DC motor simulator (LIVE only, ω in rad/s)')
    parser.add_argument('--t_end', type=float, default=2.0)
    parser.add_argument('--dt', type=float, default=1e-4)
    parser.add_argument('--V', type=float, default=24.0)
    parser.add_argument('--Tload', type=float, default=0.02)
    parser.add_argument('--pwm', action='store_true')
    parser.add_argument('--pwm_freq', type=float, default=5000.0)
    parser.add_argument('--pwm_duty', type=float, default=0.4)
    parser.add_argument('--spf', type=int, default=200, help='RK4 steps per animation frame')
    parser.add_argument('--window', type=float, default=1.0)
    parser.add_argument('--smooth_ms', type=float, default=0.0)
    parser.add_argument('--smooth_cycles', type=float, default=0.0)
    parser.add_argument('--downsample', type=int, default=8)

    # Speed PI (ω in rad/s)
    parser.add_argument('--speed_ctrl', action='store_true')
    parser.add_argument('--w_ref', type=float, default=160.0, help='Speed reference in rad/s')
    parser.add_argument('--ref_profile', type=str, default='const', choices=['const', 'step_seq'])
    parser.add_argument('--kps', type=float, default=0.002)
    parser.add_argument('--kis', type=float, default=0.4)
    parser.add_argument('--ulim', type=float, default=24.0, help='DC volt limit or PWM duty limit (±1 when PWM)')
    parser.add_argument('--avg_pwm', action='store_true')

    # Current PI
    parser.add_argument('--current_ctrl', action='store_true')
    parser.add_argument('--i_ref', type=float, default=0.4)
    parser.add_argument('--i_ref_profile', type=str, default='const',
                        choices=['const', 'step_seq'], help='Current reference profile.')
    parser.add_argument('--kpc', type=float, default=1.0)
    parser.add_argument('--kic', type=float, default=50.0)

    args = parser.parse_args()

    run_live(t_end=args.t_end, dt=args.dt, V=args.V, Tload=args.Tload,
             steps_per_frame=args.spf, scroll_window=args.window, pwm=args.pwm,
             smooth_ms=args.smooth_ms, smooth_cycles=args.smooth_cycles,
             downsample=args.downsample, pwm_freq=args.pwm_freq, pwm_duty=args.pwm_duty,
             speed_ctrl=args.speed_ctrl, w_ref=args.w_ref, ref_profile=args.ref_profile,
             kps=args.kps, kis=args.kis, ulim=args.ulim, avg_pwm=args.avg_pwm,
             current_ctrl=args.current_ctrl, i_ref=args.i_ref, kpc=args.kpc, kic=args.kic,
             i_ref_profile=args.i_ref_profile)