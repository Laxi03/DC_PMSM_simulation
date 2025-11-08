#!/usr/bin/env python3
# compute_pi_metrics_v2.py
# Robust per-segment metrics: t_r_10_90 with fallbacks (t_r_10_63, t_s_2pct),
# sign-aware overshoot, SSE only on detected holds, achieved fraction.
# NOW: Time@rail computed in seconds via integration + fraction + detected rail level.
# Usage:
#   python compute_pi_metrics_v2.py /path/to/run --force-1s

import os, sys, glob, math, argparse
import numpy as np
import pandas as pd

# ---------------- CLI ----------------
ap = argparse.ArgumentParser()
ap.add_argument("run_folder")
ap.add_argument("--force-1s", action="store_true")
ap.add_argument("--dwell", type=float, default=0.5, help="SSE dwell window (s)")
ap.add_argument("--delta-ref-min", type=float, default=0.5, help="min abs step to detect (rad/s)")
args = ap.parse_args()
ROOT = os.path.abspath(args.run_folder)

# ---------------- I/O (aliases & helpers) ----------------
SYN_T   = ["t","time","time_s","time [s]","timestamp"]
SYN_W   = ["omega","w","speed","speed_rad_s","omega_rad_s"]
SYN_WR  = ["omega_ref","w_ref","ref","reference","omega_ref_rad_s"]
SYN_U   = ["u","voltage","v","ua","u_cmd","u_a"]

def read_csv_flexible(path):
    for kw in [dict(), dict(sep=";"), dict(header=None)]:
        try:
            df = pd.read_csv(path, **kw)
            return df
        except:
            pass
    raise RuntimeError(f"Cannot read {path}")

def col(df, names):
    m = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n in m:
            return m[n]
    return None

def pick(pathlist):
    # try exact, then glob
    for p in pathlist:
        q = os.path.join(ROOT, p)
        if os.path.exists(q):
            return q
    for p in pathlist:
        g = glob.glob(os.path.join(ROOT, p))
        if g:
            return g[0]
    return None

def load_series(path, y_aliases):
    df = read_csv_flexible(path)
    # headerless fallback
    if df.shape[1] == 2 and set(df.columns) == set([0,1]):
        df.columns = ["t","val"]
    tcol = col(df, SYN_T)
    ycol = col(df, y_aliases)
    if tcol is None or ycol is None:
        raise RuntimeError(f"{os.path.basename(path)} missing cols. Have: {list(df.columns)}")
    t = df[tcol].to_numpy(); y = df[ycol].to_numpy()
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    i = np.argsort(t)
    return t[i], y[i]

def interp_to(t_src, y_src, t_dst):
    return np.interp(t_dst, t_src, y_src) if len(t_src) and len(t_dst) else np.array([])

# ---------------- Segmentation ----------------
def detect_segments_from_ref(t, r, delta_ref_min):
    d = np.diff(r, prepend=r[0])
    rng = max(1.0, float(np.nanmax(r) - np.nanmin(r)))
    thr = max(delta_ref_min, 0.01 * rng)
    idx = np.where(np.abs(d) > thr)[0]
    # debounce 25 samples
    keep, last = [], -10**9
    for i in idx:
        if i - last >= 25:
            keep.append(i); last = i
    bounds = [0] + keep + [len(t)-1]
    return [(bounds[i], bounds[i+1]) for i in range(len(bounds)-1)]

def force_one_second_segments(t, segs):
    new = []
    for i0, i1 in segs:
        tt = t[i0:i1+1]
        ints = [b for b in range(int(math.floor(tt[0]))+1, int(math.ceil(tt[-1])))
                if tt[0] < b < tt[-1]]
        if not ints:
            new.append((i0, i1)); continue
        prev = i0
        for b in ints:
            # find index near the integer boundary b
            k = prev + int(np.argmin(np.abs(tt[prev-i0:] - b)))
            new.append((prev, k)); prev = k
        new.append((prev, i1))
    return new

# ---------------- Metrics ----------------
def trapz_nan(y, t):
    m = np.isfinite(y) & np.isfinite(t)
    return float(np.trapz(y[m], t[m])) if np.any(m) else float("nan")

def rms_nan(x):
    m = np.isfinite(x)
    return float(np.sqrt(np.mean(x[m]**2))) if np.any(m) else float("nan")

def time_at_rail(t, u, eps=0.02):
    """
    Return:
      dur_s  : seconds spent at rail (|u| >= (1-eps)*max|u|)
      frac   : fraction of the segment duration spent at rail
      umax   : the detected rail magnitude [V] for the segment
    """
    t = np.asarray(t); u = np.asarray(u)
    m = np.isfinite(t) & np.isfinite(u)
    if m.sum() < 2:
        return 0.0, 0.0, float("nan")
    t = t[m]; u = u[m]

    umax = float(np.nanmax(np.abs(u)))
    if umax < 1e-9:
        return 0.0, 0.0, umax

    mask = (np.abs(u) >= (1.0 - eps) * umax)
    dur_s = float(np.trapz(mask.astype(float), t))  # integrate boolean over time
    total = float(t[-1] - t[0])
    frac = (dur_s / total) if total > 0 else float("nan")
    return dur_s, frac, umax

def crossing_time(t, y, level, ge=True):
    if ge:
        idx = np.where(y >= level)[0]
    else:
        idx = np.where(y <= level)[0]
    return float(t[idx[0]]) if len(idx) else float("nan")

def t_rise_10_90(t, y, y0, yf):
    lo = y0 + 0.10*(yf - y0)
    hi = y0 + 0.90*(yf - y0)
    if yf >= y0:
        t10 = crossing_time(t, y, lo, ge=True)
        t90 = crossing_time(t, y, hi, ge=True)
    else:
        t10 = crossing_time(t, y, lo, ge=False)
        t90 = crossing_time(t, y, hi, ge=False)
    return (t90 - t10) if (np.isfinite(t10) and np.isfinite(t90)) else float("nan")

def t_rise_10_63(t, y, y0, yf):
    lo = y0 + 0.10*(yf - y0)
    l63 = y0 + 0.63*(yf - y0)
    if yf >= y0:
        t10 = crossing_time(t, y, lo, ge=True)
        t63 = crossing_time(t, y, l63, ge=True)
    else:
        t10 = crossing_time(t, y, lo, ge=False)
        t63 = crossing_time(t, y, l63, ge=False)
    return (t63 - t10) if (np.isfinite(t10) and np.isfinite(t63)) else float("nan")

def settling_time_2pct(t, y, yf):
    tol = 0.02*max(1.0, abs(yf))
    m = np.where(np.abs(y - yf) <= tol)[0]
    return float(t[m[0]]) if len(m) else float("nan")

def overshoot_signed_pct(y, y0, yf):
    amp = abs(yf - y0)
    if amp < 1e-9:
        return float("nan")
    if yf >= y0:   # rising
        peak = float(np.nanmax(y))
        return 100.0*max(0.0, (peak - yf) / amp)
    else:          # falling
        trough = float(np.nanmin(y))
        return 100.0*max(0.0, (yf - trough) / amp)

def achieved_fraction(y, y0, yf):
    amp = abs(yf - y0)
    if amp < 1e-9:
        return float("nan")
    if yf >= y0:
        return float((np.nanmax(y) - y0) / amp)
    else:
        return float((y0 - np.nanmin(y)) / amp)

def sse_on_hold(t, e, r, dwell=0.5):
    # compute SSE only if reference is flat over last dwell window
    if len(t) == 0:
        return float("nan")
    end = t[-1]; start = max(t[0], end - dwell)
    m = (t >= start)
    r_win = r[m]
    if len(r_win) < 3:
        return float("nan")
    if np.nanstd(r_win) > 1e-3:  # not a hold → undefined SSE
        return float("nan")
    e_win = e[m]
    return float(np.nanmean(np.abs(e_win)))

# ---------------- Load data ----------------
omega_path = pick(["live_full_omega.csv","live_omega.csv","eval_time_speed_radps.csv","*omega*.csv"])
ref_path   = pick(["live_full_omega_ref.csv","live_omega_ref.csv","time_speed.csv","*ref*.csv","*omega_ref*.csv"])
volt_path  = pick(["live_full_u.csv","live_u.csv","time_voltage.csv","*volt*.csv","*u*.csv"])

if not (omega_path and ref_path and volt_path):
    sys.exit("ERROR: need omega, omega_ref (or time_speed), and u (voltage) CSVs in the run folder")

t_w, w = load_series(omega_path, SYN_W)

# Reference can be either explicit omega_ref, or the combined time_speed with both cols
if os.path.basename(ref_path).lower() == "time_speed.csv":
    df_rs = read_csv_flexible(ref_path)
    tcol = col(df_rs, SYN_T)
    rcol = col(df_rs, SYN_WR) or col(df_rs, ["omega_ref"])
    if tcol is None or rcol is None:
        sys.exit("ERROR: time_speed.csv must contain time + omega_ref columns")
    t_wr = df_rs[tcol].to_numpy()
    w_ref = df_rs[rcol].to_numpy()
else:
    t_wr, w_ref = load_series(ref_path, SYN_WR)

t_u, u = load_series(volt_path, SYN_U)

# Align reference and voltage to speed time base
w_ref = interp_to(t_wr, w_ref, t_w)
u     = interp_to(t_u, u,     t_w)

# ---------------- Segments ----------------
segs = detect_segments_from_ref(t_w, w_ref, args.delta_ref_min)
if args.force_1s:
    segs = force_one_second_segments(t_w, segs)

# ---------------- Compute per-segment metrics ----------------
rows = []
for i0, i1 in segs:
    t_seg = t_w[i0:i1+1]; y = w[i0:i1+1]; r = w_ref[i0:i1+1]; uu = u[i0:i1+1]
    m = np.isfinite(t_seg) & np.isfinite(y) & np.isfinite(r) & np.isfinite(uu)
    t_seg, y, r, uu = t_seg[m], y[m], r[m], uu[m]
    if len(t_seg) < 5:
        continue

    # Errors and endpoints
    e = r - y
    yf_ref = float(r[-1])

    # Rise/settling (with fallbacks possible at analysis time)
    tr_10_90 = t_rise_10_90(t_seg, y, y[0], yf_ref)
    tr_10_63 = t_rise_10_63(t_seg, y, y[0], yf_ref)
    ts_2pct  = settling_time_2pct(t_seg, y, yf_ref)

    # Overshoot & achieved fraction
    Mp_pct   = overshoot_signed_pct(y, y[0], yf_ref)
    ach_frac = achieved_fraction(y, y[0], yf_ref)

    # Integral metrics
    IAE = trapz_nan(np.abs(e), t_seg)
    ISE = trapz_nan(e**2, t_seg)
    SSE = sse_on_hold(t_seg, e, r, dwell=args.dwell)

    # Voltage stats incl. rail time
    dur_rail_s, frac_rail, umax = time_at_rail(t_seg, uu, eps=0.02)

    rows.append(dict(
        Segment=f"{t_seg[0]:.2f}–{t_seg[-1]:.2f}s",
        t_r_10_90_s=tr_10_90,
        t_r_10_63_s=tr_10_63,
        t_s_2pct_s=ts_2pct,
        Overshoot_pct=Mp_pct,
        Achieved_frac=ach_frac,
        SSE_rad_per_s=SSE,
        IAE=IAE,
        ISE=ISE,
        RMS_u_V=rms_nan(uu),
        Time_at_rail_s=dur_rail_s,
        Time_at_rail_frac=frac_rail,
        U_rail_V=umax
    ))

df = pd.DataFrame(rows)

# ---------------- Save CSV + LaTeX ----------------
out_csv = os.path.join(ROOT, "pi_metrics_v2.csv")
out_tex = os.path.join(ROOT, "pi_metrics_v2.tex")
df.to_csv(out_csv, index=False)

df_tex = df.rename(columns={
    "Segment":            "Segment",
    "t_r_10_90_s":        "$t_r^{10\\text{–}90}$ [s]",
    "t_r_10_63_s":        "$t_r^{10\\text{–}63}$ [s]",
    "t_s_2pct_s":         "$t_s^{2\\%}$ [s]",
    "Overshoot_pct":      "$M_p$ [\\%]",
    "Achieved_frac":      "Achieved frac.",
    "SSE_rad_per_s":      "SSE [rad/s]",
    "IAE":                "IAE",
    "ISE":                "ISE",
    "RMS_u_V":            "RMS$(u)$ [V]",
    "Time_at_rail_s":     "Time@rail [s]",
    "Time_at_rail_frac":  "Time@rail [–]",
    "U_rail_V":           "$\\|u\\|_{\\max}$ [V]"
})
# 13 columns total: 1 'l' + 12 'c'
latex = df_tex.to_latex(index=False,
                        float_format=lambda x: f"{x:.3g}",
                        column_format="lccccccccccccc",
                        caption="PI step–segment metrics with fallbacks, hold-aware SSE, and rail-time statistics.",
                        label="tab:pi_metrics_v2")

with open(out_tex, "w", encoding="utf-8") as f:
    f.write(latex)

print("Saved:\n ", out_csv, "\n ", out_tex)
print("\nPreview:")
print(df.to_string(index=False))