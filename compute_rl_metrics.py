#!/usr/bin/env python3
# compute_rl_metrics.py
# Per-segment metrics for RL evaluation runs:
#   rise time (10–90%), sign-aware overshoot, achieved fraction,
#   SSE (hold-aware), IAE, ISE, RMS(u), time@rail.
# Works with: eval_time_speed_radps.csv, eval_time_voltage.csv (+ optional current/torque).
# Usage:
#   python3 compute_rl_metrics.py /path/to/folder --force-1s

import os, sys, glob, math, argparse
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

ap = argparse.ArgumentParser()
ap.add_argument("run_folder")
ap.add_argument("--force-1s", action="store_true", help="force 1.0 s segments")
ap.add_argument("--dwell", type=float, default=0.5, help="SSE dwell window at end of segment (s)")
ap.add_argument("--delta-ref-min", type=float, default=0.5, help="min abs step to detect (rad/s)")
args = ap.parse_args()
ROOT = os.path.abspath(args.run_folder)

# ------- helpers (file I/O) -------
SYN_T   = ["t","time","time_s","time [s]","timestamp"]
SYN_W   = ["omega","w","speed","speed_radps","speed_rad/s","omega_radps","omega_rad/s"]
SYN_WR  = ["omega_ref","w_ref","ref","reference","omega_ref_radps","omega_ref_rad/s"]
SYN_U   = ["u","voltage","v","ua","u_cmd","u_a"]

def read_csv_flexible(path: str) -> pd.DataFrame:
    for kw in [dict(), dict(sep=";"), dict(header=None)]:
        try:
            df = pd.read_csv(path, **kw)
            return df
        except Exception:
            continue
    raise RuntimeError(f"Cannot read {path}")

def col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    m = {c.lower(): c for c in df.columns}
    for n in names:
        if n in m: return m[n]
    return None

def pick_exact_or_glob(names: List[str]) -> Optional[str]:
    for n in names:
        p = os.path.join(ROOT, n)
        if os.path.exists(p): return p
    # glob fallback
    for n in names:
        g = glob.glob(os.path.join(ROOT, n))
        if g: return g[0]
    return None

def load_series(path: str, y_aliases: List[str]):
    df = read_csv_flexible(path)
    if df.shape[1] == 2 and set(df.columns) == set([0,1]):
        df.columns = ["t","val"]
    tcol = col(df, SYN_T) or df.columns[0]
    ycol = col(df, y_aliases) or df.columns[1]
    t = df[tcol].to_numpy(); y = df[ycol].to_numpy()
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    i = np.argsort(t)
    return t[i], y[i], df.columns.tolist()

def interp_to(t_src, y_src, t_dst):
    return np.interp(t_dst, t_src, y_src) if len(t_src) and len(t_dst) else np.array([])

# ------- segmentation & metrics -------
def detect_segments_from_ref(t, r, delta_ref_min):
    d = np.diff(r, prepend=r[0])
    rng = max(1.0, float(np.nanmax(r) - np.nanmin(r)))
    thr = max(delta_ref_min, 0.01 * rng)
    idx = np.where(np.abs(d) > thr)[0]
    # debounce ~25 samples
    keep, last = [], -10**9
    for i in idx:
        if i - last >= 25:
            keep.append(i); last = i
    bounds = [0] + keep + [len(t)-1]
    return [(bounds[i], bounds[i+1]) for i in range(len(bounds)-1)]

def force_one_second_segments(t, segs):
    new=[]
    for i0,i1 in segs:
        tt = t[i0:i1+1]
        ints = [b for b in range(int(math.floor(tt[0]))+1, int(math.ceil(tt[-1])))
                if tt[0] < b < tt[-1]]
        if not ints:
            new.append((i0,i1)); continue
        prev=i0
        for b in ints:
            k = prev + int(np.argmin(np.abs(tt[prev-i0:] - b)))
            new.append((prev,k)); prev = k
        new.append((prev,i1))
    return new

def trapz_nan(y,t):
    m = np.isfinite(y) & np.isfinite(t)
    return float(np.trapz(y[m], t[m])) if np.any(m) else float("nan")

def rms_nan(x):
    m = np.isfinite(x)
    return float(np.sqrt(np.mean(x[m]**2))) if np.any(m) else float("nan")

def time_at_rail(u, atol=1e-3):
    m = np.isclose(np.abs(u), np.nanmax(np.abs(u)), atol=atol)
    return float(np.mean(m)) if len(u) else 0.0

def crossing_time(t,y,level,ge=True):
    idx = np.where((y>=level) if ge else (y<=level))[0]
    return float(t[idx[0]]) if len(idx) else float("nan")

def t_rise_10_90(t,y,y0,yf):
    lo = y0 + 0.10*(yf - y0)
    hi = y0 + 0.90*(yf - y0)
    if yf >= y0:
        t10 = crossing_time(t,y,lo,ge=True); t90 = crossing_time(t,y,hi,ge=True)
    else:
        t10 = crossing_time(t,y,lo,ge=False); t90 = crossing_time(t,y,hi,ge=False)
    return (t90 - t10) if (np.isfinite(t10) and np.isfinite(t90)) else float("nan")

def overshoot_signed_pct(y,y0,yf):
    amp = abs(yf - y0)
    if amp < 1e-12: return float("nan")
    if yf >= y0:  # rising
        peak = float(np.nanmax(y)); return 100.0*max(0.0,(peak - yf)/amp)
    else:        # falling
        trough = float(np.nanmin(y)); return 100.0*max(0.0,(yf - trough)/amp)

def achieved_fraction(y,y0,yf):
    amp = abs(yf - y0)
    if amp < 1e-12: return float("nan")
    if yf >= y0:
        return float((np.nanmax(y) - y0)/amp)
    else:
        return float((y0 - np.nanmin(y))/amp)

def sse_on_hold(t,e,r, dwell=0.5):
    if len(t)==0: return float("nan")
    end = t[-1]; start = max(t[0], end - dwell)
    m = (t >= start)
    if np.sum(m) < 3: return float("nan")
    if np.nanstd(r[m]) > 1e-3:  # not a flat hold
        return float("nan")
    return float(np.nanmean(np.abs(e[m])))

# ------- locate files (RL naming) -------
speed_path   = pick_exact_or_glob(["eval_time_speed_radps.csv", "*speed*radps*.csv"])
voltage_path = pick_exact_or_glob(["eval_time_voltage.csv", "*voltage*.csv", "*u*.csv"])

if not (speed_path and voltage_path):
    sys.exit("ERROR: Could not find eval_time_speed_radps.csv and eval_time_voltage.csv")

# load
t_s, w, cols_s = load_series(speed_path, SYN_W)
# try to get reference from same file (3rd col) or infer
df_s = read_csv_flexible(speed_path)
wr_name = col(df_s, SYN_WR)
if wr_name is None and df_s.shape[1] >= 3:
    wr_name = df_s.columns[2]
w_ref = df_s[wr_name].to_numpy() if wr_name else None
if w_ref is not None and len(w_ref) == len(df_s):
    # align ref to t_s
    tcol = col(df_s, SYN_T) or df_s.columns[0]
    w_ref = np.interp(t_s, df_s[tcol].to_numpy(), w_ref)
else:
    w_ref = None  # will segment by median-based change if ref missing

t_u, u, _ = load_series(voltage_path, SYN_U)
u = np.interp(t_s, t_u, u)

# If no ref column, build a plateau-like reference from a moving median
if w_ref is None:
    med = pd.Series(w).rolling(200, min_periods=1, center=True).median().to_numpy()
    w_ref = med

# segment
segs = detect_segments_from_ref(t_s, w_ref, args.delta_ref_min)
if args.force_1s:
    segs = force_one_second_segments(t_s, segs)

# compute per segment
rows=[]
for i0,i1 in segs:
    tt = t_s[i0:i1+1]; y = w[i0:i1+1]; r = w_ref[i0:i1+1]; uu = u[i0:i1+1]
    m = np.isfinite(tt) & np.isfinite(y) & np.isfinite(r) & np.isfinite(uu)
    tt, y, r, uu = tt[m], y[m], r[m], uu[m]
    if len(tt) < 5: continue

    y0_ref, yf_ref = float(r[0]), float(r[-1])
    e = r - y

    tr_10_90 = t_rise_10_90(tt, y, y[0], yf_ref)
    Mp_pct   = overshoot_signed_pct(y, y[0], yf_ref)
    ach_frac = achieved_fraction(y, y[0], yf_ref)

    IAE = trapz_nan(np.abs(e), tt)
    ISE = trapz_nan(e**2, tt)
    SSE = sse_on_hold(tt, e, r, dwell=args.dwell)

    rows.append(dict(
        Segment=f"{tt[0]:.2f}–{tt[-1]:.2f}s",
        t_r_10_90_s=tr_10_90,
        Overshoot_pct=Mp_pct,
        Achieved_frac=ach_frac,
        SSE_rad_per_s=SSE,
        IAE=IAE,
        ISE=ISE,
        RMS_u_V=rms_nan(uu),
        Time_at_rail=time_at_rail(uu)
    ))

df = pd.DataFrame(rows)

# save
out_csv = os.path.join(ROOT, "rl_metrics.csv")
out_tex = os.path.join(ROOT, "rl_metrics_table.tex")
df.to_csv(out_csv, index=False)

df_tex = df.rename(columns={
    "Segment":"Segment",
    "t_r_10_90_s":"$t_r^{10\\text{--}90}$ [s]",
    "Overshoot_pct":"$M_p$ [\\%]",
    "Achieved_frac":"Achieved frac.",
    "SSE_rad_per_s":"SSE [rad/s]",
    "IAE":"IAE",
    "ISE":"ISE",
    "RMS_u_V":"RMS$(u)$ [V]",
    "Time_at_rail":"Time@rail"
})
latex = df_tex.to_latex(index=False, float_format=lambda x: f"{x:.3g}",
                        column_format="lcccccccc",
                        caption="RL (TD3) step–segment metrics on the full–run scenario.",
                        label="tab:rl_metrics")
with open(out_tex, "w", encoding="utf-8") as f:
    f.write(latex)

print("Saved:\n ", out_csv, "\n ", out_tex)
print("\nPreview:")
print(df.to_string(index=False))