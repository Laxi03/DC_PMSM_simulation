# DC_PMSM_simulation
Simulation of Controlled Electric Drives in a Python-Based Environment

Python simulator of a **DC motor** (A-max 32, ~20 W) with optional **PI speed/current control** and **PWM**.

The app runs a live animation and exports time-series data and a static summary figure to `exports/<timestamp>/`.

---

## 🚀 Features

- Physics model: electrical + mechanical + coarse thermal states  
- 4th-order Runge–Kutta (RK4) integration  
- Optional **speed PI** or **current PI** control  
- Optional **PWM** drive (`--pwm`) with average or switching behavior  
- Built-in **step reference** profiles for speed and current  
- Live matplotlib animation + automatic **CSV export** + static **PNG/PDF** plots  
- All parameters adjustable from CLI flags  

---

## 🧩 Requirements

- **Python 3.10+**
- Packages:
  - `numpy`
  - `matplotlib`

Install dependencies:

```bash
python3 -m pip install numpy matplotlib
```

> ℹ️ On macOS, ensure the Tk backend is available:  
> `brew install python-tk`

---

## 🧭 Quick Start

### ▶️ Speed PI + PWM (averaged) – step sequence, 3 s

```bash
python3 sim_test.py --t_end 3   --pwm --avg_pwm --speed_ctrl --ref_profile step_seq   --kps 0.002 --kis 0.4 --ulim 1.0   --V 24 --Tload 0.0
```

### ▶️ Current PI (no PWM) – constant reference, 2 s

```bash
python3 sim_test.py --t_end 2   --current_ctrl --i_ref 0.4 --kpc 1.0 --kic 50   --ulim 24 --V 24 --Tload 0.02
```

After each run, results appear in `exports/<timestamp>/`:
- `live_full.png` and `live_full.pdf` — static full-run plots  
- `live_*.csv` and `live_full_*.csv` — exported time series  

---

## ⚙️ CLI Arguments

### Simulation parameters
| Flag | Type | Default | Description |
|------|------|----------|-------------|
| `--t_end` | float | `2.0` | Simulation duration [s] |
| `--dt` | float | `1e-4` | Integration step [s] |
| `--V` | float | `24.0` | DC supply voltage [V] |
| `--Tload` | float | `0.02` | Constant load torque [N·m] |
| `--spf` | int | `200` | RK4 steps per animation frame |
| `--window` | float | `1.0` | Live plot scroll window [s] |
| `--smooth_ms` | float | `0.0` | Display-only smoothing (ms) |
| `--smooth_cycles` | float | `0.0` | Display smoothing over PWM cycles |
| `--downsample` | int | `8` | Downsample factor for plotting |

---

### PWM options
| Flag | Type | Default | Description |
|------|------|----------|-------------|
| `--pwm` | flag | off | Enable PWM drive |
| `--pwm_freq` | float | `5000.0` | PWM switching frequency [Hz] |
| `--pwm_duty` | float | `0.4` | Fixed duty if not controlled |
| `--avg_pwm` | flag | off | Use averaged PWM ( u = V·duty ) instead of switching |

---

### Speed PI control (ω in rad/s)
| Flag | Type | Default | Description |
|------|------|----------|-------------|
| `--speed_ctrl` | flag | off | Enable speed PI controller |
| `--w_ref` | float | `160.0` | Speed reference [rad/s] for `ref_profile=const` |
| `--ref_profile` | choice | `const` | `const` or `step_seq` reference |
| `--kps` | float | `0.002` | Proportional gain |
| `--kis` | float | `0.4` | Integral gain |
| `--ulim` | float | `24.0` | Output limit (Volts, or ±1.0 duty if `--pwm`) |

---

### Current PI control (i in A)
| Flag | Type | Default | Description |
|------|------|----------|-------------|
| `--current_ctrl` | flag | off | Enable current PI controller |
| `--i_ref` | float | `0.4` | Current reference [A] for `i_ref_profile=const` |
| `--i_ref_profile` | choice | `const` | `const` or `step_seq` |
| `--kpc` | float | `1.0` | Proportional gain |
| `--kic` | float | `50.0` | Integral gain |

> **Controller behavior**  
> - With `--pwm --avg_pwm`, the controller output is **duty ∈ [−1, 1]**, limited by `--ulim`.  
> - Without PWM, the controller output is **voltage**, limited to ±`ulim` [V].

---

## 📊 Outputs

**During simulation**
- Live scrolling plots: **u [V]**, **i [A]**, **ω [rad/s]**, and **torques** ( T_em & T_load )

**After completion (`exports/<timestamp>/`)**
- `live_full.png` and `live_full.pdf` — static full-run figure  
- CSVs:
  - `live_u.csv`, `live_i.csv`, `live_omega.csv`
  - `live_i_ref.csv`, `live_omega_ref.csv` (if relevant)
  - `live_torque.csv` (T_em, T_load)
  - downsampled `live_full_*.csv` versions  

---

## 🧠 Tuning Tips

| Goal | Suggested change |
|------|------------------|
| Faster rise / tighter tracking | Increase `--kps` to `0.003`, `--kis` to `0.6` |
| Better load disturbance rejection | Increase `--kis`; add `--Tload 0.02` |
| Smoother PWM visualization | Add `--pwm --avg_pwm`, `--ulim 1.0` |

---

## ⚠️ Limitations

- Thermal dynamics are coarse — illustrative only, not for thermal design.
- The live animation requires the **TkAgg** or **MacOSX** backend.

---

## 📄 License

MIT License

---

## 🙏 Acknowledgements

A-max 32 nominal parameters are used as reference configuration.  
All quantities are expressed in **SI units**.