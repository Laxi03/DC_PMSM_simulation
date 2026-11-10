# Running the TD3 motor-control experiments

Use Python 3.12 and install with `python -m pip install -e .`. To train or
evaluate TD3, use `python -m pip install -e '.[rl]'`. On Linux, you can install
the CPU-only build of PyTorch first:

```sh
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
```

The command `python -m td3_drive_control` is equivalent to `td3-control`.
Use `--help` or `<subcommand> --help` to view supported options.

## PI and FOC baselines

```sh
td3-control baseline --motor dc --task speed --reference 150
td3-control baseline --motor dc --task current --reference 0.5
td3-control baseline --motor pmsm --task speed --reference 300
td3-control baseline --motor pmsm --task current --control-mode sinus_speed
```

Reference units are rad/s for speed and A for current (q-axis current for
PMSM). Use `--random-reference --seed 123` for a repeatable baseline step
profile, or `--duration 2` for a shorter simulation. Current tasks default
to a locked rotor; PMSM supports `free` and `sinus_speed`, and the DC PI
baseline also supports `free`. The original gain and plant options are
available via the specialist baseline modules:

```sh
python -m td3_drive_control.dc.pi --help
python -m td3_drive_control.pmsm.foc --help
```

## TD3 training

```sh
td3-control train --motor dc --task current --run dc-current-123
td3-control train --motor dc --task speed --run dc-speed-123
td3-control train --motor pmsm --task current --run pmsm-current-123 --control-mode locked
td3-control train --motor pmsm --task speed --run pmsm-speed-123
```

The original step budgets are 1 million transitions per task except PMSM
speed (2 million). Use `--steps` for a shorter run and `--workers` for
parallel environments. Defaults are CPU and one worker; the old scripts used
eight. Seeds and actual sampled steps are written to `manifest.json`.
A run name cannot overwrite an existing model. The DC TD3 current
environment uses a locked rotor; PMSM current supports all three modes.

## Evaluation and perturbations

```sh
td3-control eval --motor dc --task speed --checkpoint models/dc/speed/dc-speed-123/model.zip
td3-control eval --motor pmsm --task current \
  --checkpoint models/pmsm/current/pmsm-current-123/model.zip \
  --control-mode locked --perturbation R --relative-change 0.3 --perturbation-time 2
```

PMSM can change `R`, `Ld`, `Lq`, `lam`, or `Vdc` at a fixed time. FOC PI
gains remain fixed, but its model-based terms use active plant parameters,
as in the original baseline. The PMSM TD3 speed environment retains its
internal PI reference generator. These are historical design choices, not
proof of an apples-to-apples robustness comparison.

Evaluate only **trusted** checkpoint ZIPs. Select the exact motor and task:
some checkpoint pairs have equal observation/action dimensions but different
feature meanings. New checkpoints include a task manifest; legacy ZIPs do
not. `--duration` on evaluation caps the rollout and marks it as a preview
if it did not reach the normal episode end.

### Previous checkpoints

The original snapshot's `dc_rl_current.zip`, `dc_rl_speed.zip`, and
`pmsm_rl_current.zip` correspond to the same motor/tasks in this package.
The earlier `pmsm_rl_current_control.zip` and `pmsm_rl_tracking.zip` have
different observation/action shapes. They are not substitutes for the current
PMSM environments; use the historical code that trained them or train a new
policy. The previous snapshot does not include a matching 11-observation PMSM
speed checkpoint.

## Outputs

Set `--workspace` to control where new artifacts are written (default:
current directory). Paths are never relative to the installed package.

- `models/<motor>/<task>/<run>/model.zip` and `manifest.json`
- `outputs/<motor>/<task>/<controller>_<timestamp>/tracking.png`
- `timeseries.csv`, `metrics.json`, and `manifest.json` in the same output folder

MAE and RMSE are in the tracking signal's physical unit. RMS/peak current
use armature current for DC and the dq magnitude for PMSM. The raw trace
includes dq components and applied plant parameters for PMSM. Independent
baseline and learned rollouts can use different reference profiles and sample
periods; do not rank them as if they were paired.
