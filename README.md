# Motor Control Benchmark Suite

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)](pyproject.toml)
[![NumPy](https://img.shields.io/badge/NumPy-required-blue)](pyproject.toml)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-required-blue)](pyproject.toml)
[![Gymnasium RL](https://img.shields.io/badge/Gymnasium-RL-green)](pyproject.toml)
[![Stable-Baselines3 TD3](https://img.shields.io/badge/Stable--Baselines3-TD3-green)](docs/usage.md)
[![Research prototype](https://img.shields.io/badge/Status-research_prototype-orange)](#scope)
[![MIT](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

Simulate brushed DC and permanent-magnet synchronous motor (PMSM) drives with
classical PI/FOC and **TD3** controllers. The project covers current and speed
tracking with continuous motor dynamics, actuator constraints, and seeded
reference profiles.

## Quick start

Use Python 3.12. From the repository directory:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .

td3-control baseline --motor dc --task speed
td3-control baseline --motor pmsm --task current
```

The baseline commands need no checkpoint or PyTorch. Each writes a tracking
figure, raw signals, metrics and settings under `outputs/`. Add `--show` to
display the figure as well.

## Train and evaluate TD3

```sh
python -m pip install '.[rl]'
td3-control train --motor dc --task current --run smoke --steps 128
td3-control eval --motor dc --task current --checkpoint models/dc/current/smoke/model.zip
```

The 128-transition run is a software smoke test, not a trained research result.
See the [usage guide](docs/usage.md) for all motor/task combinations, rotor modes,
PMSM disturbances, historical checkpoints, and the output format.

## Repository layout

```text
src/td3_drive_control/
  dc/           Brushed DC dynamics, PI and TD3 environments
  pmsm/         dq dynamics, FOC and TD3 environments
  cli.py        Single command-line entry point
  experiments.py, training.py, evaluation.py
tests/         Determinism, simulation and CLI checks
docs/          Usage and experiment scope
.github/       Wheel build and smoke-test workflow
```

## Scope

This is the original **TD3 study**, not the separate
[PPO and ICM project](https://github.com/Laxi03/ppo-electric-drive-control).
The simulators and controller assumptions differ across the historical cases;
the repository does not claim that independently generated PI/FOC and TD3
traces are matched trials. It is not a hardware controller or a safety proof.

Checkpoints, vendor PDFs and generated images are not in the current source
tree. The previous research snapshot remains in Git history at
[`da90b3e`](https://github.com/Laxi03/DC_PMSM_simulation/tree/da90b3e32a5f941826bf60c2b1a37aeb5739b3da).

See [contributing](CONTRIBUTING.md), [citation](CITATION.cff), and the [MIT license](LICENSE).
