# PMSM Simulation Command Cheat Sheet

This file collects the main commands for all 3 codes:

1. **RL current control**
2. **RL speed control**
3. **FOC / PI baseline**

---

## Table of Contents

- [1. RL current control](#1-rl-current-control)
- [2. RL speed control](#2-rl-speed-control)
- [3. FOC / PI control](#3-foc--pi-control)
- [Quick summary: which code is for what?](#quick-summary-which-code-is-for-what)
- [Fair comparison rule](#fair-comparison-rule)
- [Recommended test bundle](#recommended-test-bundle)

---

## 1. RL current control

Example file name:

```bash
sim_PMSM_RL_Current.py
```

This version now supports 3 current-control modes:

- `locked`
- `free`
- `sinus_speed`

It also supports separate saliency-related inductances:

- `Ld`
- `Lq`

### 1.1 Training, locked rotor
Trains the RL current controller with locked rotor.

```bash
python3 sim_PMSM_RL_Current.py --train --control_mode locked
```

### 1.2 Training, locked rotor + seed
Locked rotor and reproducible training randomization.

```bash
python3 sim_PMSM_RL_Current.py --train --control_mode locked --seed 42
```

### 1.3 Training, free rotor
Same training, but with free rotor.

```bash
python3 sim_PMSM_RL_Current.py --train --control_mode free
```

### 1.4 Training, free rotor + seed
Free rotor and reproducible training.

```bash
python3 sim_PMSM_RL_Current.py --train --control_mode free --seed 42
```

### 1.5 Training, sinus speed mode
Current control while the motor receives a prescribed one-period sinusoidal speed profile.

```bash
python3 sim_PMSM_RL_Current.py --train --control_mode sinus_speed --sinus_w_amp 300
```

### 1.6 Training, sinus speed mode + seed
Sinus-speed current control with reproducible training.

```bash
python3 sim_PMSM_RL_Current.py --train --control_mode sinus_speed --sinus_w_amp 300 --seed 42
```

### 1.7 Nominal evaluation, locked rotor
Loads the trained model and tests it under nominal locked-rotor conditions.

```bash
python3 sim_PMSM_RL_Current.py --eval --control_mode locked
```

### 1.8 Nominal evaluation, locked rotor + seed
Same reference profile and same evaluation domain.

```bash
python3 sim_PMSM_RL_Current.py --eval --control_mode locked --seed 42
```

### 1.9 Nominal evaluation, free rotor
Free-rotor current RL test.

```bash
python3 sim_PMSM_RL_Current.py --eval --control_mode free
```

### 1.10 Nominal evaluation, free rotor + seed
Free rotor and reproducible reference.

```bash
python3 sim_PMSM_RL_Current.py --eval --control_mode free --seed 42
```

### 1.11 Nominal evaluation, sinus speed mode
Current control benchmark under prescribed sinusoidal speed profile.

```bash
python3 sim_PMSM_RL_Current.py --eval --control_mode sinus_speed --sinus_w_amp 300
```

### 1.12 Nominal evaluation, sinus speed mode + seed
Same sinus-speed current-control benchmark with reproducible profile.

```bash
python3 sim_PMSM_RL_Current.py --eval --control_mode sinus_speed --sinus_w_amp 300 --seed 42
```

### 1.13 Evaluation with R perturbation, random time
Changes **R** during runtime.

```bash
python3 sim_PMSM_RL_Current.py --eval --control_mode locked --seed 42 --perturb --perturb_target R --perturb_change 0.3 --perturb_random_time
```

### 1.14 Evaluation with R perturbation, fixed time
Applies the **R** change at an exact chosen time.

```bash
python3 sim_PMSM_RL_Current.py --eval --control_mode locked --seed 42 --perturb --perturb_target R --perturb_change 0.3 --perturb_time 2.0
```

### 1.15 Evaluation with Ld perturbation
d-axis inductance change test.

```bash
python3 sim_PMSM_RL_Current.py --eval --control_mode locked --seed 42 --perturb --perturb_target Ld --perturb_change 0.5 --perturb_time 2.0
```

### 1.16 Evaluation with Lq perturbation
q-axis inductance change test.

```bash
python3 sim_PMSM_RL_Current.py --eval --control_mode locked --seed 42 --perturb --perturb_target Lq --perturb_change 0.5 --perturb_time 2.0
```

### 1.17 Evaluation with lam perturbation
Flux-linkage change test.

```bash
python3 sim_PMSM_RL_Current.py --eval --control_mode locked --seed 42 --perturb --perturb_target lam --perturb_change -0.2 --perturb_time 2.0
```

### 1.18 Evaluation with Vdc perturbation
DC bus voltage change test.

```bash
python3 sim_PMSM_RL_Current.py --eval --control_mode locked --seed 42 --perturb --perturb_target Vdc --perturb_change -0.15 --perturb_time 2.0
```

### 1.19 Sinus-speed evaluation with Lq perturbation
Harder current-control benchmark: prescribed sinusoidal speed profile plus `Lq` change.

```bash
python3 sim_PMSM_RL_Current.py --eval --control_mode sinus_speed --sinus_w_amp 300 --seed 42 --perturb --perturb_target Lq --perturb_change 0.4 --perturb_time 2.0
```

---

## 2. RL speed control

Example file name:

```bash
sim_PMSM_RL_Speed.py
```

This version now supports separate saliency-related inductances:

- `Ld`
- `Lq`

It also uses internal PI-generated current references inside the observation / shaping setup.

### 2.1 Training
Trains the RL speed controller.

```bash
python3 sim_PMSM_RL_Speed.py --train
```

### 2.2 Training with fixed seed
Reproducible training.

```bash
python3 sim_PMSM_RL_Speed.py --train --seed 42
```

### 2.3 Training with different load torque
Trains the speed RL under a different load.

```bash
python3 sim_PMSM_RL_Speed.py --train --seed 42 --Tload 0.003
```

### 2.4 Training with custom internal PI reference generator
Changes the internal PI speed loop that generates `id_ref_pi` and `iq_ref_pi`.

```bash
python3 sim_PMSM_RL_Speed.py --train --seed 42 --Kp_w_pi 0.03 --Ki_w_pi 0.03 --base_speed 500
```

### 2.5 Nominal evaluation
Loads the trained speed RL model and tests it under nominal conditions.

```bash
python3 sim_PMSM_RL_Speed.py --eval
```

### 2.6 Nominal evaluation with seed
Same speed reference profile.

```bash
python3 sim_PMSM_RL_Speed.py --eval --seed 42
```

### 2.7 Evaluation with different load torque
Tests the trained speed RL under a different load.

```bash
python3 sim_PMSM_RL_Speed.py --eval --seed 42 --Tload 0.003
```

### 2.8 Evaluation with R perturbation, random time
Runtime **R** change during speed control.

```bash
python3 sim_PMSM_RL_Speed.py --eval --seed 42 --perturb --perturb_target R --perturb_change 0.3 --perturb_random_time
```

### 2.9 Evaluation with R perturbation, fixed time
Applies an **R** step at a chosen time.

```bash
python3 sim_PMSM_RL_Speed.py --eval --seed 42 --perturb --perturb_target R --perturb_change 0.3 --perturb_time 2.0
```

### 2.10 Evaluation with Ld perturbation
d-axis inductance change during speed RL.

```bash
python3 sim_PMSM_RL_Speed.py --eval --seed 42 --perturb --perturb_target Ld --perturb_change 0.5 --perturb_time 2.0
```

### 2.11 Evaluation with Lq perturbation
q-axis inductance change during speed RL.

```bash
python3 sim_PMSM_RL_Speed.py --eval --seed 42 --perturb --perturb_target Lq --perturb_change 0.5 --perturb_time 2.0
```

### 2.12 Evaluation with lam perturbation
Flux-linkage change during speed RL.

```bash
python3 sim_PMSM_RL_Speed.py --eval --seed 42 --perturb --perturb_target lam --perturb_change -0.2 --perturb_time 2.0
```

### 2.13 Evaluation with Vdc perturbation
DC bus voltage change during speed RL.

```bash
python3 sim_PMSM_RL_Speed.py --eval --seed 42 --perturb --perturb_target Vdc --perturb_change -0.15 --perturb_time 2.0
```

### 2.14 Evaluation with custom internal PI reference generator
Same RL speed benchmark but with modified PI-reference generator settings.

```bash
python3 sim_PMSM_RL_Speed.py --eval --seed 42 --Kp_w_pi 0.03 --Ki_w_pi 0.03 --base_speed 500
```

---

## 3. FOC / PI control

Example file name:

```bash
sim_PMSM_FOC.py
```

This is the classical controller / baseline simulation.

This version now supports:

- separate `Ld` and `Lq`
- runtime perturbations in `R`, `Ld`, `Lq`, `lam`, `Vdc`
- current-control modes:
  - `locked`
  - `free`
  - `sinus_speed`

### 3.1 Speed mode, fixed reference
FOC speed control with constant speed reference.

```bash
python3 sim_PMSM_FOC.py --mode speed --w_ref 300
```

### 3.2 Speed mode, random reference
FOC speed control with random speed profile.

```bash
python3 sim_PMSM_FOC.py --mode speed --random_ref
```

### 3.3 Speed mode, random reference + seed
Same speed reference profile.

```bash
python3 sim_PMSM_FOC.py --mode speed --random_ref --seed 42
```

### 3.4 Speed mode, longer run
Longer FOC speed simulation.

```bash
python3 sim_PMSM_FOC.py --mode speed --random_ref --seed 42 --t_end 8.0
```

### 3.5 Speed mode, different load torque
FOC speed control under a different load.

```bash
python3 sim_PMSM_FOC.py --mode speed --random_ref --seed 42 --Tload 0.003
```

### 3.6 Speed mode with Ld/Lq set explicitly
Nominal saliency setup for the benchmark.

```bash
python3 sim_PMSM_FOC.py --mode speed --random_ref --seed 42 --Ld_pp 1.61e-3 --Lq_pp 1.90e-3
```

### 3.7 Current mode, fixed reference, locked
FOC current control with constant **iq** reference and locked rotor.

```bash
python3 sim_PMSM_FOC.py --mode current --current_control_mode locked --iq_ref 0.5
```

### 3.8 Current mode, random reference, locked
FOC current control with random current reference.

```bash
python3 sim_PMSM_FOC.py --mode current --current_control_mode locked --random_ref
```

### 3.9 Current mode, random reference, locked + seed
Same reference profile in current mode.

```bash
python3 sim_PMSM_FOC.py --mode current --current_control_mode locked --random_ref --seed 42
```

### 3.10 Current mode, free rotor
FOC current control with free rotor.

```bash
python3 sim_PMSM_FOC.py --mode current --current_control_mode free --iq_ref 0.5
```

### 3.11 Current mode, sinus speed
FOC current control with imposed one-period sinusoidal speed profile.

```bash
python3 sim_PMSM_FOC.py --mode current --current_control_mode sinus_speed --iq_ref 0.5 --sinus_w_amp 300
```

### 3.12 Current mode, sinus speed + random current reference
Harder current-control case.

```bash
python3 sim_PMSM_FOC.py --mode current --current_control_mode sinus_speed --random_ref --sinus_w_amp 300 --seed 42
```

### 3.13 Speed mode, R perturbation, random time
Runtime **R** change during FOC speed control.

```bash
python3 sim_PMSM_FOC.py --mode speed --random_ref --seed 42 --perturb --perturb_target R --perturb_change 0.3 --perturb_random_time
```

### 3.14 Speed mode, Ld perturbation, fixed time
Runtime **Ld** change during FOC speed control.

```bash
python3 sim_PMSM_FOC.py --mode speed --random_ref --seed 42 --perturb --perturb_target Ld --perturb_change 0.8 --perturb_time 2.0
```

### 3.15 Speed mode, Lq perturbation, fixed time
Runtime **Lq** change during FOC speed control.

```bash
python3 sim_PMSM_FOC.py --mode speed --random_ref --seed 42 --perturb --perturb_target Lq --perturb_change 0.8 --perturb_time 2.0
```

### 3.16 Speed mode, lam perturbation
Runtime **lam** change during FOC speed control.

```bash
python3 sim_PMSM_FOC.py --mode speed --random_ref --seed 42 --perturb --perturb_target lam --perturb_change -0.2 --perturb_time 2.0
```

### 3.17 Speed mode, Vdc perturbation
Runtime **Vdc** change during FOC speed control.

```bash
python3 sim_PMSM_FOC.py --mode speed --random_ref --seed 42 --perturb --perturb_target Vdc --perturb_change -0.2 --perturb_time 2.0
```

### 3.18 Current mode, locked, R perturbation
Runtime **R** change during FOC current control.

```bash
python3 sim_PMSM_FOC.py --mode current --current_control_mode locked --random_ref --seed 42 --perturb --perturb_target R --perturb_change 0.3 --perturb_random_time
```

### 3.19 Current mode, locked, Ld perturbation
Runtime **Ld** change during FOC current control.

```bash
python3 sim_PMSM_FOC.py --mode current --current_control_mode locked --random_ref --seed 42 --perturb --perturb_target Ld --perturb_change 0.5 --perturb_time 2.0
```

### 3.20 Current mode, locked, Lq perturbation
Runtime **Lq** change during FOC current control.

```bash
python3 sim_PMSM_FOC.py --mode current --current_control_mode locked --random_ref --seed 42 --perturb --perturb_target Lq --perturb_change 0.5 --perturb_time 2.0
```

### 3.21 Current mode, locked, Vdc perturbation
Runtime **Vdc** change during FOC current control.

```bash
python3 sim_PMSM_FOC.py --mode current --current_control_mode locked --random_ref --seed 42 --perturb --perturb_target Vdc --perturb_change -0.2 --perturb_time 2.0
```

### 3.22 Current mode, sinus-speed benchmark with Lq perturbation
Harder benchmark with imposed sinusoidal speed profile and saliency perturbation.

```bash
python3 sim_PMSM_FOC.py --mode current --current_control_mode sinus_speed --random_ref --sinus_w_amp 300 --seed 42 --perturb --perturb_target Lq --perturb_change 0.5 --perturb_time 2.0
```

---

## Quick summary: which code is for what?

### RL current
Use this when you want to evaluate the RL as a **current controller**.

Main commands:

```bash
python3 sim_PMSM_RL_Current.py --train --control_mode locked --seed 42
python3 sim_PMSM_RL_Current.py --eval --control_mode locked --seed 42
python3 sim_PMSM_RL_Current.py --eval --control_mode locked --seed 42 --perturb --perturb_target R --perturb_change 0.3 --perturb_random_time
```

Harder current-control benchmark:

```bash
python3 sim_PMSM_RL_Current.py --train --control_mode sinus_speed --sinus_w_amp 300 --seed 42
python3 sim_PMSM_RL_Current.py --eval --control_mode sinus_speed --sinus_w_amp 300 --seed 42
python3 sim_PMSM_RL_Current.py --eval --control_mode sinus_speed --sinus_w_amp 300 --seed 42 --perturb --perturb_target Lq --perturb_change 0.4 --perturb_time 2.0
```

### RL speed
Use this when you want to evaluate the RL as a **speed controller**.

Main commands:

```bash
python3 sim_PMSM_RL_Speed.py --train --seed 42
python3 sim_PMSM_RL_Speed.py --eval --seed 42
python3 sim_PMSM_RL_Speed.py --eval --seed 42 --perturb --perturb_target R --perturb_change 0.3 --perturb_random_time
```

With saliency perturbation:

```bash
python3 sim_PMSM_RL_Speed.py --eval --seed 42 --perturb --perturb_target Ld --perturb_change 0.5 --perturb_time 2.0
python3 sim_PMSM_RL_Speed.py --eval --seed 42 --perturb --perturb_target Lq --perturb_change 0.5 --perturb_time 2.0
```

### FOC
Use this when you need the classical PI/FOC baseline for comparison.

Main commands:

```bash
python3 sim_PMSM_FOC.py --mode speed --random_ref --seed 42
python3 sim_PMSM_FOC.py --mode speed --random_ref --seed 42 --perturb --perturb_target R --perturb_change 0.3 --perturb_random_time

python3 sim_PMSM_FOC.py --mode current --current_control_mode locked --random_ref --seed 42
python3 sim_PMSM_FOC.py --mode current --current_control_mode locked --random_ref --seed 42 --perturb --perturb_target R --perturb_change 0.3 --perturb_random_time
```

Harder current-control FOC benchmark:

```bash
python3 sim_PMSM_FOC.py --mode current --current_control_mode sinus_speed --random_ref --sinus_w_amp 300 --seed 42
python3 sim_PMSM_FOC.py --mode current --current_control_mode sinus_speed --random_ref --sinus_w_amp 300 --seed 42 --perturb --perturb_target Lq --perturb_change 0.5 --perturb_time 2.0
```

---

## Fair comparison rule

Always use the same seed for nominal and perturbed runs.

Example:

```bash
python3 sim_PMSM_RL_Speed.py --eval --seed 42
python3 sim_PMSM_RL_Speed.py --eval --seed 42 --perturb --perturb_target R --perturb_change 0.3 --perturb_random_time
```

This ensures:
- same reference profile
- same domain
- only the perturbation is different

The same applies to RL current:

```bash
python3 sim_PMSM_RL_Current.py --eval --control_mode sinus_speed --sinus_w_amp 300 --seed 42
python3 sim_PMSM_RL_Current.py --eval --control_mode sinus_speed --sinus_w_amp 300 --seed 42 --perturb --perturb_target Lq --perturb_change 0.4 --perturb_time 2.0
```

And to FOC:

```bash
python3 sim_PMSM_FOC.py --mode current --current_control_mode sinus_speed --random_ref --sinus_w_amp 300 --seed 42
python3 sim_PMSM_FOC.py --mode current --current_control_mode sinus_speed --random_ref --sinus_w_amp 300 --seed 42 --perturb --perturb_target Lq --perturb_change 0.5 --perturb_time 2.0
```

---

## Recommended test bundle

### Current RL — locked rotor

```bash
python3 sim_PMSM_RL_Current.py --train --control_mode locked --seed 42
python3 sim_PMSM_RL_Current.py --eval --control_mode locked --seed 42
python3 sim_PMSM_RL_Current.py --eval --control_mode locked --seed 42 --perturb --perturb_target R --perturb_change 0.3 --perturb_random_time
python3 sim_PMSM_RL_Current.py --eval --control_mode locked --seed 42 --perturb --perturb_target Ld --perturb_change 0.5 --perturb_time 2.0
python3 sim_PMSM_RL_Current.py --eval --control_mode locked --seed 42 --perturb --perturb_target Lq --perturb_change 0.5 --perturb_time 2.0
```

### Current RL — sinus speed benchmark

```bash
python3 sim_PMSM_RL_Current.py --train --control_mode sinus_speed --sinus_w_amp 300 --seed 42
python3 sim_PMSM_RL_Current.py --eval --control_mode sinus_speed --sinus_w_amp 300 --seed 42
python3 sim_PMSM_RL_Current.py --eval --control_mode sinus_speed --sinus_w_amp 300 --seed 42 --perturb --perturb_target R --perturb_change 0.3 --perturb_random_time
python3 sim_PMSM_RL_Current.py --eval --control_mode sinus_speed --sinus_w_amp 300 --seed 42 --perturb --perturb_target Lq --perturb_change 0.4 --perturb_time 2.0
```

### Current RL — free rotor

```bash
python3 sim_PMSM_RL_Current.py --train --control_mode free --seed 42
python3 sim_PMSM_RL_Current.py --eval --control_mode free --seed 42
python3 sim_PMSM_RL_Current.py --eval --control_mode free --seed 42 --perturb --perturb_target R --perturb_change 0.3 --perturb_random_time
```

### Speed RL

```bash
python3 sim_PMSM_RL_Speed.py --train --seed 42
python3 sim_PMSM_RL_Speed.py --eval --seed 42
python3 sim_PMSM_RL_Speed.py --eval --seed 42 --perturb --perturb_target R --perturb_change 0.3 --perturb_random_time
python3 sim_PMSM_RL_Speed.py --eval --seed 42 --perturb --perturb_target Ld --perturb_change 0.5 --perturb_time 2.0
python3 sim_PMSM_RL_Speed.py --eval --seed 42 --perturb --perturb_target Lq --perturb_change 0.5 --perturb_time 2.0
python3 sim_PMSM_RL_Speed.py --eval --seed 42 --perturb --perturb_target Vdc --perturb_change -0.15 --perturb_time 2.0
```

### FOC baseline

```bash
python3 sim_PMSM_FOC.py --mode speed --random_ref --seed 42
python3 sim_PMSM_FOC.py --mode speed --random_ref --seed 42 --perturb --perturb_target R --perturb_change 0.3 --perturb_random_time
python3 sim_PMSM_FOC.py --mode speed --random_ref --seed 42 --perturb --perturb_target Ld --perturb_change 0.8 --perturb_time 2.0
python3 sim_PMSM_FOC.py --mode speed --random_ref --seed 42 --perturb --perturb_target Lq --perturb_change 0.8 --perturb_time 2.0

python3 sim_PMSM_FOC.py --mode current --current_control_mode locked --random_ref --seed 42
python3 sim_PMSM_FOC.py --mode current --current_control_mode locked --random_ref --seed 42 --perturb --perturb_target R --perturb_change 0.3 --perturb_random_time
python3 sim_PMSM_FOC.py --mode current --current_control_mode sinus_speed --random_ref --sinus_w_amp 300 --seed 42
python3 sim_PMSM_FOC.py --mode current --current_control_mode sinus_speed --random_ref --sinus_w_amp 300 --seed 42 --perturb --perturb_target Lq --perturb_change 0.5 --perturb_time 2.0
```
