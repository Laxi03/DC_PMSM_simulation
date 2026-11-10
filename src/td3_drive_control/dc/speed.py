"""DC motor speed-tracking environment from the TD3 study."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .plant import AMax32Motor, AMax32Params, MotorState


def generate_speed_profile(duration: float, rng: np.random.Generator):
    n_segments = rng.integers(6, 12)
    durations = rng.uniform(0.4, 1.2, n_segments)
    t_points = np.cumsum(durations)
    t_points = t_points[t_points < duration]

    if len(t_points) == 0 or t_points[-1] < duration:
        t_points = np.concatenate(([0.0], t_points, [duration]))
    else:
        t_points = np.concatenate(([0.0], t_points))

    w_values = rng.uniform(-600.0, 600.0, len(t_points))

    for i in range(1, len(w_values)):
        w_values[i] = 0.6 * w_values[i - 1] + 0.4 * w_values[i]

    return t_points, w_values


def speed_ref(t: float, t_points: np.ndarray, w_values: np.ndarray) -> float:
    idx = np.searchsorted(t_points, t, side="right") - 1
    idx = int(np.clip(idx, 0, len(w_values) - 1))
    return float(w_values[idx])


class DCMotorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        dt=1e-4,
        frame_skip=20,
        Vdc=24.0,
        ep_len_s=8.0,
        tau_act=1e-3,
        load_torque=0.02
    ):
        super().__init__()

        self.dt = dt
        self.frame_skip = frame_skip
        self.ep_len_s = ep_len_s
        self.max_rl_steps = max(1, int(self.ep_len_s / (self.dt * self.frame_skip)))

        self.Vdc = Vdc
        self.tau_act = tau_act
        self.base_load_torque = load_torque

        self.p = AMax32Params()
        self.motor = AMax32Motor(self.p)

        self.state = MotorState(0.0, 0.0, 0.0, self.p.T_amb)

        self.t = 0.0
        self.rl_step_count = 0

        self.u_applied = 0.0
        self.prev_u = 0.0
        self.prev_error = 0.0
        self.error_integral = 0.0

        self.ref_t = None
        self.ref_w = None
        self.last_wref = 0.0

        self.i_abs_max = 6.0
        self.omega_abs_max = 900.0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # [omega_n, i_n, u_n, wref_n, error_n, int_e]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = MotorState(0.0, 0.0, 0.0, self.p.T_amb)
        self.t = 0.0
        self.rl_step_count = 0

        self.u_applied = 0.0
        self.prev_u = 0.0
        self.prev_error = 0.0
        self.error_integral = 0.0

        self.ref_t, self.ref_w = generate_speed_profile(self.ep_len_s, self.np_random)
        self.last_wref = speed_ref(0.0, self.ref_t, self.ref_w)

        return self._obs(), {}

    def _obs(self):
        wref = speed_ref(self.t, self.ref_t, self.ref_w)
        e = (wref - self.state.omega) / self.omega_abs_max

        obs = np.array([
            np.clip(self.state.omega / self.omega_abs_max, -1.0, 1.0),
            np.clip(self.state.i / self.i_abs_max, -1.0, 1.0),
            np.clip(self.prev_u / self.Vdc, -1.0, 1.0),
            np.clip(wref / self.omega_abs_max, -1.0, 1.0),
            np.clip(e, -1.0, 1.0),
            np.clip(self.error_integral, -1.0, 1.0)
        ], dtype=np.float32)

        return obs

    def step(self, action):
        duty = float(np.clip(action[0], -1.0, 1.0))
        u_cmd = self.Vdc * duty

        reward = 0.0
        last_terms = None

        for _ in range(self.frame_skip):
            alpha = self.dt / max(self.tau_act, 1e-6)
            self.u_applied += alpha * (u_cmd - self.u_applied)

            self.state = AMax32Motor.rk4_step(
                lambda tau, s: self.motor.derivatives(tau, s, self.u_applied, self.base_load_torque),
                self.t,
                self.dt,
                self.state
            )
            self.t += self.dt

            self.state.i = float(np.clip(self.state.i, -1.2 * self.i_abs_max, 1.2 * self.i_abs_max))
            self.state.omega = float(np.clip(self.state.omega, -1.2 * self.omega_abs_max, 1.2 * self.omega_abs_max))

            wref = speed_ref(self.t, self.ref_t, self.ref_w)
            self.last_wref = wref

            e = (wref - self.state.omega) / self.omega_abs_max
            de = np.clip((e - self.prev_error) / self.dt, -50.0, 50.0)

            self.error_integral += e * self.dt
            self.error_integral = float(np.clip(self.error_integral, -0.5, 0.5))

            u_norm = self.u_applied / self.Vdc
            i_norm = self.state.i / self.i_abs_max
            du = (self.u_applied - self.prev_u) / self.Vdc

            tracking = -20.0 * (e ** 2)
            derivative_pen = -0.0005 * (de ** 2)
            integral_pen = -0.1 * (self.error_integral ** 2)
            effort_pen = -0.01 * (u_norm ** 2)
            current_pen = -0.01 * (i_norm ** 2)
            smooth_pen = -0.01 * (du ** 2)
            snap_bonus = 0.1 if abs(e) < 0.01 else 0.0

            reward_step_raw = (
                tracking +
                derivative_pen +
                integral_pen +
                effort_pen +
                current_pen +
                smooth_pen +
                snap_bonus
            )

            reward_step = float(np.clip(reward_step_raw, -1.0, 0.5))
            reward += reward_step

            self.prev_error = e
            self.prev_u = self.u_applied

            last_terms = {
                "reward_step": reward_step,
                "tracking": tracking,
                "derivative_pen": derivative_pen,
                "integral_pen": integral_pen,
                "effort_pen": effort_pen,
                "current_pen": current_pen,
                "smooth_pen": smooth_pen,
                "snap_bonus": snap_bonus,
                "error": e
            }

        self.rl_step_count += 1
        terminated = False
        truncated = self.rl_step_count >= self.max_rl_steps

        return self._obs(), reward, terminated, truncated, last_terms
