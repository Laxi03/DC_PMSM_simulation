"""Shared armature and mechanical equations for the DC TD3 environments."""
import math
from dataclasses import dataclass


@dataclass
class AMax32Params:
    R: float = 3.99
    L: float = 0.556e-3
    kT: float = 35.2e-3
    Ke: float = 35.2e-3
    J: float = 45.3e-7
    B: float = 1.0e-6
    T_coulomb: float = 0.001506
    T_amb: float = 25.0
    n0_rpm: float = 6460.0

    def derived(self):
        omega0 = self.n0_rpm * 2.0 * math.pi / 60.0
        return {"omega0": omega0}


@dataclass
class MotorState:
    i: float
    omega: float
    theta: float
    Tw: float


class AMax32Motor:
    def __init__(self, p=None):
        self.p = p or AMax32Params()
        self.d = self.p.derived()

    def derivatives(self, t: float, x: MotorState, u_v: float, load_torque: float, lock_rotor: bool = False) -> MotorState:
        p = self.p

        di = (u_v - p.R * x.i - p.Ke * x.omega) / p.L

        if lock_rotor:
            domega = 0.0
            dtheta = 0.0
        else:
            sign = 0.0
            if abs(x.omega) > 1e-6:
                sign = 1.0 if x.omega > 0 else -1.0

            T_fric = p.B * x.omega + p.T_coulomb * sign
            T_em = p.kT * x.i
            domega = (T_em - load_torque - T_fric) / p.J
            dtheta = x.omega

        dTw = 0.0

        return MotorState(di, domega, dtheta, dTw)

    @staticmethod
    def rk4_step(deriv, t: float, dt: float, x: MotorState) -> MotorState:
        k1 = deriv(t, x)
        k2 = deriv(
            t + 0.5 * dt,
            MotorState(
                x.i + 0.5 * dt * k1.i,
                x.omega + 0.5 * dt * k1.omega,
                x.theta + 0.5 * dt * k1.theta,
                x.Tw
            )
        )
        k3 = deriv(
            t + 0.5 * dt,
            MotorState(
                x.i + 0.5 * dt * k2.i,
                x.omega + 0.5 * dt * k2.omega,
                x.theta + 0.5 * dt * k2.theta,
                x.Tw
            )
        )
        k4 = deriv(
            t + dt,
            MotorState(
                x.i + dt * k3.i,
                x.omega + dt * k3.omega,
                x.theta + dt * k3.theta,
                x.Tw
            )
        )

        return MotorState(
            x.i + (dt / 6.0) * (k1.i + 2 * k2.i + 2 * k3.i + k4.i),
            x.omega + (dt / 6.0) * (k1.omega + 2 * k2.omega + 2 * k3.omega + k4.omega),
            x.theta + (dt / 6.0) * (k1.theta + 2 * k2.theta + 2 * k3.theta + k4.theta),
            x.Tw
        )
