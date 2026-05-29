"""
Space4Bar: spatial four-bar ankle kinematics for CASBOT-02.

Strict Python translation of space4bar_cas02.cpp / .hpp.
Conventions
-----------
  serial  x = [pitch, roll]         (串联等效关节)
  parallel q = [qUp, qDown]         (并联驱动关节)
  Left leg:  q(0) = right-side rod,  q(1) = left-side rod
  Right leg: q(0) = left-side rod,   q(1) = right-side rod

Jacobian J = dq/dx
  velocity:  q_v  = J   * x_v        (s2p)
             x_v  = J^-1 * q_v       (p2s)
  torque:    q_tau = J^-T * x_tau     (s2p)
             x_tau = J^T  * q_tau     (p2s)
"""

import math
import numpy as np
from numpy.linalg import svd

_EPS = 1e-12


def _clamp_unit(x: float) -> float:
    if x > 1.0:
        return 1.0
    if x < -1.0:
        return -1.0
    return x


def _safe_acos_ratio(num: float, den: float) -> float:
    if abs(den) < _EPS:
        return 0.0 if num >= 0.0 else math.pi
    return math.acos(_clamp_unit(num / den))


def _damped_pinv_2x2(A: np.ndarray, lam: float = 1e-6) -> np.ndarray:
    U, S, Vt = svd(A)
    S_pinv = np.diag([s / (s * s + lam * lam) for s in S])
    return Vt.T @ S_pinv @ U.T


def _robust_solve_2x2(A: np.ndarray, b: np.ndarray,
                       sigma_min_th: float = 1e-8,
                       lam: float = 1e-6) -> np.ndarray:
    _, S, _ = svd(A)
    if S[-1] > sigma_min_th:
        return np.linalg.solve(A, b)
    return _damped_pinv_2x2(A, lam) @ b


def _robust_left_solve_2x2(A: np.ndarray, B: np.ndarray,
                            sigma_min_th: float = 1e-8,
                            lam: float = 1e-6) -> np.ndarray:
    """Solve A X = B for X (2x2)."""
    _, S, _ = svd(A)
    if S[-1] > sigma_min_th:
        return np.linalg.solve(A, B)
    return _damped_pinv_2x2(A, lam) @ B


class Space4Bar:
    def __init__(self):
        self._e1 = np.array([1.0, 0.0, 0.0])
        self.initialAngle = 0.2838
        self.bar = 25.0
        self.rod_1 = 338.0
        self.rod_2 = 247.0
        self.x_ = -35.0
        self.kR = 25e-3
        self.kL = 50.0 / 2.0 * 1e-3
        self.kRL = self.kR / self.kL

    # ------------------------------------------------------------------ #
    #  Approximate FK (initial guess for Newton)
    # ------------------------------------------------------------------ #
    def left4BarFK(self, q: np.ndarray) -> np.ndarray:
        qm5 = -q[0]
        qm6 = q[1]
        tan_tr = self.kRL * math.sin(0.5 * (qm6 - qm5))
        q5 = 0.5 * (qm5 + qm6)
        q6 = math.atan(tan_tr)
        return np.array([q5, q6])

    def right4BarFK(self, q: np.ndarray) -> np.ndarray:
        qm5 = q[0]
        qm6 = -q[1]
        tan_tr = self.kRL * math.sin(0.5 * (qm5 - qm6))
        q5 = 0.5 * (qm5 + qm6)
        q6 = math.atan(tan_tr)
        return np.array([q5, q6])

    # ------------------------------------------------------------------ #
    #  Exact spatial IK
    # ------------------------------------------------------------------ #
    def rightIkBase(self, x: np.ndarray) -> np.ndarray:
        cp, sp = math.cos(x[0]), math.sin(x[0])
        cr, sr = math.cos(x[1]), math.sin(x[1])

        R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        R_roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

        _r_C1 = np.array([self.x_, self.bar, 0.0])
        _r_C2 = np.array([self.x_, -self.bar, 0.0])
        _r_A1 = np.array([self.x_, 0.0, self.rod_1 - math.sin(self.initialAngle) * self.bar])
        _r_A2 = np.array([self.x_, 0.0, self.rod_2 - math.sin(self.initialAngle) * self.bar])

        r_A1C1 = R_pitch @ R_roll @ _r_C1 - _r_A1
        r_A2C2 = R_pitch @ R_roll @ _r_C2 - _r_A2

        a1, b1, c1 = r_A1C1
        a2, b2, c2 = r_A2C2

        d1 = a1*a1 + b1*b1 + c1*c1 + self.bar**2 - self.rod_1**2
        f1 = 2.0 * self.bar * math.sqrt(b1*b1 + c1*c1)
        d2 = a2*a2 + b2*b2 + c2*c2 + self.bar**2 - self.rod_2**2
        f2 = 2.0 * self.bar * math.sqrt(b2*b2 + c2*c2)

        q = np.zeros(2)
        q[0] = _safe_acos_ratio(d1, f1) + math.atan2(c1, b1) - self.initialAngle
        q[1] = _safe_acos_ratio(-d2, f2) + math.atan2(c2, b2) + self.initialAngle
        return q

    def leftIkBase(self, x: np.ndarray) -> np.ndarray:
        cp, sp = math.cos(x[0]), math.sin(x[0])
        cr, sr = math.cos(x[1]), math.sin(x[1])

        R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        R_roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

        _r_C1 = np.array([self.x_, self.bar, 0.0])
        _r_C2 = np.array([self.x_, -self.bar, 0.0])
        _r_A1 = np.array([self.x_, 0.0, self.rod_2 - math.sin(self.initialAngle) * self.bar])
        _r_A2 = np.array([self.x_, 0.0, self.rod_1 - math.sin(self.initialAngle) * self.bar])

        r_A1C1 = R_pitch @ R_roll @ _r_C1 - _r_A1
        r_A2C2 = R_pitch @ R_roll @ _r_C2 - _r_A2

        a1, b1, c1 = r_A1C1
        a2, b2, c2 = r_A2C2

        d1 = a1*a1 + b1*b1 + c1*c1 + self.bar**2 - self.rod_2**2
        f1 = 2.0 * self.bar * math.sqrt(b1*b1 + c1*c1)
        d2 = a2*a2 + b2*b2 + c2*c2 + self.bar**2 - self.rod_1**2
        f2 = 2.0 * self.bar * math.sqrt(b2*b2 + c2*c2)

        q = np.zeros(2)
        q[1] = _safe_acos_ratio(d1, f1) + math.atan2(c1, b1) - self.initialAngle
        q[0] = _safe_acos_ratio(-d2, f2) + math.atan2(c2, b2) + self.initialAngle
        return q

    # Public IK wrappers
    def left4BarIK(self, x: np.ndarray) -> np.ndarray:
        return self.leftIkBase(x)

    def right4BarIK(self, x: np.ndarray) -> np.ndarray:
        return self.rightIkBase(x)

    # ------------------------------------------------------------------ #
    #  Jacobian  J = dq/dx
    # ------------------------------------------------------------------ #
    def rightJaco(self, q: np.ndarray, x: np.ndarray) -> np.ndarray:
        cp, sp = math.cos(x[0]), math.sin(x[0])
        cr, sr = math.cos(x[1]), math.sin(x[1])

        R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        R_roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

        r_A1B1 = np.array([0.0,
                           math.cos(q[0] + self.initialAngle) * self.bar,
                           math.sin(q[0] + self.initialAngle) * self.bar])
        r_A2B2 = np.array([0.0,
                           -math.cos(-q[1] + self.initialAngle) * self.bar,
                           math.sin(-q[1] + self.initialAngle) * self.bar])

        _r_C1 = np.array([self.x_, self.bar, 0.0])
        _r_C2 = np.array([self.x_, -self.bar, 0.0])
        r_C1 = R_pitch @ R_roll @ _r_C1
        r_C2 = R_pitch @ R_roll @ _r_C2

        _r_A1 = np.array([self.x_, 0.0, self.rod_1 - math.sin(self.initialAngle) * self.bar])
        _r_A2 = np.array([self.x_, 0.0, self.rod_2 - math.sin(self.initialAngle) * self.bar])

        r_B1C1 = r_C1 - (_r_A1 + r_A1B1)
        r_B2C2 = r_C2 - (_r_A2 + r_A2B2)

        J_w = np.vstack([
            np.concatenate([r_B1C1, np.cross(r_C1, r_B1C1)]),
            np.concatenate([r_B2C2, np.cross(r_C2, r_B2C2)]),
        ])

        J_q = np.array([
            [np.dot(self._e1, np.cross(r_A1B1, r_B1C1)), 0.0],
            [0.0, np.dot(self._e1, np.cross(r_A2B2, r_B2C2))],
        ])

        J_wx = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, cp],
            [1.0, 0.0],
            [0.0, -sp],
        ])

        return _robust_left_solve_2x2(J_q, J_w @ J_wx)

    def leftJaco(self, q: np.ndarray, x: np.ndarray) -> np.ndarray:
        cp, sp = math.cos(x[0]), math.sin(x[0])
        cr, sr = math.cos(x[1]), math.sin(x[1])

        R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        R_roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

        r_A1B1 = np.array([0.0,
                           math.cos(q[1] + self.initialAngle) * self.bar,
                           math.sin(q[1] + self.initialAngle) * self.bar])
        r_A2B2 = np.array([0.0,
                           -math.cos(-q[0] + self.initialAngle) * self.bar,
                           math.sin(-q[0] + self.initialAngle) * self.bar])

        _r_C1 = np.array([self.x_, self.bar, 0.0])
        _r_C2 = np.array([self.x_, -self.bar, 0.0])
        r_C1 = R_pitch @ R_roll @ _r_C1
        r_C2 = R_pitch @ R_roll @ _r_C2

        _r_A1 = np.array([self.x_, 0.0, self.rod_2 - math.sin(self.initialAngle) * self.bar])
        _r_A2 = np.array([self.x_, 0.0, self.rod_1 - math.sin(self.initialAngle) * self.bar])

        r_B1C1 = r_C1 - (_r_A1 + r_A1B1)
        r_B2C2 = r_C2 - (_r_A2 + r_A2B2)

        J_w = np.vstack([
            np.concatenate([r_B1C1, np.cross(r_C1, r_B1C1)]),
            np.concatenate([r_B2C2, np.cross(r_C2, r_B2C2)]),
        ])

        J_q = np.array([
            [np.dot(self._e1, np.cross(r_A1B1, r_B1C1)), 0.0],
            [0.0, np.dot(self._e1, np.cross(r_A2B2, r_B2C2))],
        ])

        J_wx = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, cp],
            [1.0, 0.0],
            [0.0, -sp],
        ])

        return _robust_left_solve_2x2(J_q, J_w @ J_wx)

    # ------------------------------------------------------------------ #
    #  Exact FK by Newton iteration
    #
    #  leftJaco rows map to [chain1=q(1), chain2=-q(0)] internal ordering,
    #  NOT the [q(0), q(1)] ordering of leftIkBase output.
    #  rightJaco rows map to [chain1=q(0), chain2=-q(1)].
    #  The residual must be transformed to match before solving.
    # ------------------------------------------------------------------ #
    def leftFkBase(self, q: np.ndarray) -> np.ndarray:
        fallback = self.left4BarFK(q)
        x = fallback.copy()
        for _ in range(100):
            qk = self.leftIkBase(x)
            J = self.leftJaco(qk, x)
            qd = q - qk
            if not (np.all(np.isfinite(J)) and np.all(np.isfinite(qd))):
                return fallback
            xd = _robust_solve_2x2(J, qd[::-1])
            if not np.all(np.isfinite(xd)):
                return fallback
            norm = np.linalg.norm(xd)
            if norm > 0.2:
                xd *= 0.2 / norm
            x += xd
            if np.linalg.norm(qd) <= 1e-5:
                return x
        return fallback

    def rightFkBase(self, q: np.ndarray) -> np.ndarray:
        fallback = self.right4BarFK(q)
        x = fallback.copy()
        for _ in range(100):
            qk = self.rightIkBase(x)
            J = self.rightJaco(qk, x)
            qd = q - qk
            if not (np.all(np.isfinite(J)) and np.all(np.isfinite(qd))):
                return fallback
            xd = _robust_solve_2x2(J, qd)
            if not np.all(np.isfinite(xd)):
                return fallback
            norm = np.linalg.norm(xd)
            if norm > 0.2:
                xd *= 0.2 / norm
            x += xd
            if np.linalg.norm(qd) <= 1e-5:
                return x
        return fallback

    # ------------------------------------------------------------------ #
    #  Velocity mapping   J = dq/dx
    #    s2p:  q_v  = J * x_v
    #    p2s:  x_v  = J^-1 * q_v
    # ------------------------------------------------------------------ #
    def leftP2SVel(self, q: np.ndarray, q_v: np.ndarray) -> np.ndarray:
        x = self.leftFkBase(q)
        J = self.leftJaco(q, x)
        return _robust_solve_2x2(J, q_v[::-1])

    def rightP2SVel(self, q: np.ndarray, q_v: np.ndarray) -> np.ndarray:
        x = self.rightFkBase(q)
        J = self.rightJaco(q, x)
        return _robust_solve_2x2(J, q_v)

    def leftS2PVel(self, x: np.ndarray, x_v: np.ndarray) -> np.ndarray:
        q = self.leftIkBase(x)
        J = self.leftJaco(q, x)
        return (J @ x_v)[::-1]

    def rightS2PVel(self, x: np.ndarray, x_v: np.ndarray) -> np.ndarray:
        q = self.rightIkBase(x)
        J = self.rightJaco(q, x)
        return J @ x_v

    # ------------------------------------------------------------------ #
    #  Torque mapping   q_v = J x_v
    #    p2s:  x_tau = J^T  * q_tau
    #    s2p:  q_tau = J^-T * x_tau
    # ------------------------------------------------------------------ #
    def leftP2STorque(self, q: np.ndarray, q_tau: np.ndarray) -> np.ndarray:
        x = self.leftFkBase(q)
        J = self.leftJaco(q, x)
        return J.T @ q_tau[::-1]

    def rightP2STorque(self, q: np.ndarray, q_tau: np.ndarray) -> np.ndarray:
        x = self.rightFkBase(q)
        J = self.rightJaco(q, x)
        return J.T @ q_tau

    def leftS2PTorque(self, x: np.ndarray, x_tau: np.ndarray) -> np.ndarray:
        q = self.leftIkBase(x)
        J = self.leftJaco(q, x)
        return _robust_solve_2x2(J.T, x_tau)[::-1]

    def rightS2PTorque(self, x: np.ndarray, x_tau: np.ndarray) -> np.ndarray:
        q = self.rightIkBase(x)
        J = self.rightJaco(q, x)
        return _robust_solve_2x2(J.T, x_tau)
