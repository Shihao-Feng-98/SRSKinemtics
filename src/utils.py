import numpy as np
from enum import Enum, auto

K_EPS_LARGE = 1e-4
K_EPS = 1e-6
K_EPS_SMALL = 1e-8

def dh_transform(a, alpha, d, theta) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st*ca,  st*sa, a*ct],
                    [st,  ct*ca, -ct*sa, a*st],
                    [0.,     sa,     ca,    d],
                    [0.,     0.,     0.,   1.]])

def mdh_transform(a, alpha, d, theta) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st,  0., a],
                    [st*ca,  ct*ca, -sa, -d*sa],
                    [st*sa,  ct*sa,  ca,  d*ca],
                    [0.,     0.,     0.,   1.]])

def skew(vec) -> np.ndarray:
    return np.array([[0., -vec[2], vec[1]],
                    [vec[2], 0., -vec[0]],
                    [-vec[1], vec[0], 0.]])


def wrap_to_pi(q):
    if q < -np.pi:
        return q + 2*np.pi
    elif q < np.pi:
        return q
    return q - 2*np.pi

def safe_sqrt(x):
    return np.sqrt(max(x, 0.0))

def safe_acos(x):
    return np.arccos(np.clip(x, -1.0, 1.0))

def safe_asin(x):
    return np.arcsin(np.clip(x, -1.0, 1.0))


class SolutionStatus(Enum):
    NONE = auto()
    FINITE = auto()
    INFINITE = auto()

def solve_sin_cos_eq(a, b, c) -> tuple[SolutionStatus, list]:
    """
    a * sin(psi) + b * cos(psi) + c = 0
    sqrt(a**2 + b**2) * sin(psi + arctan(b,a)) + c = 0
    sqrt(a**2 + b**2) * cos(psi - arctan(a,b)) + c = 0
    """
    solutions = []

    r = np.hypot(a, b)
    if r < K_EPS_SMALL:
        if abs(c) < K_EPS_SMALL:
            return SolutionStatus.INFINITE, []
        return SolutionStatus.NONE, []

    phi = np.arctan2(a, b)
    val = -c / r

    if val > 1.0 + K_EPS_SMALL or val < -1.0 - K_EPS_SMALL:
        return SolutionStatus.NONE, []

    temp = safe_acos(val)

    solutions.append(wrap_to_pi(temp + phi))
    solutions.append(wrap_to_pi(-temp + phi))

    solutions.sort()
    return SolutionStatus.FINITE, solutions