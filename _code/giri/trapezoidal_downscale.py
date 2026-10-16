"""Composite-trapezoid numerical representation of the calibrated HEV response.

This implementation is selectable in step_07; the default remains analytic.
The integrated quantity is the derivative of a risk multiplier, not risk
itself. Parameter assumptions are retained; this is not a new hazard law.
"""
from __future__ import annotations

import numpy as np

from _config import ALPHA_H, ALPHA_E, ALPHA_V
from r_ph_downscale import apply_r_ph_soft_cap

MAX_LOG_STEP = 0.001
LOG_DOMAIN = (-8.0, 8.0)


def trapezoid_response(ratio: float, alpha: float, max_step: float = MAX_LOG_STEP) -> float:
    """Return 1 plus signed trapezoid areas under s(u)=alpha*exp(alpha*u).

    Input zero uses the continuous limiting response zero. Positive inputs
    outside the tested log domain fail rather than silently extrapolate.
    Decreasing integration coordinates correctly subtract area for ratio<1.
    """
    if not np.isfinite(ratio) or ratio < 0:
        raise ValueError("ratio must be finite and nonnegative")
    if not np.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    if not np.isfinite(max_step) or not 0 < max_step <= MAX_LOG_STEP:
        raise ValueError(f"max_step must be in (0, {MAX_LOG_STEP}]")
    if ratio == 0:
        return 0.0
    z = float(np.log(ratio))
    if not LOG_DOMAIN[0] <= z <= LOG_DOMAIN[1]:
        raise ValueError("ratio is outside the validated log domain [-8, 8]")
    count = max(1, int(np.ceil(abs(z) / max_step)))
    u = np.linspace(0.0, z, count + 1)
    slopes = alpha * np.exp(alpha * u)
    areas = 0.5 * (slopes[:-1] + slopes[1:]) * np.diff(u)
    result = 1.0 + float(areas.sum())
    if result < 0 or not np.isfinite(result):
        raise ValueError("trapezoid response lost positivity; refine the step")
    return result


def trapezoidal_risk(national: float, h: float, e: float, v: float,
                     max_step: float = MAX_LOG_STEP) -> float:
    """National risk times three trapezoidal responses, then existing soft cap."""
    if not np.isfinite(national) or not 0 <= national <= 1:
        raise ValueError("national risk must be finite and in [0, 1]")
    factors = [trapezoid_response(q, a, max_step)
               for q, a in zip((h, e, v), (ALPHA_H, ALPHA_E, ALPHA_V))]
    return apply_r_ph_soft_cap(float(national * np.prod(factors)))
