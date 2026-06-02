"""
Hemodynamic indices from velocity data.

**RI** and **PI** use aggregate PS/ED (means over retained beats or your agreed definition).

On digitized / waveform paths, **TAmax** is the temporal mean of the envelope samples
(``tamax_from_envelope_temporal_mean``), matching the **PI** denominator. When only PS and ED
are available (e.g. OCR repair), use ``tamax_from_ps_ed_approximation`` as a weighted surrogate.
"""

from __future__ import annotations

import math


def resistive_index_from_ps_ed(ps: float, ed: float) -> float:
    """Resistive index: (PS - ED) / PS. Returns 0.0 if PS is zero."""
    ps = float(ps)
    ed = float(ed)
    if ps == 0.0:
        return 0.0
    return (ps - ed) / ps


def tamax_from_envelope_temporal_mean(temporal_mean_velocity: float) -> float:
    """
    Time-averaged maximum velocity when the sampled series is the maximum-frequency
    envelope: use the temporal mean of those samples over the analysed segment.

    This matches the denominator used for :func:`pulsatility_index_from_ps_ed_and_mean_velocity`,
    so PI and TAmax stay internally consistent on digitized / waveform paths.
    """
    m = float(temporal_mean_velocity)
    if m == 0.0 or not math.isfinite(m):
        return 0.0
    return m


def tamax_from_ps_ed_approximation(ps: float, ed: float) -> float:
    """
    Legacy weighted-mean surrogate when only PS and ED are known (no envelope trace),
    e.g. some OCR consistency checks: (PS + 2 * ED) / 3.

    Prefer :func:`tamax_from_envelope_temporal_mean` when the full envelope ``y`` is available.
    """
    return (float(ps) + 2.0 * float(ed)) / 3.0


def pulsatility_index_from_ps_ed_and_mean_velocity(
    ps: float, ed: float, temporal_mean_velocity: float
) -> float:
    """
    Pulsatility index using the same convention as the prior inline implementation:

    ``PI = (PS - ED) / mean(v)`` where ``mean(v)`` is the temporal mean of the velocity
    trace over the analysed segment (full ``y`` array for that curve). On waveform paths,
    ``mean(v)`` is the same quantity reported as TAmax via ``tamax_from_envelope_temporal_mean``.
    """
    m = float(temporal_mean_velocity)
    if m == 0.0 or not math.isfinite(m):
        return 0.0
    return (float(ps) - float(ed)) / m
