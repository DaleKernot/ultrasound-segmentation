"""
Traffic-light styling for digitized vs extracted (OCR) metrics in HTML tables.

Velocity metrics (PSV, EDV, TAmax): absolute error (cm/s) and percentage error;
color is the worst of the two rule sets.

Ratio metrics (RI, PI): absolute error only; percentage error is not used.
"""

from __future__ import annotations

import math

import pandas as pd

# CSS background values (amber = orange for browser consistency with prior HTML)
_STYLE_GREEN = "background-color: green"
_STYLE_AMBER = "background-color: orange"
_STYLE_RED = "background-color: red"
_STYLE_WHITE = "background-color: white"


def metric_kind_for_digitized_row(word: object) -> str:
    """
    Return ``\"velocity\"`` or ``\"ratio\"`` for traffic-light rules.

    OCR ``Word`` values look like ``\"Lt Ut-PS\"`` / ``\"Lt Ut-TAmax\"`` — use substring
    checks (same idea as ``str.contains(\"PS\")`` in ``plot_correction``), not equality
    on stripped tokens.
    """
    s = str(word if word is not None else "").strip().upper()
    if "PS/ED" in s or "ED/PS" in s or "S/D" in s:
        return "ratio"
    if "TA" in s:
        return "velocity"
    if "PS" in s:
        return "velocity"
    if "ED" in s:
        return "velocity"
    if "PI" in s:
        return "ratio"
    if "RI" in s:
        return "ratio"
    return "velocity"


def velocity_digitized_style(truth: float, predicted: float) -> str:
    """
    PSV / EDV / TAmax style.

    Extracted or digitized velocities may be signed negative; comparisons use
    magnitudes ``abs(truth)`` and ``abs(predicted)``.

    AE = | |predicted| - |truth| | (cm/s). PE = (AE / |truth|) * 100 when |truth| > 0.

    Green: AE <= 5 and PE <= 10%.
    Red: AE > 15 or PE > 20%.
    Amber: otherwise.

    When |truth| is ~0: both ~0 → green; any mismatch → red.
    """
    t = abs(float(truth))
    p = abs(float(predicted))
    ae = abs(p - t)
    if t < 1e-12:
        return _STYLE_GREEN if ae < 1e-9 else _STYLE_RED

    pe = (ae / t) * 100.0
    if ae <= 5.0 and pe <= 5.0:
        return _STYLE_GREEN
    if ae > 10.0 or pe > 10.0:
        return _STYLE_RED
    return _STYLE_AMBER


def ratio_digitized_style(truth: float, predicted: float) -> str:
    """
    RI / PI (and similar unitless indices) style. AE = |predicted - truth| only.

    Green: AE <= 0.05. Amber: 0.05 < AE <= 0.10. Red: AE > 0.10.
    """
    ae = abs(float(predicted) - float(truth))
    if ae <= 0.05:
        return _STYLE_GREEN
    if ae <= 0.10:
        return _STYLE_AMBER
    return _STYLE_RED


def digitized_cell_background_style(word: object, truth: float, predicted: float) -> str:
    """
    Background CSS for one digitized comparison cell.

    ``truth`` is the extracted (OCR) ``Value``; ``predicted`` is the digitized metric.
    """
    if metric_kind_for_digitized_row(word) == "ratio":
        return ratio_digitized_style(truth, predicted)
    return velocity_digitized_style(truth, predicted)


def traffic_light_counts_by_metric_kind(
    df: pd.DataFrame,
    digitized_col: str,
    *,
    value_col: str = "Value",
    word_col: str = "Word",
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """
    Count green / amber / red separately for **velocity** (PS, ED, TA) vs **ratio**
    (S/D, RI, PI) rows, using the same rules as ``digitized_cell_background_style``.

    Returns ``(velocity_g, velocity_a, velocity_r), (ratio_g, ratio_a, ratio_r)``.
    """
    v_green = v_amber = v_red = 0
    r_green = r_amber = r_red = 0
    if df.empty or digitized_col not in df.columns:
        return (0, 0, 0), (0, 0, 0)

    for _, row in df.iterrows():
        pred_raw = row.get(digitized_col)
        if pred_raw is None or (isinstance(pred_raw, str) and pred_raw.strip() == ""):
            continue
        try:
            predicted = float(pred_raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(predicted):
            continue
        truth_raw = row.get(value_col)
        try:
            truth = float(truth_raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(truth):
            continue
        word = row.get(word_col)
        style = digitized_cell_background_style(word, truth, predicted)
        kind = metric_kind_for_digitized_row(word)
        if style == _STYLE_GREEN:
            if kind == "ratio":
                r_green += 1
            else:
                v_green += 1
        elif style == _STYLE_AMBER:
            if kind == "ratio":
                r_amber += 1
            else:
                v_amber += 1
        elif style == _STYLE_RED:
            if kind == "ratio":
                r_red += 1
            else:
                v_red += 1

    return (v_green, v_amber, v_red), (r_green, r_amber, r_red)


def traffic_light_counts_for_digitized_column(
    df: pd.DataFrame,
    digitized_col: str,
    *,
    value_col: str = "Value",
    word_col: str = "Word",
) -> tuple[int, int, int]:
    """
    Count green / amber / red cells for one digitized column vs OCR ``Value``,
    using the same rules as ``digitized_cell_background_style``.

    Only rows with a finite numeric ``Value`` and a parseable non-empty digitized
    cell are counted (e.g. PS, ED, S/D, RI, TA, PI rows after ``plot_correction``).
    """
    (vg, va, vr), (rg, ra, rr) = traffic_light_counts_by_metric_kind(
        df, digitized_col, value_col=value_col, word_col=word_col
    )
    return vg + rg, va + ra, vr + rr


def _column_has_nonempty_digitized(df: pd.DataFrame, digitized_col: str) -> bool:
    if digitized_col not in df.columns:
        return False
    for v in df[digitized_col]:
        if v is None or (isinstance(v, str) and v.strip() == ""):
            continue
        if pd.isna(v):
            continue
        try:
            float(v)
            return True
        except (TypeError, ValueError):
            continue
    return False


def _model_selection_sort_key(
    velocity_counts: tuple[int, int, int],
    ratio_counts: tuple[int, int, int],
) -> tuple[int, int, int, int, int, int]:
    """
    Higher is better when compared with ``reverse=True`` lexicographic sort.

    Order: velocity greens; ratio greens; fewer velocity ambers; fewer ratio ambers;
    fewer velocity reds; fewer ratio reds.
    """
    vg, va, vr = velocity_counts
    rg, ra, rr = ratio_counts
    return (vg, rg, -va, -ra, -vr, -rr)


def select_best_digitized_model_for_image(df: pd.DataFrame) -> str:
    """
    Pick **ray**, **morph**, or **grow** by best match of digitized metrics to OCR ``Value``.

    **Priority 1:** greens on **velocity** metrics (PS, ED, TA) — best clinical match
    for those first.

    **Priority 2 (ties):** **ratio** metrics (S/D, RI, PI) — more greens, then fewer
    ambers, then fewer reds (via the same combined sort key as velocity within the
    ratio bucket).

    **Further ties:** fewer velocity ambers/reds, then ratio ambers/reds (encoded in
    ``_model_selection_sort_key``).

    If no cell was scorable, prefer the first column that has any numeric digitized
    value, else **ray**.
    """
    candidates: list[tuple[str, str]] = [
        ("ray", "Digitized Value (ray)"),
        ("morph", "Digitized Value (morph)"),
        ("grow", "Digitized Value (grow)"),
    ]
    scored: list[
        tuple[
            str,
            tuple[int, int, int, int, int, int],
            tuple[int, int, int, int, int, int],
        ]
    ] = []
    for key, col in candidates:
        if col not in df.columns:
            continue
        v_counts, r_counts = traffic_light_counts_by_metric_kind(df, col)
        vg, va, vr = v_counts
        rg, ra, rr = r_counts
        sort_key = _model_selection_sort_key(v_counts, r_counts)
        scored.append((key, (vg, va, vr, rg, ra, rr), sort_key))

    if not scored:
        return "ray"

    nonempty = [t for t in scored if sum(t[1]) > 0]
    if nonempty:
        nonempty.sort(key=lambda t: t[2], reverse=True)
        return nonempty[0][0]

    for key, col in candidates:
        if col in df.columns and _column_has_nonempty_digitized(df, col):
            return key
    return "ray"


def returned_model_cell_value(model_key: str) -> str:
    """Display string for the ``Returned model`` column (e.g. ``\"morph algorithm\"``)."""
    return f"{model_key} algorithm"
