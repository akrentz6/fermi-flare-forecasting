from __future__ import annotations

import argparse
import itertools
import inspect
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, FuncFormatter

# Shared visual identity only; statistical calculations are unchanged.
METHOD_COLOURS = dict(zip(
    ["Bayesian Blocks", "PELT-linear", "PELT-RBF", "PELT-normal", "Window", "HMM"],
    ["#0072B2", "#D55E00", "#009E73", "#8C5DA5", "#B77A00", "#52657A"],
))
SHORT_NAMES = {"Bayesian Blocks": "BB", "PELT-linear": "Linear",
               "PELT-RBF": "RBF", "PELT-normal": "Normal",
               "Window": "Window", "HMM": "HMM"}
PLOT_DPI = 200


def _plot_style():
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "semibold",
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#BCC5CF", "axes.labelcolor": "#243447",
        "xtick.color": "#344454", "ytick.color": "#344454",
        "text.color": "#243447", "grid.color": "#D8DEE5",
        "grid.linewidth": 0.65, "figure.facecolor": "white", "axes.facecolor": "white",
        "legend.frameon": False, "savefig.facecolor": "white",
    })


def _save_plot(fig, path):
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)


def _number(x):
    if not np.isfinite(x):
        return "NA"
    return f"{x:,.0f}" if abs(x) >= 100 or float(x).is_integer() else f"{x:.3g}"


def _pair_order():
    return list(itertools.combinations(METHOD_ORDER, 2))


def _pair_label(a, b):
    return f"{SHORT_NAMES.get(a, a)} / {SHORT_NAMES.get(b, b)}"


def _annotate_matrix(ax, array, image, *, precision=2):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            v = array[i, j]
            if np.isfinite(v):
                rgba = image.cmap(image.norm(v))
                light = 0.2126*rgba[0] + 0.7152*rgba[1] + 0.0722*rgba[2]
                ax.text(j, i, f"{v:.{precision}f}", ha="center", va="center",
                        color="white" if light < 0.52 else "#243447", fontsize=9)
            else:
                ax.text(j, i, "–", ha="center", va="center", color="#87929F", fontsize=9)
    ax.set_xticks(np.arange(array.shape[1]+1)-0.5, minor=True)
    ax.set_yticks(np.arange(array.shape[0]+1)-0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=0)



METHOD_FILES = {
    "Bayesian Blocks": "segments_bayesian-blocks.csv",
    "PELT-linear": "segments_pelt-linear.csv",
    "PELT-RBF": "segments_pelt-rbf.csv",
    "PELT-normal": "segments_pelt-normal.csv",
    "Window": "segments_window.csv",
    "HMM": "segments_hmm.csv",
}

METHOD_ORDER = list(METHOD_FILES)

ALIASES = {
    "source": [
        "source", "source_name", "source_id", "src_name", "name",
        "sourcekey", "4fgl_name",
    ],
    "time": [
        "time_MJD", "time_mjd", "mjd", "time", "t_mjd", "t",
    ],
    "flux": [
        "flux", "flux_value", "photon_flux", "flux_ph_cm2_s",
        "flux_100_300000", "flux_0p1_300",
    ],
    "error": [
        "flux_error", "flux_err", "flux_uncertainty", "uncertainty",
        "error", "flux_sigma", "flux_unc",
    ],
    "start_time": [
        "segment_start_mjd", "start_mjd", "start_time_mjd",
        "segment_start_time", "start_time", "t_start_mjd", "t_start",
    ],
    "end_time": [
        "segment_end_mjd", "end_mjd", "end_time_mjd",
        "segment_end_time", "end_time", "t_end_mjd", "t_end",
    ],
    "start_index": [
        "start_idx", "start_index", "first_index", "i_start",
        "start_bin", "start_observation",
    ],
    "end_index": [
        "end_idx", "end_index", "last_index", "i_end",
        "end_bin", "end_observation",
    ],
    "generic_start": ["segment_start", "start"],
    "generic_end": ["segment_end", "end"],
    "fitted_flux": [
        "fitted_flux", "fit_flux", "fitted_value", "mean_flux",
        "segment_flux", "level", "predicted_flux",
    ],
    "slope": [
        "slope_per_day", "slope_day", "slope", "trend_slope",
        "flux_slope_per_day",
    ],
    "state": ["state", "hmm_state", "hidden_state", "latent_state"],
    "reference_time": [
        "fit_reference_mjd", "reference_time_mjd", "reference_time",
        "center_time_mjd", "centre_time_mjd", "center_time",
        "centre_time", "time_center", "time_centre", "t_ref",
        "segment_center_mjd", "segment_centre_mjd",
    ],
}


def _normalise_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _find_column(
    frame: pd.DataFrame,
    aliases: list[str],
    *,
    required: bool = False,
    description: str = "",
) -> Optional[str]:
    by_normalised = {_normalise_name(c): c for c in frame.columns}
    for alias in aliases:
        match = by_normalised.get(_normalise_name(alias))
        if match is not None:
            return match
    if required:
        raise ValueError(
            f"Could not identify {description or aliases[0]} column. "
            f"Available columns: {list(frame.columns)}"
        )
    return None


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _read_observations(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cleaned observation file not found: {path}")

    raw = pd.read_csv(path)
    source_col = _find_column(raw, ALIASES["source"], required=True, description="source")
    time_col = _find_column(raw, ALIASES["time"], required=True, description="MJD/time")
    flux_col = _find_column(raw, ALIASES["flux"], required=True, description="flux")
    error_col = _find_column(raw, ALIASES["error"])

    out = pd.DataFrame({
        "source": raw[source_col].astype(str).str.strip(),
        "time": _numeric(raw[time_col]),
        "flux": _numeric(raw[flux_col]),
    })
    out["error"] = _numeric(raw[error_col]) if error_col else np.nan
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["source", "time", "flux"])
    out = out[out["source"] != ""]
    out = out.sort_values(["source", "time"], kind="mergesort").reset_index(drop=True)

    if out.empty:
        raise ValueError(f"No valid source/time/flux rows found in {path}")

    print(
        f"Loaded {len(out):,} cleaned detections from {path} "
        f"({out['source'].nunique():,} sources)."
    )
    if error_col is None:
        print("WARNING: no flux-error column found; standardised residual diagnostics will be skipped.")
    else:
        n_bad = int((~np.isfinite(out["error"]) | (out["error"] <= 0)).sum())
        print(f"Rows with missing/non-positive flux errors (excluded from standardised residuals): {n_bad:,}")

    return out


def _infer_time_or_index(
    values: pd.Series,
    observations: pd.DataFrame,
    explicit_time: bool,
    explicit_index: bool,
) -> str:
    if explicit_time:
        return "time"
    if explicit_index:
        return "index"

    vals = _numeric(values).dropna()
    if vals.empty:
        raise ValueError("Segment start/end column contains no numeric values.")

    # MJD and MET values are far larger than within-source observation indices.
    if float(vals.abs().median()) > 10_000:
        return "time"

    n_max = int(observations.groupby("source").size().max())
    integer_like = np.all(np.isclose(vals.to_numpy(), np.round(vals.to_numpy())))
    if integer_like and vals.min() >= 0 and vals.max() <= n_max:
        return "index"

    # If generic values overlap the observed time range, interpret them as times.
    t_min, t_max = observations["time"].min(), observations["time"].max()
    if vals.min() >= t_min - 5 and vals.max() <= t_max + 5:
        return "time"

    raise ValueError(
        "Could not infer whether generic segment start/end columns are times or "
        "indices. Rename them to start_MJD/end_MJD or start_idx/end_idx."
    )


def _read_segments(
    path: Path,
    method: str,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Segmentation output not found: {path}")

    raw = pd.read_csv(path)
    source_col = _find_column(raw, ALIASES["source"], required=True, description="segment source")
    start_time_col = _find_column(raw, ALIASES["start_time"])
    end_time_col = _find_column(raw, ALIASES["end_time"])
    start_idx_col = _find_column(raw, ALIASES["start_index"])
    end_idx_col = _find_column(raw, ALIASES["end_index"])
    generic_start_col = _find_column(raw, ALIASES["generic_start"])
    generic_end_col = _find_column(raw, ALIASES["generic_end"])

    start_col = start_time_col or start_idx_col or generic_start_col
    end_col = end_time_col or end_idx_col or generic_end_col
    if start_col is None or end_col is None:
        raise ValueError(
            f"{path.name}: could not identify segment start/end columns. "
            f"Available columns: {list(raw.columns)}"
        )

    fitted_col = _find_column(raw, ALIASES["fitted_flux"], required=True, description="fitted_flux")
    slope_col = _find_column(raw, ALIASES["slope"])
    ref_col = _find_column(raw, ALIASES["reference_time"])
    state_col = _find_column(raw, ALIASES["state"])

    # Explicitly named time/index columns take precedence. Generic start/end
    # columns are inferred from their scale and the source observation counts.
    kind = _infer_time_or_index(
        raw[start_col],
        observations,
        explicit_time=(start_col == start_time_col),
        explicit_index=(start_col == start_idx_col),
    )
    end_kind = _infer_time_or_index(
        raw[end_col],
        observations,
        explicit_time=(end_col == end_time_col),
        explicit_index=(end_col == end_idx_col),
    )
    if kind != end_kind:
        raise ValueError(f"{path.name}: start and end columns appear to use different units.")

    out = pd.DataFrame({
        "source": raw[source_col].astype(str).str.strip(),
        "start_value": _numeric(raw[start_col]),
        "end_value": _numeric(raw[end_col]),
        "fitted_flux": _numeric(raw[fitted_col]),
    })
    out["slope_per_day"] = _numeric(raw[slope_col]) if slope_col else 0.0
    out["explicit_reference_time"] = _numeric(raw[ref_col]) if ref_col else np.nan
    out["state"] = _numeric(raw[state_col]) if state_col else np.nan
    out["source_row"] = np.arange(len(out))
    out = out.dropna(subset=["source", "start_value", "end_value", "fitted_flux"])
    out = out[out["source"] != ""]

    # Convert index-based boundaries to the corresponding observation times.
    if kind == "index":
        converted = []
        obs_groups = {
            source: group.sort_values("time").reset_index(drop=True)
            for source, group in observations.groupby("source", sort=False)
        }
        for row in out.itertuples(index=False):
            group = obs_groups.get(row.source)
            if group is None or group.empty:
                converted.append((np.nan, np.nan))
                continue
            i0, i1 = int(round(row.start_value)), int(round(row.end_value))
            if i0 < 0 or i1 < 0 or i0 >= len(group) or i1 >= len(group):
                converted.append((np.nan, np.nan))
                continue
            converted.append((float(group.loc[i0, "time"]), float(group.loc[i1, "time"])))
        out[["start", "end"]] = pd.DataFrame(converted, index=out.index)
    else:
        out["start"] = out["start_value"]
        out["end"] = out["end_value"]

    out = out.dropna(subset=["start", "end"])
    out = out[out["end"] >= out["start"]]
    out["method"] = method
    out = out.sort_values(["source", "start", "end"]).reset_index(drop=True)

    if out.empty:
        raise ValueError(f"No valid segments found in {path}")

    print(
        f"Loaded {len(out):,} segments for {method} "
        f"({out['source'].nunique():,} sources; boundaries interpreted as {kind})."
    )
    return out


def _assign_observations(
    obs: pd.DataFrame,
    segments: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return segment-row positions and fitted values for sorted source observations."""
    seg = segments.sort_values(["start", "end"]).reset_index(drop=True)
    starts = seg["start"].to_numpy(dtype=float)
    ends = seg["end"].to_numpy(dtype=float)
    times = obs["time"].to_numpy(dtype=float)

    # Select the latest segment whose start is <= the observation time.
    idx = np.searchsorted(starts, times, side="right") - 1
    valid = idx >= 0
    safe_idx = np.clip(idx, 0, max(len(seg) - 1, 0))
    valid &= times <= ends[safe_idx] + 1e-8

    # If boundaries have tiny numerical gaps, leave those observations unassigned
    # rather than silently extrapolating a fitted segment across the gap.
    assigned = np.where(valid, idx, -1)
    return assigned, safe_idx


def _segment_reference_time(
    segment_row: pd.Series,
    assigned_obs: pd.DataFrame,
) -> float:
    """Reference time at which the exported PELT-linear fitted_flux is exact.

    segment_light_curves.py exports fitted_flux = flux[i:j].mean() and obtains
    slope_per_day from an OLS regression with an intercept. The fitted line
    therefore passes through (mean(time[i:j]), mean(flux[i:j])).
    """
    explicit = segment_row["explicit_reference_time"]
    if np.isfinite(explicit):
        return float(explicit)
    if assigned_obs.empty:
        return float((segment_row["start"] + segment_row["end"]) / 2.0)
    return float(assigned_obs["time"].mean())



def _evaluate_source_method(
    obs: pd.DataFrame,
    segments: pd.DataFrame,
    method: str,
) -> tuple[pd.DataFrame, dict, np.ndarray]:
    obs = obs.sort_values("time").reset_index(drop=True)
    seg = segments.sort_values(["start", "end"]).reset_index(drop=True)
    assigned, _ = _assign_observations(obs, seg)

    prediction = np.full(len(obs), np.nan, dtype=float)
    counts = np.bincount(assigned[assigned >= 0], minlength=len(seg))

    segment_gap_values: list[np.ndarray] = []
    segment_observed_spans: list[float] = []
    segment_max_gaps: list[float] = []
    segment_gap_counts = {threshold: [] for threshold in (9.0, 30.0, 90.0)}

    for j, row in seg.iterrows():
        mask = assigned == j
        assigned_times = obs.loc[mask, "time"].to_numpy(dtype=float)
        gaps = np.diff(assigned_times) if len(assigned_times) > 1 else np.array([], dtype=float)
        segment_gap_values.append(gaps)
        segment_observed_spans.append(
            float(assigned_times[-1] - assigned_times[0]) if len(assigned_times) > 1 else 0.0
        )
        segment_max_gaps.append(float(np.max(gaps)) if len(gaps) else np.nan)
        for threshold in segment_gap_counts:
            segment_gap_counts[threshold].append(int(np.sum(gaps > threshold)))

        if not mask.any():
            continue

        fitted = float(row["fitted_flux"])
        slope = float(row["slope_per_day"]) if np.isfinite(row["slope_per_day"]) else 0.0

        if method == "PELT-linear":
            reference = _segment_reference_time(row, obs.loc[mask])
            prediction[mask] = fitted + slope * (
                obs.loc[mask, "time"].to_numpy(dtype=float) - reference
            )
        else:
            prediction[mask] = fitted

    valid = np.isfinite(prediction)
    y = obs["flux"].to_numpy(dtype=float)
    residual = y[valid] - prediction[valid]
    raw_rmse = float(np.sqrt(np.mean(residual ** 2))) if residual.size else np.nan
    raw_mae = float(np.mean(np.abs(residual))) if residual.size else np.nan

    errors = obs["error"].to_numpy(dtype=float)
    std_mask = valid & np.isfinite(errors) & (errors > 0)
    std_resid = (y[std_mask] - prediction[std_mask]) / errors[std_mask]
    std_rmse = float(np.sqrt(np.mean(std_resid ** 2))) if std_resid.size else np.nan
    std_bias = float(np.mean(std_resid)) if std_resid.size else np.nan
    std_sd = float(np.std(std_resid, ddof=1)) if std_resid.size > 1 else np.nan
    frac_3sigma = float(np.mean(np.abs(std_resid) > 3)) if std_resid.size else np.nan

    durations = np.maximum(seg["end"].to_numpy(dtype=float) - seg["start"].to_numpy(dtype=float), 0.0)
    obs_counts = counts.astype(float)
    within_segment_gaps = (
        np.concatenate([g for g in segment_gap_values if len(g)])
        if any(len(g) for g in segment_gap_values)
        else np.array([], dtype=float)
    )
    all_observation_gaps = np.diff(obs["time"].to_numpy(dtype=float))
    obs_span_days = float(obs["time"].max() - obs["time"].min()) if len(obs) > 1 else 0.0
    observed_span_years = obs_span_days / 365.25
    n_segments = len(seg)
    n_change_points = max(n_segments - 1, 0)

    obs_flux_span = float(np.nanquantile(y, 0.95) - np.nanquantile(y, 0.05))
    fit_values = prediction[valid]
    fit_span = (
        float(np.nanquantile(fit_values, 0.95) - np.nanquantile(fit_values, 0.05))
        if fit_values.size else np.nan
    )
    amplitude_retention = fit_span / obs_flux_span if obs_flux_span > 0 else np.nan

    # Magnitude of the fitted jump at a segment boundary.
    jumps = []
    for j in range(1, len(seg)):
        boundary = float(seg.loc[j, "start"])
        left = seg.loc[j - 1]
        right = seg.loc[j]
        left_slope = float(left["slope_per_day"]) if np.isfinite(left["slope_per_day"]) else 0.0
        right_slope = float(right["slope_per_day"]) if np.isfinite(right["slope_per_day"]) else 0.0
        if method == "PELT-linear":
            left_obs = obs.loc[assigned == (j - 1)]
            right_obs = obs.loc[assigned == j]
            left_ref = _segment_reference_time(left, left_obs)
            right_ref = _segment_reference_time(right, right_obs)
            left_value = float(left["fitted_flux"]) + left_slope * (boundary - left_ref)
            right_value = float(right["fitted_flux"]) + right_slope * (boundary - right_ref)
        else:
            left_value = float(left["fitted_flux"])
            right_value = float(right["fitted_flux"])
        jumps.append(abs(right_value - left_value))

    gap_counts_by_threshold = {
        threshold: int(np.sum(within_segment_gaps > threshold))
        for threshold in (9.0, 30.0, 90.0)
    }
    segment_counts_by_threshold = {
        threshold: int(np.sum(np.asarray(segment_gap_counts[threshold]) > 0))
        for threshold in (9.0, 30.0, 90.0)
    }
    n_within_gaps = int(len(within_segment_gaps))

    metrics = {
        "source": str(obs["source"].iloc[0]),
        "method": method,
        "n_observations": int(len(obs)),
        "n_assigned_observations": int(valid.sum()),
        "fit_coverage": float(valid.mean()) if len(valid) else np.nan,
        "n_segments": int(n_segments),
        "n_change_points": int(n_change_points),
        "change_points_per_year": (
            n_change_points / observed_span_years if observed_span_years > 0 else np.nan
        ),
        "median_segment_duration_days": float(np.median(durations)) if len(durations) else np.nan,
        "p90_segment_duration_days": float(np.quantile(durations, 0.90)) if len(durations) else np.nan,
        "median_observed_span_per_segment_days": (
            float(np.median(segment_observed_spans)) if segment_observed_spans else np.nan
        ),
        "median_observations_per_segment": float(np.median(obs_counts)) if len(obs_counts) else np.nan,
        "p90_observations_per_segment": float(np.quantile(obs_counts, 0.90)) if len(obs_counts) else np.nan,
        "median_within_segment_gap_days": (
            float(np.median(within_segment_gaps)) if n_within_gaps else np.nan
        ),
        "max_within_segment_gap_days": (
            float(np.max(within_segment_gaps)) if n_within_gaps else np.nan
        ),
        "max_observation_gap_days": (
            float(np.max(all_observation_gaps)) if len(all_observation_gaps) else np.nan
        ),
        "n_within_segment_observation_gaps": n_within_gaps,
        "n_observation_gaps_gt_9_days": int(np.sum(all_observation_gaps > 9.0)),
        "n_observation_gaps_gt_30_days": int(np.sum(all_observation_gaps > 30.0)),
        "n_observation_gaps_gt_90_days": int(np.sum(all_observation_gaps > 90.0)),
        "raw_rmse": raw_rmse,
        "raw_mae": raw_mae,
        "standardised_residual_rmse": std_rmse,
        "standardised_residual_bias": std_bias,
        "standardised_residual_sd": std_sd,
        "fraction_residuals_over_3sigma": frac_3sigma,
        "n_standardised_residuals": int(std_mask.sum()),
        "observed_robust_amplitude": obs_flux_span,
        "fitted_robust_amplitude": fit_span,
        "amplitude_retention": amplitude_retention,
        "median_boundary_jump": float(np.median(jumps)) if jumps else np.nan,
        "median_absolute_slope_per_day": (
            float(np.median(np.abs(seg["slope_per_day"].to_numpy(dtype=float))))
            if method == "PELT-linear" else np.nan
        ),
    }

    for threshold in (9.0, 30.0, 90.0):
        tag = int(threshold)
        metrics[f"n_within_segment_gaps_gt_{tag}_days"] = gap_counts_by_threshold[threshold]
        metrics[f"fraction_within_segment_gaps_gt_{tag}_days"] = (
            gap_counts_by_threshold[threshold] / n_within_gaps if n_within_gaps else np.nan
        )
        metrics[f"n_segments_with_gap_gt_{tag}_days"] = segment_counts_by_threshold[threshold]
        metrics[f"fraction_segments_with_gap_gt_{tag}_days"] = (
            segment_counts_by_threshold[threshold] / n_segments if n_segments else np.nan
        )

    segment_metrics = seg.copy()
    segment_metrics["n_assigned_observations"] = counts
    segment_metrics["duration_days"] = durations
    segment_metrics["observed_span_days"] = segment_observed_spans
    segment_metrics["max_within_segment_gap_days"] = segment_max_gaps
    for threshold in (9.0, 30.0, 90.0):
        tag = int(threshold)
        segment_metrics[f"n_within_segment_gaps_gt_{tag}_days"] = segment_gap_counts[threshold]
        segment_metrics[f"has_within_segment_gap_gt_{tag}_days"] = (
            segment_metrics[f"n_within_segment_gaps_gt_{tag}_days"] > 0
        )

    return segment_metrics, metrics, prediction



def _boundary_gap_info(
    change_points: np.ndarray,
    observation_times: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    """Map each change point to the observation interval containing it, if any."""
    times = np.unique(np.sort(np.asarray(observation_times, dtype=float)))
    cps = np.asarray(change_points, dtype=float)
    gap_ids = np.full(len(cps), None, dtype=object)
    details: list[dict] = []

    for index, cp in enumerate(cps):
        right = int(np.searchsorted(times, cp, side="left"))
        previous_time = np.nan
        next_time = np.nan
        gap_days = np.nan
        location = "outside_observed_range"

        if right < len(times) and np.isclose(times[right], cp, rtol=0.0, atol=1e-8):
            previous_time = float(times[right])
            next_time = float(times[right])
            gap_days = 0.0
            location = "at_observation"
        elif 0 < right < len(times):
            previous_time = float(times[right - 1])
            next_time = float(times[right])
            gap_days = next_time - previous_time
            if previous_time < cp < next_time:
                gap_ids[index] = right - 1
                location = "between_observations"

        details.append({
            "boundary_index": index + 1,
            "change_point_mjd": float(cp),
            "previous_observation_mjd": previous_time,
            "next_observation_mjd": next_time,
            "observational_gap_days": gap_days,
            "boundary_location": location,
            "is_in_observation_gap": location == "between_observations",
            "in_gap_gt_9_days": bool(np.isfinite(gap_days) and gap_days > 9.0),
            "in_gap_gt_30_days": bool(np.isfinite(gap_days) and gap_days > 30.0),
            "in_gap_gt_90_days": bool(np.isfinite(gap_days) and gap_days > 90.0),
        })

    return gap_ids, details


def _optimal_boundary_match(
    a: np.ndarray,
    b: np.ndarray,
    tolerance: float,
    gap_ids_a: Optional[np.ndarray] = None,
    gap_ids_b: Optional[np.ndarray] = None,
) -> dict:
    """Maximum-cardinality, minimum-total-error monotone one-to-one matching.

    When gap IDs are supplied, boundaries in the same observation-to-observation
    interval may also match. This is the gap-aware variant: the exact time within
    that interval is not observationally resolved.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    order_a = np.argsort(a)
    order_b = np.argsort(b)
    a = a[order_a]
    b = b[order_b]
    n, m = len(a), len(b)

    if gap_ids_a is not None:
        gap_ids_a = np.asarray(gap_ids_a, dtype=object)[order_a]
    if gap_ids_b is not None:
        gap_ids_b = np.asarray(gap_ids_b, dtype=object)[order_b]

    if n == 0 and m == 0:
        return {
            "fraction_a_matched": 1.0,
            "fraction_b_matched": 1.0,
            "f1": 1.0,
            "n_matches": 0,
            "n_within_tolerance_matches": 0,
            "n_same_gap_matches": 0,
            "n_same_gap_only_matches": 0,
            "n_unmatched_a": 0,
            "n_unmatched_b": 0,
            "median_match_error_days": np.nan,
            "mean_match_error_days": np.nan,
            "max_match_error_days": np.nan,
            "matched_pairs": [],
        }
    if n == 0 or m == 0:
        return {
            "fraction_a_matched": 0.0 if n else 1.0,
            "fraction_b_matched": 0.0 if m else 1.0,
            "f1": 0.0,
            "n_matches": 0,
            "n_within_tolerance_matches": 0,
            "n_same_gap_matches": 0,
            "n_same_gap_only_matches": 0,
            "n_unmatched_a": n,
            "n_unmatched_b": m,
            "median_match_error_days": np.nan,
            "mean_match_error_days": np.nan,
            "max_match_error_days": np.nan,
            "matched_pairs": [],
        }

    dp_count = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp_cost = np.zeros((n + 1, m + 1), dtype=float)
    choice = np.zeros((n + 1, m + 1), dtype=np.uint8)

    def same_gap(i: int, j: int) -> bool:
        if gap_ids_a is None or gap_ids_b is None:
            return False
        ga, gb = gap_ids_a[i], gap_ids_b[j]
        return ga is not None and gb is not None and ga == gb

    def is_better(count, cost, best_count, best_cost):
        return (
            count > best_count
            or (count == best_count and cost < best_cost - 1e-12)
        )

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best_count = int(dp_count[i - 1, j])
            best_cost = float(dp_cost[i - 1, j])
            best_choice = 1

            candidate_count = int(dp_count[i, j - 1])
            candidate_cost = float(dp_cost[i, j - 1])
            if is_better(candidate_count, candidate_cost, best_count, best_cost):
                best_count, best_cost, best_choice = candidate_count, candidate_cost, 2

            distance = abs(a[i - 1] - b[j - 1])
            within_tolerance = distance <= tolerance
            same_observation_gap = same_gap(i - 1, j - 1)
            if within_tolerance or same_observation_gap:
                candidate_count = int(dp_count[i - 1, j - 1]) + 1
                candidate_cost = float(dp_cost[i - 1, j - 1]) + distance
                if is_better(candidate_count, candidate_cost, best_count, best_cost):
                    best_count, best_cost, best_choice = candidate_count, candidate_cost, 3
                elif (
                    candidate_count == best_count
                    and abs(candidate_cost - best_cost) <= 1e-12
                ):
                    best_choice = 3

            dp_count[i, j] = best_count
            dp_cost[i, j] = best_cost
            choice[i, j] = best_choice

    matched_pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        selected = choice[i, j]
        if selected == 3:
            distance = abs(a[i - 1] - b[j - 1])
            same_observation_gap = same_gap(i - 1, j - 1)
            matched_pairs.append({
                "a_index": i - 1,
                "b_index": j - 1,
                "error_days": float(distance),
                "same_observation_gap": bool(same_observation_gap),
                "within_tolerance": bool(distance <= tolerance),
                "same_gap_only": bool(same_observation_gap and distance > tolerance),
            })
            i -= 1
            j -= 1
        elif selected == 1:
            i -= 1
        elif selected == 2:
            j -= 1
        else:
            break

    matched_pairs.reverse()
    errors = np.asarray([pair["error_days"] for pair in matched_pairs], dtype=float)
    matches = len(matched_pairs)
    fraction_a = matches / n
    fraction_b = matches / m
    f1 = (
        2 * fraction_a * fraction_b / (fraction_a + fraction_b)
        if fraction_a + fraction_b else 0.0
    )

    return {
        "fraction_a_matched": fraction_a,
        "fraction_b_matched": fraction_b,
        "f1": f1,
        "n_matches": matches,
        "n_within_tolerance_matches": int(sum(p["within_tolerance"] for p in matched_pairs)),
        "n_same_gap_matches": int(sum(p["same_observation_gap"] for p in matched_pairs)),
        "n_same_gap_only_matches": int(sum(p["same_gap_only"] for p in matched_pairs)),
        "n_unmatched_a": n - matches,
        "n_unmatched_b": m - matches,
        "median_match_error_days": float(np.median(errors)) if matches else np.nan,
        "mean_match_error_days": float(np.mean(errors)) if matches else np.nan,
        "max_match_error_days": float(np.max(errors)) if matches else np.nan,
        "matched_pairs": matched_pairs,
    }


def _collect_change_point_diagnostics(
    source_segments: dict,
    source_observations: dict,
) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        for source, seg in source_segments.get(method, {}).items():
            obs = source_observations.get(source)
            if obs is None or len(seg) < 2:
                continue
            ordered = seg.sort_values(["start", "end"]).reset_index(drop=True)
            cps = ordered["start"].to_numpy(dtype=float)[1:]
            _, details = _boundary_gap_info(cps, obs["time"].to_numpy(dtype=float))
            for detail in details:
                rows.append({"source": source, "method": method, **detail})
    return pd.DataFrame(rows)


def _make_pairwise_comparisons(
    method_predictions: dict,
    source_segments: dict,
    source_observations: dict,
    tolerances: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    curve_rows, boundary_rows, event_rows = [], [], []

    for method_a, method_b in itertools.combinations(METHOD_ORDER, 2):
        common_sources = sorted(
            set(method_predictions.get(method_a, {}))
            & set(method_predictions.get(method_b, {}))
        )
        for source in common_sources:
            obs = source_observations[source]
            pa = method_predictions[method_a][source]
            pb = method_predictions[method_b][source]
            valid = np.isfinite(pa) & np.isfinite(pb)
            if valid.any():
                difference = pa[valid] - pb[valid]
                rmse = float(np.sqrt(np.mean(difference ** 2)))
                scale = float(
                    np.nanquantile(obs["flux"], 0.95) - np.nanquantile(obs["flux"], 0.05)
                )
                nrmse = rmse / scale if scale > 0 else np.nan
            else:
                rmse, nrmse = np.nan, np.nan

            curve_rows.append({
                "source": source,
                "method_a": method_a,
                "method_b": method_b,
                "curve_rmse": rmse,
                "normalised_curve_rmse": nrmse,
                "n_common_observations": int(valid.sum()),
            })

            seg_a = source_segments[method_a].get(source)
            seg_b = source_segments[method_b].get(source)
            if seg_a is None or seg_b is None:
                continue

            seg_a = seg_a.sort_values("start").reset_index(drop=True)
            seg_b = seg_b.sort_values("start").reset_index(drop=True)
            cp_a = seg_a["start"].to_numpy(dtype=float)[1:]
            cp_b = seg_b["start"].to_numpy(dtype=float)[1:]
            times = obs["time"].to_numpy(dtype=float)
            gap_ids_a, _ = _boundary_gap_info(cp_a, times)
            gap_ids_b, _ = _boundary_gap_info(cp_b, times)

            n_a, n_b = len(cp_a), len(cp_b)
            presence_category = (
                "neither" if n_a == 0 and n_b == 0
                else "only_method_a" if n_a > 0 and n_b == 0
                else "only_method_b" if n_a == 0 and n_b > 0
                else "both"
            )

            for tolerance in tolerances:
                standard = _optimal_boundary_match(cp_a, cp_b, tolerance)
                gap_aware = _optimal_boundary_match(
                    cp_a, cp_b, tolerance, gap_ids_a=gap_ids_a, gap_ids_b=gap_ids_b
                )
                standard_matches = standard.pop("matched_pairs")
                gap_matches = gap_aware.pop("matched_pairs")

                boundary_rows.append({
                    "source": source,
                    "method_a": method_a,
                    "method_b": method_b,
                    "tolerance_days": tolerance,
                    "n_change_points_a": n_a,
                    "n_change_points_b": n_b,
                    "change_point_presence_category": presence_category,
                    "both_no_change_points": n_a == 0 and n_b == 0,
                    "only_method_a_has_change_points": n_a > 0 and n_b == 0,
                    "only_method_b_has_change_points": n_a == 0 and n_b > 0,
                    "both_have_change_points": n_a > 0 and n_b > 0,
                    "any_match": len(standard_matches) > 0,
                    "both_have_change_points_no_match": (
                        n_a > 0 and n_b > 0 and len(standard_matches) == 0
                    ),
                    **standard,
                    "gap_aware_any_match": len(gap_matches) > 0,
                    **{f"gap_aware_{key}": value for key, value in gap_aware.items()},
                })

                for matching_mode, matched_pairs in (
                    ("time_tolerance", standard_matches),
                    ("gap_aware", gap_matches),
                ):
                    for match in matched_pairs:
                        ia, ib = match["a_index"], match["b_index"]
                        same_observation_gap = (
                            gap_ids_a[ia] is not None
                            and gap_ids_b[ib] is not None
                            and gap_ids_a[ia] == gap_ids_b[ib]
                        )
                        within_tolerance = match["error_days"] <= tolerance
                        event_rows.append({
                            "source": source,
                            "method_a": method_a,
                            "method_b": method_b,
                            "tolerance_days": tolerance,
                            "matching_mode": matching_mode,
                            "change_point_a_index": ia + 1,
                            "change_point_b_index": ib + 1,
                            "change_point_a_mjd": float(cp_a[ia]),
                            "change_point_b_mjd": float(cp_b[ib]),
                            "absolute_error_days": match["error_days"],
                            "matched_in_same_observation_gap": bool(same_observation_gap),
                            "matched_by_same_gap_only": bool(
                                matching_mode == "gap_aware"
                                and same_observation_gap
                                and not within_tolerance
                            ),
                            "within_time_tolerance": bool(within_tolerance),
                            "observation_gap_id_a": (
                                int(gap_ids_a[ia]) if gap_ids_a[ia] is not None else np.nan
                            ),
                            "observation_gap_id_b": (
                                int(gap_ids_b[ib]) if gap_ids_b[ib] is not None else np.nan
                            ),
                        })

    return pd.DataFrame(curve_rows), pd.DataFrame(boundary_rows), pd.DataFrame(event_rows)



def _aggregate_pairwise(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    result = (
        frame.groupby(["method_a", "method_b"], as_index=False)
        .agg(
            n_sources=("source", "nunique"),
            median=(value, "median"),
            q25=(value, lambda x: x.quantile(0.25)),
            q75=(value, lambda x: x.quantile(0.75)),
        )
    )
    return result.rename(columns={
        "median": value,
        "q25": f"{value}_q25",
        "q75": f"{value}_q75",
    })


def _aggregate_boundary_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    def pct(values: pd.Series) -> float:
        return 100.0 * float(values.astype(float).mean()) if len(values) else np.nan

    rows = []
    grouping = ["tolerance_days", "method_a", "method_b"]
    for keys, group in frame.groupby(grouping, sort=False):
        tolerance, method_a, method_b = keys
        row = {
            "tolerance_days": tolerance,
            "method_a": method_a,
            "method_b": method_b,
            "n_sources": int(group["source"].nunique()),
            "median_f1": float(group["f1"].median()),
            "mean_f1": float(group["f1"].mean()),
            "q25_f1": float(group["f1"].quantile(0.25)),
            "q75_f1": float(group["f1"].quantile(0.75)),
            "median_f1_conditional_both_have_change_points": (
                float(group.loc[group["both_have_change_points"], "f1"].median())
                if group["both_have_change_points"].any() else np.nan
            ),
            "pct_sources_with_any_match": pct(group["any_match"]),
            "pct_sources_no_change_points_either": pct(group["both_no_change_points"]),
            "pct_sources_only_method_a_has_change_points": pct(
                group["only_method_a_has_change_points"]
            ),
            "pct_sources_only_method_b_has_change_points": pct(
                group["only_method_b_has_change_points"]
            ),
            "pct_sources_with_change_points_both": pct(group["both_have_change_points"]),
            "pct_sources_both_have_change_points_no_match": pct(
                group["both_have_change_points_no_match"]
            ),
            "median_n_matches": float(group["n_matches"].median()),
            "median_match_error_days": float(group["median_match_error_days"].median()),
            "mean_match_error_days": float(group["mean_match_error_days"].mean()),
            "gap_aware_median_f1": float(group["gap_aware_f1"].median()),
            "gap_aware_mean_f1": float(group["gap_aware_f1"].mean()),
            "gap_aware_q25_f1": float(group["gap_aware_f1"].quantile(0.25)),
            "gap_aware_q75_f1": float(group["gap_aware_f1"].quantile(0.75)),
            "gap_aware_median_f1_conditional_both_have_change_points": (
                float(group.loc[group["both_have_change_points"], "gap_aware_f1"].median())
                if group["both_have_change_points"].any() else np.nan
            ),
            "pct_sources_gap_aware_with_any_match": pct(group["gap_aware_any_match"]),
            "median_gap_aware_n_matches": float(group["gap_aware_n_matches"].median()),
            "median_gap_aware_match_error_days": float(
                group["gap_aware_median_match_error_days"].median()
            ),
            "pct_gap_aware_matches_same_gap_only": (
                100.0 * group["gap_aware_n_same_gap_only_matches"].sum()
                / group["gap_aware_n_matches"].sum()
                if group["gap_aware_n_matches"].sum() > 0 else np.nan
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_event_matches(event_matches: pd.DataFrame) -> pd.DataFrame:
    if event_matches.empty:
        return pd.DataFrame()
    return (
        event_matches.groupby(
            ["matching_mode", "tolerance_days", "method_a", "method_b"],
            as_index=False,
        )
        .agg(
            n_matched_events=("absolute_error_days", "size"),
            median_error_days=("absolute_error_days", "median"),
            mean_error_days=("absolute_error_days", "mean"),
            p90_error_days=("absolute_error_days", lambda x: x.quantile(0.90)),
            max_error_days=("absolute_error_days", "max"),
            pct_matches_same_gap_only=("matched_by_same_gap_only", lambda x: 100.0 * x.mean()),
        )
    )


def _plot_boxplots(metrics, column, ylabel, title, path, *, log_scale=False):
    if metrics.empty or "method" not in metrics or column not in metrics:
        return
    data, labels = [], []
    for method in METHOD_ORDER:
        vals = metrics.loc[metrics["method"] == method, column].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            data.append(vals)
            labels.append(method)
    if not data:
        return
    # Preserve zeros: a singleton segment has zero first-to-last observed span.
    # Symlog shows zero and small values without sacrificing the long upper tail.
    use_symlog = log_scale and any(np.any(v <= 0) for v in data)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    orientation = ({"orientation": "horizontal"} if "orientation" in inspect.signature(ax.boxplot).parameters else {"vert": False})
    bp = ax.boxplot(data, **orientation, patch_artist=True, showfliers=False,
                    medianprops={"color": "#182B3A", "linewidth": 2},
                    whiskerprops={"color": "#73808D"}, capprops={"color": "#73808D"})
    for box, method in zip(bp["boxes"], labels):
        box.set_facecolor(METHOD_COLOURS[method]); box.set_alpha(0.58)
        box.set_edgecolor(METHOD_COLOURS[method])
    ax.set_yticks(np.arange(1, len(labels)+1), labels)
    ax.invert_yaxis()
    if use_symlog:
        ax.set_xscale("symlog", linthresh=1, linscale=0.7)
        if all(np.min(v) >= 0 for v in data):
            ax.set_xlim(left=0)
    elif log_scale:
        ax.set_xscale("log")
    ax.set_xlabel(ylabel + (" (symmetric log scale)" if use_symlog else
                            " (log scale)" if log_scale else ""))
    ax.set_title(title, loc="left", pad=15)
    ax.grid(axis="x"); ax.set_axisbelow(True)
    # Place summary values in a separate aligned column, not over the data.
    ax.text(1.035, 1.025, "Median / n", transform=ax.transAxes, fontsize=9,
            fontweight="semibold", va="bottom", clip_on=False)
    for i, vals in enumerate(data, 1):
        ax.text(1.035, i, f"{_number(np.median(vals))} / {len(vals):,}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=9, clip_on=False)
    if column in {"duration_days", "observed_span_days", "max_within_segment_gap_days", "n_assigned_observations"}:
        unit = "Pooled segments; sources with more segments contribute more."
    else:
        unit = "One value per source; each source has equal weight."
    fig.text(.02, .015, unit + " Boxes: IQR; whiskers: 1.5×IQR; outliers not drawn.", fontsize=8.5)
    fig.tight_layout(rect=(0, .055, .85, 1))
    _save_plot(fig, path)


def _plot_heatmap(pairwise, value, title, path, *, agreement_scale=False):
    if pairwise.empty or value not in pairwise:
        return
    matrix = np.full((len(METHOD_ORDER), len(METHOD_ORDER)), np.nan)
    index = {m: i for i, m in enumerate(METHOD_ORDER)}
    for row in pairwise.itertuples(index=False):
        a, b = index[row.method_a], index[row.method_b]
        matrix[a, b] = matrix[b, a] = getattr(row, value)
    # Omit the uninformative self-comparisons to make the off-diagonal readable.
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    cmap = plt.get_cmap("Blues" if agreement_scale else "YlOrBr").copy()
    cmap.set_bad("#F1F3F5")
    finite = matrix[np.isfinite(matrix)]
    vmax = 1.0 if agreement_scale else max(float(finite.max()) if finite.size else 0, 1e-9)
    image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(METHOD_ORDER)), [SHORT_NAMES[m] for m in METHOD_ORDER])
    ax.set_yticks(range(len(METHOD_ORDER)), METHOD_ORDER)
    ax.set_title(title, loc="left", fontsize=11, pad=15)
    fig.colorbar(image, ax=ax, shrink=.75,
                 label="Median F1 (higher = more agreement)" if agreement_scale else
                       "Normalised RMSE (lower = closer)")
    _annotate_matrix(ax, matrix, image)
    fig.tight_layout()
    _save_plot(fig, path)


def _elapsed_time_residual_autocorrelation(
    obs: pd.DataFrame,
    prediction: np.ndarray,
    method: str,
    source: str,
    lag_edges: tuple[float, ...] = (9.0, 30.0, 90.0, 180.0),
) -> list[dict]:
    """Correlate standardised residual pairs grouped by elapsed-time separation."""
    obs = obs.sort_values("time").reset_index(drop=True)
    times = obs["time"].to_numpy(dtype=float)
    flux = obs["flux"].to_numpy(dtype=float)
    errors = obs["error"].to_numpy(dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    valid = (
        np.isfinite(times) & np.isfinite(flux) & np.isfinite(errors)
        & (errors > 0) & np.isfinite(prediction)
    )
    times = times[valid]
    residuals = (flux[valid] - prediction[valid]) / errors[valid]

    labels = [
        f"(0, {lag_edges[0]:g}]",
        *[
            f"({lag_edges[i]:g}, {lag_edges[i + 1]:g}]"
            for i in range(len(lag_edges) - 1)
        ],
    ]
    pair_x = [[] for _ in labels]
    pair_y = [[] for _ in labels]

    if len(times) >= 2:
        max_lag = float(lag_edges[-1])
        for i in range(len(times) - 1):
            j_end = int(np.searchsorted(times, times[i] + max_lag, side="right"))
            if j_end <= i + 1:
                continue
            js = np.arange(i + 1, j_end)
            deltas = times[js] - times[i]
            bin_indices = np.searchsorted(np.asarray(lag_edges), deltas, side="left")
            positive = deltas > 0
            for bin_index in range(len(labels)):
                selected = positive & (bin_indices == bin_index)
                if np.any(selected):
                    selected_js = js[selected]
                    pair_x[bin_index].extend([float(residuals[i])] * len(selected_js))
                    pair_y[bin_index].extend(residuals[selected_js].astype(float).tolist())

    rows = []
    lower = 0.0
    for index, label in enumerate(labels):
        upper = float(lag_edges[index])
        x = np.asarray(pair_x[index], dtype=float)
        y = np.asarray(pair_y[index], dtype=float)
        corr = np.nan
        if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
            corr = float(np.corrcoef(x, y)[0, 1])
        rows.append({
            "source": source,
            "method": method,
            "lag_bin": label,
            "lag_lower_exclusive_days": lower,
            "lag_upper_inclusive_days": upper,
            "n_residual_pairs": int(len(x)),
            "residual_pair_correlation": corr,
        })
        lower = upper
    return rows


def _aggregate_residual_autocorrelation(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby(["method", "lag_bin", "lag_lower_exclusive_days", "lag_upper_inclusive_days"],
                      as_index=False)
        .agg(
            n_sources=("residual_pair_correlation", "count"),
            median_correlation=("residual_pair_correlation", "median"),
            q25_correlation=("residual_pair_correlation", lambda x: x.quantile(0.25)),
            q75_correlation=("residual_pair_correlation", lambda x: x.quantile(0.75)),
            median_n_residual_pairs=("n_residual_pairs", "median"),
        )
    )


def _make_hmm_diagnostics(
    source_segments: dict,
    source_observations: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarise decoded HMM state runs; these are not the fitted HMM transmat_."""
    dwell_rows, occupancy_rows, source_rows, transition_rows = [], [], [], []
    all_state_values = []
    for candidate_segments in source_segments.get("HMM", {}).values():
        if "state" in candidate_segments:
            candidate_states = pd.to_numeric(candidate_segments["state"], errors="coerce").dropna()
            all_state_values.extend(int(round(value)) for value in candidate_states)
    global_states = sorted(set(all_state_values))

    for source, raw_segments in source_segments.get("HMM", {}).items():
        if "state" not in raw_segments or raw_segments["state"].notna().sum() == 0:
            continue
        obs = source_observations[source].sort_values("time").reset_index(drop=True)
        seg = raw_segments.sort_values(["start", "end"]).reset_index(drop=True)
        assigned, _ = _assign_observations(obs, seg)
        counts = np.bincount(assigned[assigned >= 0], minlength=len(seg))
        states = pd.to_numeric(seg["state"], errors="coerce").to_numpy(dtype=float)

        valid_state_runs = []
        state_observation_counts: dict[int, int] = {}
        state_run_counts: dict[int, int] = {}
        state_dwell_obs: dict[int, list[int]] = {}
        state_dwell_days: dict[int, list[float]] = {}

        for run_index, row in seg.iterrows():
            state_value = states[run_index]
            if not np.isfinite(state_value):
                continue
            state = int(round(state_value))
            run_mask = assigned == run_index
            run_times = obs.loc[run_mask, "time"].to_numpy(dtype=float)
            n_obs = int(counts[run_index])
            observed_span = (
                float(run_times[-1] - run_times[0]) if len(run_times) > 1 else 0.0
            )
            run_gaps = np.diff(run_times) if len(run_times) > 1 else np.array([], dtype=float)
            dwell_rows.append({
                "source": source,
                "run_index": run_index + 1,
                "state": state,
                "n_assigned_observations": n_obs,
                "segment_interval_days": float(max(row["end"] - row["start"], 0.0)),
                "observed_span_days": observed_span,
                "max_within_run_gap_days": float(np.max(run_gaps)) if len(run_gaps) else np.nan,
            })
            valid_state_runs.append(state)
            state_observation_counts[state] = state_observation_counts.get(state, 0) + n_obs
            state_run_counts[state] = state_run_counts.get(state, 0) + 1
            state_dwell_obs.setdefault(state, []).append(n_obs)
            state_dwell_days.setdefault(state, []).append(observed_span)

        total_state_observations = sum(state_observation_counts.values())
        for state in sorted(state_observation_counts):
            occupancy_rows.append({
                "source": source,
                "state": state,
                "n_assigned_observations": state_observation_counts[state],
                "observation_occupancy_fraction": (
                    state_observation_counts[state] / total_state_observations
                    if total_state_observations else np.nan
                ),
                "n_runs": state_run_counts[state],
                "median_dwell_observations": float(np.median(state_dwell_obs[state])),
                "median_dwell_observed_span_days": float(np.median(state_dwell_days[state])),
            })

        # Expand run-level state labels back to the observation sequence so the
        # decoded transition matrix includes self-transitions as well as switches.
        observation_states = np.full(len(obs), np.nan, dtype=float)
        for run_index, state_value in enumerate(states):
            if np.isfinite(state_value):
                observation_states[assigned == run_index] = int(round(state_value))

        adjacent_valid = (
            np.isfinite(observation_states[:-1]) & np.isfinite(observation_states[1:])
        )
        decoded_pairs = list(zip(
            observation_states[:-1][adjacent_valid].astype(int),
            observation_states[1:][adjacent_valid].astype(int),
        ))
        source_transitions: dict[tuple[int, int], int] = {}
        for from_state, to_state in decoded_pairs:
            key = (int(from_state), int(to_state))
            source_transitions[key] = source_transitions.get(key, 0) + 1

        outgoing: dict[int, int] = {}
        for (from_state, _), count in source_transitions.items():
            outgoing[from_state] = outgoing.get(from_state, 0) + count

        for from_state in global_states:
            if outgoing.get(from_state, 0) == 0:
                continue
            for to_state in global_states:
                count = source_transitions.get((from_state, to_state), 0)
                transition_rows.append({
                    "source": source,
                    "from_state": from_state,
                    "to_state": to_state,
                    "n_transitions": count,
                    "decoded_path_empirical_transition_probability": (
                        count / outgoing[from_state]
                    ),
                })

        n_state_changes = int(sum(
            count for (from_state, to_state), count in source_transitions.items()
            if from_state != to_state
        ))
        n_observation_transitions = int(sum(source_transitions.values()))
        observed_span_days = (
            float(obs["time"].max() - obs["time"].min()) if len(obs) > 1 else 0.0
        )
        source_summary = {
            "source": source,
            "n_state_runs": len(valid_state_runs),
            "n_state_changes": n_state_changes,
            "n_observation_state_transitions": n_observation_transitions,
            "n_assigned_state_observations": int(total_state_observations),
            "state_changes_per_100_observations": (
                100.0 * n_state_changes / total_state_observations
                if total_state_observations > 0 else np.nan
            ),
            "state_changes_per_observed_year": (
                n_state_changes / (observed_span_days / 365.25)
                if observed_span_days > 0 else np.nan
            ),
            "median_dwell_observations": (
                float(np.median([x for values in state_dwell_obs.values() for x in values]))
                if state_dwell_obs else np.nan
            ),
            "median_dwell_observed_span_days": (
                float(np.median([x for values in state_dwell_days.values() for x in values]))
                if state_dwell_days else np.nan
            ),
            "n_states_observed": len(state_observation_counts),
        }
        for state in sorted(state_observation_counts):
            source_summary[f"state_{state}_occupancy_fraction"] = (
                state_observation_counts[state] / total_state_observations
                if total_state_observations else np.nan
            )
        source_rows.append(source_summary)

    dwell = pd.DataFrame(dwell_rows)
    occupancy = pd.DataFrame(occupancy_rows)
    source_summary = pd.DataFrame(source_rows)
    transitions_by_source = pd.DataFrame(transition_rows)

    if transitions_by_source.empty:
        transition_summary = pd.DataFrame()
    else:
        transition_summary = (
            transitions_by_source.groupby(["from_state", "to_state"], as_index=False)
            .agg(
                n_transitions=("n_transitions", "sum"),
                n_sources=("source", "nunique"),
                median_source_transition_probability=(
                    "decoded_path_empirical_transition_probability", "median"
                ),
                q25_source_transition_probability=(
                    "decoded_path_empirical_transition_probability",
                    lambda x: x.quantile(0.25),
                ),
                q75_source_transition_probability=(
                    "decoded_path_empirical_transition_probability",
                    lambda x: x.quantile(0.75),
                ),
            )
        )
        outgoing_totals = transition_summary.groupby("from_state")["n_transitions"].transform("sum")
        transition_summary["decoded_path_empirical_transition_probability"] = (
            transition_summary["n_transitions"] / outgoing_totals
        )

    return dwell, occupancy, source_summary, transitions_by_source, transition_summary


def _bootstrap_median_interval(
    values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    estimate = float(np.median(values))
    if len(values) == 1 or n_bootstrap <= 0:
        return estimate, np.nan, np.nan
    draws = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
    bootstrap_medians = np.median(draws, axis=1)
    lower, upper = np.quantile(bootstrap_medians, [0.025, 0.975])
    return estimate, float(lower), float(upper)


BOOTSTRAP_METRICS = [
    "n_segments",
    "median_segment_duration_days",
    "standardised_residual_rmse",
    "amplitude_retention",
]


def _make_bootstrap_summaries(
    metrics: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if metrics.empty:
        return pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(seed)
    method_rows = []
    for method in METHOD_ORDER:
        group = metrics.loc[metrics["method"] == method]
        for metric in BOOTSTRAP_METRICS:
            values = group[metric].to_numpy(dtype=float)
            estimate, lower, upper = _bootstrap_median_interval(values, n_bootstrap, rng)
            method_rows.append({
                "method": method,
                "metric": metric,
                "n_sources": int(np.isfinite(values).sum()),
                "median": estimate,
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "bootstrap_replicates": n_bootstrap,
                "random_seed": seed,
            })

    paired_rows = []
    for metric in BOOTSTRAP_METRICS:
        pivot = metrics.pivot(index="source", columns="method", values=metric)
        for method_a, method_b in itertools.combinations(METHOD_ORDER, 2):
            if method_a not in pivot or method_b not in pivot:
                continue
            paired = (pivot[method_a] - pivot[method_b]).dropna().to_numpy(dtype=float)
            estimate, lower, upper = _bootstrap_median_interval(paired, n_bootstrap, rng)
            paired_rows.append({
                "method_a": method_a,
                "method_b": method_b,
                "metric": metric,
                "difference_definition": "method_a minus method_b",
                "n_paired_sources": int(len(paired)),
                "median_paired_difference": estimate,
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "bootstrap_replicates": n_bootstrap,
                "random_seed": seed,
            })

    return pd.DataFrame(method_rows), pd.DataFrame(paired_rows)


def _select_representative_sources(
    observations: dict,
    predictions: dict,
) -> dict[str, tuple[str, int]]:
    common = set(observations)
    for method in METHOD_ORDER:
        common &= set(predictions.get(method, {}))
    if not common:
        return {}

    counts = {source: len(observations[source]) for source in common}
    ordered = sorted(common, key=lambda source: (counts[source], source))
    count_values = np.asarray([counts[source] for source in ordered], dtype=float)
    selected: dict[str, tuple[str, int]] = {}
    used: set[str] = set()

    for label, quantile in (("low", 0.10), ("median", 0.50), ("high", 0.90)):
        target = float(np.quantile(count_values, quantile))
        candidates = sorted(
            ordered,
            key=lambda source: (abs(counts[source] - target), counts[source], source),
        )
        chosen = next((source for source in candidates if source not in used), candidates[0])
        used.add(chosen)
        selected[label] = (chosen, counts[chosen])
    return selected


def _plot_representative_source(source, label, observations, predictions, segments,
                                output_path, gap_threshold_days):
    obs = observations[source].sort_values("time").reset_index(drop=True)
    times = obs["time"].to_numpy(dtype=float)
    flux = obs["flux"].to_numpy(dtype=float)
    positive = flux[np.isfinite(flux) & (flux > 0)]
    scale = float(np.median(positive)) if len(positive) else 1.0
    # Normalise for readability only. Symmetric log keeps extreme and negative
    # fitted values visible without clipping or compressing the baseline to zero.
    all_values = np.concatenate([flux/scale] + [predictions[m][source]/scale for m in METHOD_ORDER])
    finite = all_values[np.isfinite(all_values)]
    max_abs = max(1.0, float(np.max(np.abs(finite)))) if len(finite) else 1.0
    use_symlog = max_abs > 20 or (len(finite) and np.min(finite) < -2)
    fig, axes = plt.subplots(3, 2, figsize=(16, 11), sharex=True, sharey=True)
    gap_indices = np.flatnonzero(np.diff(times) > gap_threshold_days)
    for ax, method in zip(axes.ravel(), METHOD_ORDER):
        colour = METHOD_COLOURS[method]
        ax.scatter(times, flux/scale, s=9, alpha=.38, color="#64748B", zorder=2, linewidths=0)
        # Use a thin coverage strip rather than dozens of full-height grey bands.
        for k in gap_indices:
            ax.axvspan(times[k], times[k+1], ymin=0, ymax=.035, color="#9AA6B2", alpha=.65, lw=0)
        seg = segments[method].get(source)
        if seg is not None and not seg.empty:
            seg = seg.sort_values(["start", "end"]).reset_index(drop=True)
            pred = predictions[method][source] / scale
            assigned, _ = _assign_observations(obs, seg)
            for j in range(len(seg)):
                indices = np.flatnonzero((assigned == j) & np.isfinite(pred))
                if not len(indices):
                    continue
                for run in np.split(indices, np.flatnonzero(np.diff(times[indices]) > gap_threshold_days)+1):
                    if len(run):
                        ax.plot(times[run], pred[run], color=colour, linewidth=1.8,
                                marker="o" if len(run)==1 else None, markersize=3, zorder=4)
            for cp in seg["start"].to_numpy(dtype=float)[1:]:
                right = int(np.searchsorted(times, cp, side="right")); left = right-1
                in_gap = (left >= 0 and right < len(times) and times[left] < cp < times[right]
                          and times[right]-times[left] > gap_threshold_days)
                ax.axvline(cp, ymin=.90, ymax=.99, color=colour, alpha=.8, lw=1.1,
                           linestyle="--" if in_gap else "-")
        ax.set_title(f"{method}  ·  {len(seg) if seg is not None else 0} segments", loc="left", color=colour)
        ax.axhline(1, color="#9AA6B2", lw=.7, linestyle=":", zorder=1)
        ax.grid(axis="y", alpha=.65); ax.set_axisbelow(True)
        if use_symlog:
            ax.set_yscale("symlog", linthresh=1, linscale=.8)
        ax.margins(x=.005, y=.09)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.0f}"))
    fig.suptitle(f"{label.capitalize()} detection count  |  {source}  |  {len(obs):,} detections",
                 fontsize=16, fontweight="semibold", y=.985)
    fig.supxlabel("Time (MJD)", y=.055)
    fig.supylabel("Flux / median detected flux" + (" (symmetric log)" if use_symlog else ""), x=.015)
    handles = [Line2D([0],[0], marker="o", color="none", markerfacecolor="#64748B", markeredgecolor="none", label="Detections"),
               Line2D([0],[0], color="#243447", lw=2, label="Fitted flux (method colour)"),
               Line2D([0],[0], color="#243447", lw=1, label="Change point (top)"),
               Line2D([0],[0], color="#243447", lw=1, ls="--", label="Change point in long gap"),
               Patch(facecolor="#9AA6B2", label=f"Gap > {gap_threshold_days:g} d (bottom)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5,.951), ncol=5, fontsize=9)
    fig.text(.5,.018,f"Median detected flux = {scale:.3g} ph/cm²/s. Shared axes; fitted lines stop at long gaps. No values clipped.",ha="center",fontsize=9)
    fig.tight_layout(rect=(.04,.075,1,.91))
    _save_plot(fig, output_path)


def _plot_bootstrap_intervals(bootstrap_summary, output_path):
    if bootstrap_summary.empty:
        return
    labels = {"n_segments": "Segments per source", "median_segment_duration_days": "Median segment duration (days)",
              "standardised_residual_rmse": "Standardised residual RMSE", "amplitude_retention": "Amplitude retention"}
    fig, axes = plt.subplots(2,2, figsize=(13,8))
    for ax, metric in zip(axes.ravel(), BOOTSTRAP_METRICS):
        subset = bootstrap_summary[bootstrap_summary["metric"] == metric].set_index("method").reindex(METHOD_ORDER)
        for i, (method,row) in enumerate(subset.iterrows()):
            y, lo, hi = row["median"], row["ci_95_lower"], row["ci_95_upper"]
            if not np.isfinite(y): continue
            colour = METHOD_COLOURS[method]
            if np.isfinite(lo) and np.isfinite(hi):
                ax.hlines(i,lo,hi,color=colour,lw=2.4)
                ax.plot([lo,hi],[i,i],"|",color=colour,ms=8)
            ax.scatter(y,i,color=colour,s=35,zorder=3)
        ax.set_yticks(range(len(METHOD_ORDER)), METHOD_ORDER)
        ax.set_ylim(len(METHOD_ORDER)-.45,-.55)
        ax.set_title(labels[metric],loc="left")
        if metric in {"n_segments", "median_segment_duration_days"} and (subset["median"].dropna()>0).all():
            ax.set_xscale("log"); ax.set_xlabel("Log scale")
        ax.grid(axis="x"); ax.set_axisbelow(True)
    fig.suptitle("Method medians and 95% source-bootstrap intervals", fontsize=16, fontweight="semibold")
    fig.text(.5,.015,"Points are source-level medians; bars are percentile bootstrap intervals, not source-to-source spread.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.94))
    _save_plot(fig, output_path)


def _plot_residual_autocorrelation(summary, output_path):
    if summary.empty: return
    lag_order = summary[["lag_bin","lag_lower_exclusive_days"]].drop_duplicates().sort_values("lag_lower_exclusive_days")["lag_bin"].tolist()
    x = np.arange(len(lag_order))
    fig, axes = plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
    for ax, method in zip(axes.ravel(),METHOD_ORDER):
        sub = summary[summary["method"]==method].set_index("lag_bin").reindex(lag_order)
        colour=METHOD_COLOURS[method]
        med=sub["median_correlation"].to_numpy(dtype=float)
        q25=sub["q25_correlation"].to_numpy(dtype=float); q75=sub["q75_correlation"].to_numpy(dtype=float)
        ax.fill_between(x,q25,q75,color=colour,alpha=.15)
        ax.plot(x,med,color=colour,marker="o",lw=2)
        ax.axhline(0,color="#7B8794",ls="--",lw=.8)
        ax.set_title(method,loc="left",color=colour)
        ax.set_xticks(x,lag_order,fontsize=8)
        ax.grid(axis="y"); ax.set_axisbelow(True)
    fig.suptitle("Residual dependence by elapsed-time lag",fontsize=16,fontweight="semibold")
    fig.supxlabel("Elapsed-time lag (days)",y=.055); fig.supylabel("Residual-pair Pearson correlation",x=.015)
    fig.text(.5,.016,"Lines: median across sources. Shading: interquartile range across sources (not a confidence interval).",ha="center",fontsize=9)
    fig.tight_layout(rect=(.04,.08,1,.94)); _save_plot(fig,output_path)


def _plot_event_error_distribution(event_matches, output_path, preferred_tolerance=9.0, matching_mode="time_tolerance"):
    if event_matches.empty: return
    selected=event_matches[event_matches["matching_mode"]==matching_mode]
    if selected.empty: return
    tolerances=np.sort(selected["tolerance_days"].unique())
    tol=float(tolerances[np.argmin(np.abs(tolerances-preferred_tolerance))])
    selected=selected[np.isclose(selected["tolerance_days"],tol)]
    fig,ax=plt.subplots(figsize=(11,8))
    pairs=_pair_order()
    for i,(a,b) in enumerate(pairs):
        vals=selected.loc[(selected["method_a"]==a)&(selected["method_b"]==b),"absolute_error_days"].to_numpy(dtype=float)
        vals=vals[np.isfinite(vals)]
        if not len(vals):
            ax.text(1.025,i,"0 / –",transform=ax.get_yaxis_transform(),va="center",fontsize=9); continue
        q25,med,q75,p90=np.quantile(vals,[.25,.5,.75,.9])
        ax.hlines(i,0,p90,color="#ADB9C5",lw=1.3)
        ax.hlines(i,q25,q75,color="#0072B2",lw=5)
        ax.scatter(med,i,color="#182B3A",s=27,zorder=3)
        ax.plot(p90,i,"|",color="#52657A",ms=9)
        ax.text(1.025,i,f"{len(vals):,} / {100*np.mean(vals<=1e-8):.0f}%",transform=ax.get_yaxis_transform(),va="center",fontsize=9)
    ax.set_yticks(range(len(pairs)),[_pair_label(a,b) for a,b in pairs]); ax.set_ylim(len(pairs)-.5,-.5)
    ax.set_xlim(left=-.15); ax.set_xlabel("Absolute timing difference (days)")
    ax.set_title(f"Matched boundary timing | {matching_mode.replace('_',' ')} | ±{tol:g} days",loc="left")
    ax.text(1.025,1.015,"Matches / exact",transform=ax.transAxes,fontsize=9,fontweight="semibold")
    ax.grid(axis="x"); ax.set_axisbelow(True)
    fig.text(.02,.018,"Pooled matched events only. Thick bar: IQR; dot: median; thin bar ends at P90. Exact = zero timing difference.",fontsize=9)
    fig.tight_layout(rect=(0,.05,.85,1)); _save_plot(fig,output_path)


def _plot_tolerance_sensitivity(boundary_summary, output_path, value="median_f1",
                                title="Time-based change-point agreement vs matching tolerance"):
    if boundary_summary.empty or value not in boundary_summary: return
    # A labelled pair-by-tolerance matrix is easier to read than 15 overlapping curves.
    pairs=_pair_order(); tolerances=sorted(boundary_summary["tolerance_days"].unique())
    array=np.full((len(pairs),len(tolerances)),np.nan)
    for i,(a,b) in enumerate(pairs):
        sub=boundary_summary[(boundary_summary["method_a"]==a)&(boundary_summary["method_b"]==b)].set_index("tolerance_days")
        array[i,:]=sub[value].reindex(tolerances).to_numpy(dtype=float)
    fig,ax=plt.subplots(figsize=(8.5,8.5))
    cmap=plt.get_cmap("Blues").copy(); cmap.set_bad("#F1F3F5")
    image=ax.imshow(array,cmap=cmap,vmin=0,vmax=1,aspect="auto")
    ax.set_xticks(range(len(tolerances)),[f"±{t:g} d" for t in tolerances]); ax.set_yticks(range(len(pairs)),[_pair_label(a,b) for a,b in pairs])
    ax.set_xlabel("Matching tolerance"); ax.set_title(title,loc="left",fontsize=11,pad=12)
    _annotate_matrix(ax,array,image); fig.colorbar(image,ax=ax,shrink=.8,label="Median source-level F1")
    fig.text(.02,.015,"BB = Bayesian Blocks; Linear/RBF/Normal = PELT variants. Identical scale for both matching modes.",fontsize=8.5)
    fig.tight_layout(rect=(0,.05,1,1)); _save_plot(fig,output_path)


def _aggregate_change_point_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for method, group in frame.groupby("method", sort=False):
        in_gap = group[group["is_in_observation_gap"]]
        rows.append({
            "method": method,
            "n_change_points": int(len(group)),
            "pct_change_points_between_observations": (
                100.0 * float(group["is_in_observation_gap"].mean())
            ),
            "pct_change_points_in_gap_gt_9_days": 100.0 * float(group["in_gap_gt_9_days"].mean()),
            "pct_change_points_in_gap_gt_30_days": 100.0 * float(group["in_gap_gt_30_days"].mean()),
            "pct_change_points_in_gap_gt_90_days": 100.0 * float(group["in_gap_gt_90_days"].mean()),
            "median_gap_length_days_when_boundary_in_gap": (
                float(in_gap["observational_gap_days"].median()) if not in_gap.empty else np.nan
            ),
            "p90_gap_length_days_when_boundary_in_gap": (
                float(in_gap["observational_gap_days"].quantile(0.90))
                if not in_gap.empty else np.nan
            ),
        })
    return pd.DataFrame(rows).set_index("method").reindex(METHOD_ORDER).reset_index()


def _plot_change_point_gap_support(summary, output_path):
    if summary.empty: return
    sub=summary.set_index("method").reindex(METHOD_ORDER)
    fig,axes=plt.subplots(1,3,figsize=(12,4.6),sharey=True)
    columns=[("pct_change_points_in_gap_gt_9_days",">9 days"),("pct_change_points_in_gap_gt_30_days",">30 days"),("pct_change_points_in_gap_gt_90_days",">90 days")]
    for ax,(column,title) in zip(axes,columns):
        vals=sub[column].to_numpy(dtype=float)
        ax.barh(range(len(METHOD_ORDER)),vals,color=[METHOD_COLOURS[m] for m in METHOD_ORDER],alpha=.85)
        ax.set_yticks(range(len(METHOD_ORDER)),METHOD_ORDER); ax.set_title(title,loc="left")
        for i,v in enumerate(vals):
            if np.isfinite(v): ax.annotate(f"{v:.1f}%",(v,i),xytext=(4,0),textcoords="offset points",va="center",fontsize=9)
        ax.set_xlim(0,max(1,float(np.nanmax(vals))*1.28) if np.isfinite(vals).any() else 1)
        ax.grid(axis="x"); ax.set_axisbelow(True); ax.set_xlabel("% of boundaries")
    axes[0].invert_yaxis()
    fig.suptitle("Change points inside long detection gaps",fontsize=15,fontweight="semibold")
    fig.text(.5,.015,"Pooled boundaries; thresholds are cumulative. Each panel has its own horizontal scale.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.93)); _save_plot(fig,output_path)


def _plot_hmm_transition_matrix(transition_summary, output_path):
    if transition_summary.empty: return
    states=sorted(set(transition_summary["from_state"])|set(transition_summary["to_state"]))
    matrix=np.full((len(states),len(states)),np.nan); idx={s:i for i,s in enumerate(states)}
    for state in states:
        if (transition_summary["from_state"]==state).any(): matrix[idx[state],:]=0
    for row in transition_summary.itertuples(index=False):
        matrix[idx[row.from_state],idx[row.to_state]]=row.decoded_path_empirical_transition_probability
    fig,ax=plt.subplots(figsize=(6,5))
    image=ax.imshow(matrix,cmap="Blues",vmin=0,vmax=1,aspect="equal")
    names=[f"State {st:g}" for st in states]
    ax.set_xticks(range(len(states)),names); ax.set_yticks(range(len(states)),names)
    ax.set_xlabel("Next decoded state"); ax.set_ylabel("Current decoded state")
    ax.set_title("HMM decoded-path transitions",loc="left",pad=12)
    _annotate_matrix(ax,matrix,image)
    fig.colorbar(image,ax=ax,shrink=.8,label="Pooled empirical probability")
    fig.text(.5,.015,"Consecutive detections, not fixed time steps; not the fitted HMM transition matrix.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1)); _save_plot(fig,output_path)


def _plot_hmm_state_summary(occupancy, output_path):
    if occupancy.empty: return
    fig,axes=plt.subplots(1,3,figsize=(12,4.8))
    states=sorted(occupancy["state"].dropna().unique())
    fields=[("observation_occupancy_fraction","Fraction of detections",False),
            ("median_dwell_observations","Median run length (detections)",True),
            ("median_dwell_observed_span_days","Median run observed span (days)",True)]
    for ax,(field,title,log) in zip(axes,fields):
        data=[]; labels=[]
        for state in states:
            v=occupancy.loc[occupancy["state"]==state,field].to_numpy(dtype=float); v=v[np.isfinite(v)]
            if len(v): data.append(v); labels.append(f"State {state:g}")
        if not data: continue
        bp=ax.boxplot(data,patch_artist=True,showfliers=False,medianprops={"color":"#182B3A","linewidth":2})
        for i,b in enumerate(bp["boxes"]): b.set_facecolor(["#0072B2","#D55E00","#009E73","#8C5DA5"][i%4]); b.set_alpha(.55)
        ax.set_xticks(range(1,len(labels)+1),labels); ax.set_title(title,fontsize=10,loc="left")
        if log:
            if any(np.any(v<=0) for v in data):
                ax.set_yscale("symlog",linthresh=1)
                if all(np.min(v)>=0 for v in data): ax.set_ylim(bottom=0)
            else: ax.set_yscale("log")
        else: ax.set_ylim(0,1)
        ax.grid(axis="y"); ax.set_axisbelow(True)
    fig.suptitle("HMM state occupancy and dwell times",fontsize=15,fontweight="semibold")
    fig.text(.5,.015,"One value per source and decoded state. Run spans include gaps; states ordered by fitted mean flux. Boxes: IQR.",ha="center",fontsize=8.5)
    fig.tight_layout(rect=(0,.055,1,.93)); _save_plot(fig,output_path)


def _write_interpretation(
    output_dir: Path,
    method_summary: pd.DataFrame,
    curve_summary: pd.DataFrame,
    boundary_summary: pd.DataFrame,
    event_summary: pd.DataFrame,
    cp_gap_summary: pd.DataFrame,
    residual_acf_summary: pd.DataFrame,
    hmm_source_summary: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    representative_sources: pd.DataFrame,
    tolerances: list[float],
    gap_threshold_days: float,
    bootstrap_replicates: int,
    random_seed: int,
) -> None:
    lines = [
        "FERMI LIGHT-CURVE SEGMENTATION COMPARISON",
        "=" * 44,
        "",
        "This report is descriptive; it does not identify a universally best method.",
        "Method summaries use one source-level row per method, so sources receive equal weight.",
        "Each method uses sources with a valid saved segmentation; paired comparisons use",
        "only sources available for both methods (reported as n_sources / n_paired_sources).",
        "",
        "PELT-linear reconstruction",
        "-" * 44,
        "* segment_light_curves.py exports fitted_flux as the segment mean flux.",
        "* It estimates slope_per_day by OLS with an intercept.",
        "* Therefore fitted_flux is exact at the mean observation time in the segment.",
        "* Reconstruction: fitted_flux + slope_per_day * (time - mean_segment_observation_time).",
        "",
        f"Time-based change-point tolerances (days): {', '.join(f'{x:g}' for x in tolerances)}",
        f"Representative-plot gap threshold: {gap_threshold_days:g} days",
        f"Source-bootstrap replicates: {bootstrap_replicates}; random seed: {random_seed}",
        "",
        "METHOD SUMMARY (source-level medians, with Q25/Q75 columns in method_summary.csv)",
        "-" * 44,
    ]
    if not method_summary.empty:
        preferred = [
            "method", "n_sources",
            "n_segments_median", "n_change_points_median",
            "median_segment_duration_days_median",
            "standardised_residual_rmse_median", "amplitude_retention_median",
            "fit_coverage_median",
        ]
        available = [column for column in preferred if column in method_summary.columns]
        lines.append(method_summary[available].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    else:
        lines.append("No source-level method metrics were produced.")
    lines.extend([
        "",
        "CHANGE-POINT AGREEMENT",
        "-" * 44,
        "* Time-based F1 uses optimal one-to-one matching within the stated tolerance.",
        "* Gap-aware F1 additionally allows two boundaries inside the same observation-to-observation",
        "  interval to match, because their exact ordering within that unobserved interval is unresolved.",
        "* The time-based and gap-aware scores are both retained; neither replaces the other.",
        "* Both methods having no change points is assigned F1=1; only one method having boundaries",
        "  is assigned F1=0. Pairwise summary CSVs separately report these cases.",
        "* Event timing errors are conditional on matched events. The event-level CSV gives the",
        "  pooled distribution; source-level pairwise CSVs retain one row per source.",
        "* change_point_gap_diagnostics.csv records whether each boundary falls in a gap >9, >30, >90 days.",
        "",
        "RESIDUAL DEPENDENCE",
        "-" * 44,
        "* Residual correlations use standardised residual pairs grouped by elapsed-time separation:",
        "  (0,9], (9,30], (30,90], and (90,180] days.",
        "* Correlations are calculated within each source and then summarised across sources.",
        "* This is a descriptive irregular-time pair-correlation diagnostic, not a conventional",
        "  equally-spaced sample ACF or a formal significance test.",
        "",
        "HMM STATE DIAGNOSTICS",
        "-" * 44,
        "* State labels are retained from the saved HMM segment output; labels are ordered by",
        "  increasing fitted emission mean in the segmentation implementation.",
        "* Dwell times are measured in assigned observations and observed first-to-last calendar span.",
        "* Transition probabilities are empirical adjacent-observation transitions in the",
        "  decoded state sequence (including self-transitions), not the fitted GaussianHMM",
        "  transmat_ (that matrix is not exported by the segmentation CSV).",
        "* State-change rates per observed year are descriptive; the HMM transitions occur",
        "  per observation step, so they are not continuous-time transition intensities.",
        "",
        "BOOTSTRAP UNCERTAINTY",
        "-" * 44,
        "* Confidence intervals resample sources, not individual observations.",
        "* Paired method differences are method_a minus method_b on sources available for both methods.",
        "* These intervals quantify source-to-source sampling uncertainty, not segmentation uncertainty.",
        "",
        "GENERAL CAVEATS",
        "-" * 44,
        "* Calendar segment duration and first-to-last observed span are both reported;",
        "  they differ when segments contain long intervals without detections.",
        "* Segment counts and durations describe saved segmentations, not latent truth.",
        "* Residual diagnostics are evaluated only at cleaned detected observations.",
        "* Standardised residuals exclude missing/non-positive flux errors.",
        "* Pairwise fitted-curve distances compare predictions at common observed times.",
        "* High in-sample fit quality is not by itself evidence of good forecasting.",
        "* This comparison does not assess censored/non-detection bins or synthetic recovery.",
        "",
    ])

    if not representative_sources.empty:
        lines.extend([
            "REPRESENTATIVE SOURCES",
            "-" * 44,
            representative_sources.to_string(index=False),
            "",
        ])

    if not curve_summary.empty:
        lines.extend([
            "CLOSEST FITTED-CURVE PAIRS (median source-level normalised RMSE)",
            "-" * 44,
            curve_summary.sort_values("normalised_curve_rmse")
            .head(10).to_string(index=False, float_format=lambda x: f"{x:.4g}"),
            "",
        ])

    if not boundary_summary.empty:
        for tolerance in tolerances:
            subset = boundary_summary[
                np.isclose(boundary_summary["tolerance_days"], tolerance)
            ]
            lines.extend([
                f"TOP TIME-BASED CHANGE-POINT AGREEMENT PAIRS (±{tolerance:g} days)",
                "-" * 44,
                subset.sort_values("median_f1", ascending=False)[
                    ["method_a", "method_b", "median_f1",
                     "median_f1_conditional_both_have_change_points",
                     "pct_sources_with_any_match", "pct_sources_no_change_points_either",
                     "pct_sources_with_change_points_both"]
                ].head(8).to_string(index=False, float_format=lambda x: f"{x:.4g}"),
                "",
                f"TOP GAP-AWARE CHANGE-POINT AGREEMENT PAIRS (±{tolerance:g} days)",
                "-" * 44,
                subset.sort_values("gap_aware_median_f1", ascending=False)[
                    ["method_a", "method_b", "gap_aware_median_f1",
                     "gap_aware_median_f1_conditional_both_have_change_points",
                     "pct_sources_gap_aware_with_any_match", "pct_gap_aware_matches_same_gap_only"]
                ].head(8).to_string(index=False, float_format=lambda x: f"{x:.4g}"),
                "",
            ])

    if not cp_gap_summary.empty:
        lines.extend([
            "CHANGE-POINT LOCATION RELATIVE TO OBSERVATIONAL GAPS",
            "-" * 44,
            cp_gap_summary.to_string(index=False, float_format=lambda x: f"{x:.4g}"),
            "",
        ])

    if not event_summary.empty:
        lines.extend([
            "EVENT-LEVEL MATCHED CHANGE-POINT TIMING ERRORS",
            "-" * 44,
            event_summary.to_string(index=False, float_format=lambda x: f"{x:.4g}"),
            "",
        ])

    if not residual_acf_summary.empty:
        lines.extend([
            "ELAPSED-TIME RESIDUAL CORRELATION (source-level medians)",
            "-" * 44,
            residual_acf_summary.to_string(index=False, float_format=lambda x: f"{x:.4g}"),
            "",
        ])

    if not hmm_source_summary.empty:
        lines.extend([
            "HMM DECODED-STATE SUMMARY (source-level medians)",
            "-" * 44,
            hmm_source_summary.select_dtypes(include=[np.number]).median().to_string(
                float_format=lambda x: f"{x:.4g}"
            ),
            "",
        ])

    if not bootstrap_summary.empty:
        lines.extend([
            "BOOTSTRAP MEDIAN CONFIDENCE INTERVALS",
            "-" * 44,
            bootstrap_summary.to_string(index=False, float_format=lambda x: f"{x:.4g}"),
            "",
        ])

    (output_dir / "interpretation.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _plot_style()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observations",
        type=Path,
        default=None,
        help="Cleaned observations CSV (default: cleaned/cleaned_min300.csv).",
    )
    parser.add_argument(
        "--segmentation-dir",
        type=Path,
        default=None,
        help="Directory containing the six segments_*.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for comparison outputs (default: cleaned/segmentation/comparison).",
    )
    parser.add_argument(
        "--boundary-tolerances",
        type=float,
        nargs="+",
        default=[3.0, 9.0, 30.0, 60.0],
        help="Time-based change-point matching tolerances in days (default: 3 9 30 60).",
    )
    parser.add_argument(
        "--plot-gap-days",
        type=float,
        default=9.0,
        help="Split representative fitted curves across larger observational gaps (default: 9 days).",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=2000,
        help="Number of source-bootstrap replicates for 95% intervals (default: 2000).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible source bootstraps (default: 42).",
    )
    args = parser.parse_args()

    if not args.boundary_tolerances or any(
        not np.isfinite(value) or value < 0 for value in args.boundary_tolerances
    ):
        parser.error("--boundary-tolerances must contain finite, nonnegative values")
    args.boundary_tolerances = sorted(set(args.boundary_tolerances))
    if not np.isfinite(args.plot_gap_days) or args.plot_gap_days < 0:
        parser.error("--plot-gap-days must be finite and nonnegative")
    if args.bootstrap_replicates < 0:
        parser.error("--bootstrap-replicates must be nonnegative")

    repo_root = Path(__file__).resolve().parents[1]
    observations_path = args.observations or (repo_root / "cleaned" / "cleaned_min300.csv")
    segmentation_dir = args.segmentation_dir or (repo_root / "cleaned" / "segmentation")
    output_dir = args.output_dir or (segmentation_dir / "comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    observations = _read_observations(observations_path)
    all_segments: dict[str, pd.DataFrame] = {}
    for method, filename in METHOD_FILES.items():
        all_segments[method] = _read_segments(
            segmentation_dir / filename, method, observations
        )

    source_observations = {
        source: group.sort_values("time").reset_index(drop=True)
        for source, group in observations.groupby("source", sort=False)
    }

    source_segments: dict[str, dict[str, pd.DataFrame]] = {}
    method_predictions: dict[str, dict[str, np.ndarray]] = {
        method: {} for method in METHOD_ORDER
    }
    source_metrics = []
    segment_metrics = []
    residual_acf_rows = []

    for method in METHOD_ORDER:
        source_segments[method] = {
            source: group.sort_values("start").reset_index(drop=True)
            for source, group in all_segments[method].groupby("source", sort=False)
        }

        common_sources = sorted(
            set(source_observations) & set(source_segments[method])
        )
        for source in common_sources:
            obs = source_observations[source]
            seg = source_segments[method][source]
            seg_detail, metric, prediction = _evaluate_source_method(obs, seg, method)
            source_metrics.append(metric)
            segment_metrics.append(seg_detail)
            method_predictions[method][source] = prediction
            residual_acf_rows.extend(
                _elapsed_time_residual_autocorrelation(obs, prediction, method, source)
            )

    metrics = pd.DataFrame(source_metrics)
    segment_detail = (
        pd.concat(segment_metrics, ignore_index=True)
        if segment_metrics else pd.DataFrame()
    )

    # Include count metrics (segments and change points) as well as continuous metrics.
    summary_rows = []
    if not metrics.empty:
        numeric_columns = metrics.select_dtypes(include=[np.number]).columns.tolist()
        for method, group in metrics.groupby("method"):
            row = {"method": method, "n_sources": group["source"].nunique()}
            for column in numeric_columns:
                values = group[column].replace([np.inf, -np.inf], np.nan).dropna()
                row[f"{column}_median"] = float(values.median()) if len(values) else np.nan
                row[f"{column}_q25"] = float(values.quantile(0.25)) if len(values) else np.nan
                row[f"{column}_q75"] = float(values.quantile(0.75)) if len(values) else np.nan
            summary_rows.append(row)
    method_summary = (
        pd.DataFrame(summary_rows).set_index("method").reindex(METHOD_ORDER).reset_index()
        if summary_rows else pd.DataFrame(columns=["method", "n_sources"])
    )

    curve_pairs, boundary_pairs, event_matches = _make_pairwise_comparisons(
        method_predictions,
        source_segments,
        source_observations,
        args.boundary_tolerances,
    )
    curve_summary = _aggregate_pairwise(curve_pairs, "normalised_curve_rmse")
    boundary_summary = _aggregate_boundary_pairs(boundary_pairs)
    event_summary = _aggregate_event_matches(event_matches)
    cp_diagnostics = _collect_change_point_diagnostics(source_segments, source_observations)
    cp_gap_summary = _aggregate_change_point_gaps(cp_diagnostics)

    residual_acf = pd.DataFrame(residual_acf_rows)
    residual_acf_summary = _aggregate_residual_autocorrelation(residual_acf)

    (
        hmm_dwell,
        hmm_occupancy,
        hmm_source_summary,
        hmm_transitions_by_source,
        hmm_transition_summary,
    ) = _make_hmm_diagnostics(source_segments, source_observations)

    bootstrap_summary, paired_bootstrap_summary = _make_bootstrap_summaries(
        metrics, args.bootstrap_replicates, args.random_seed
    )

    representative_selection = _select_representative_sources(
        source_observations, method_predictions
    )
    representative_rows = []
    for label, (source, count) in representative_selection.items():
        target_quantile = {"low": 0.10, "median": 0.50, "high": 0.90}[label]
        representative_rows.append({
            "selection": label,
            "target_detection_count_quantile": target_quantile,
            "source": source,
            "n_detections": count,
        })
        representative_plot = output_dir / f"representative_source_{label}_detections.png"
        _plot_representative_source(
            source,
            label,
            source_observations,
            method_predictions,
            source_segments,
            representative_plot,
            args.plot_gap_days,
        )
        if label == "median":
            # Retain the previous filename as an alias for existing workflows.
            (output_dir / "representative_source_comparison.png").write_bytes(
                representative_plot.read_bytes()
            )
    representative_sources = pd.DataFrame(representative_rows)

    # Save all source-level and aggregate tables.
    outputs = {
        "source_method_metrics.csv": metrics,
        "segment_metrics.csv": segment_detail,
        "method_summary.csv": method_summary,
        "pairwise_curve_distances_by_source.csv": curve_pairs,
        "pairwise_curve_distance_summary.csv": curve_summary,
        "pairwise_change_point_agreement_by_source.csv": boundary_pairs,
        "pairwise_change_point_agreement_summary.csv": boundary_summary,
        "pairwise_change_point_matches.csv": event_matches,
        "pairwise_change_point_event_error_summary.csv": event_summary,
        "change_point_gap_diagnostics.csv": cp_diagnostics,
        "change_point_gap_summary_by_method.csv": cp_gap_summary,
        "elapsed_time_residual_autocorrelation_by_source.csv": residual_acf,
        "elapsed_time_residual_autocorrelation_summary.csv": residual_acf_summary,
        "hmm_state_dwell_times_by_run.csv": hmm_dwell,
        "hmm_state_occupancy_by_source.csv": hmm_occupancy,
        "hmm_source_diagnostics.csv": hmm_source_summary,
        "hmm_state_transitions_by_source.csv": hmm_transitions_by_source,
        "hmm_decoded_transition_summary.csv": hmm_transition_summary,
        "method_metric_bootstrap_ci.csv": bootstrap_summary,
        "paired_method_difference_bootstrap_ci.csv": paired_bootstrap_summary,
        "representative_sources.csv": representative_sources,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    _plot_boxplots(
        metrics, "n_segments", "Segments per source",
        "Segmentation complexity by method", output_dir / "segments_per_source.png",
        log_scale=True,
    )
    _plot_boxplots(
        segment_detail, "duration_days", "Segment duration (days)",
        "Segment-duration distributions", output_dir / "segment_durations.png",
        log_scale=True,
    )
    _plot_boxplots(
        segment_detail, "observed_span_days", "First-to-last observed span (days)",
        "Observed support within segments", output_dir / "segment_observed_spans.png",
        log_scale=True,
    )
    _plot_boxplots(
        segment_detail, "max_within_segment_gap_days", "Largest within-segment observation gap (days)",
        "Largest unobserved gap inside each segment", output_dir / "segment_max_internal_gaps.png",
        log_scale=True,
    )
    _plot_boxplots(
        segment_detail, "n_assigned_observations", "Detections per segment",
        "Observations per segment", output_dir / "observations_per_segment.png",
        log_scale=True,
    )
    _plot_boxplots(
        metrics, "standardised_residual_rmse", "Standardised residual RMSE",
        "Residual fit diagnostics by method", output_dir / "standardised_residual_rmse.png",
    )
    _plot_boxplots(
        metrics, "amplitude_retention", "Fitted / observed robust amplitude",
        "Amplitude preservation by method", output_dir / "amplitude_retention.png",
    )
    _plot_heatmap(
        curve_summary, "normalised_curve_rmse",
        "Pairwise fitted-curve distance (median normalised RMSE)",
        output_dir / "pairwise_curve_distance_heatmap.png",
        agreement_scale=False,
    )

    if not boundary_summary.empty:
        for tolerance in args.boundary_tolerances:
            tolerance_summary = boundary_summary[
                np.isclose(boundary_summary["tolerance_days"], tolerance)
            ]
            tolerance_tag = f"{tolerance:g}".replace(".", "p")
            _plot_heatmap(
                tolerance_summary, "median_f1",
                f"Time-based change-point agreement (median F1; ±{tolerance:g} days)",
                output_dir / f"change_point_agreement_heatmap_{tolerance_tag}d.png",
                agreement_scale=True,
            )
            _plot_heatmap(
                tolerance_summary, "gap_aware_median_f1",
                f"Gap-aware change-point agreement (median F1; ±{tolerance:g} days)",
                output_dir / f"change_point_gap_aware_heatmap_{tolerance_tag}d.png",
                agreement_scale=True,
            )

        # Refresh the legacy filename using the tolerance nearest nine days.
        nearest = min(args.boundary_tolerances, key=lambda t: abs(t-9.0))
        tag = f"{nearest:g}".replace(".", "p")
        legacy_source = output_dir / f"change_point_agreement_heatmap_{tag}d.png"
        if legacy_source.exists():
            (output_dir / "change_point_agreement_heatmap.png").write_bytes(legacy_source.read_bytes())

        _plot_tolerance_sensitivity(
            boundary_summary,
            output_dir / "change_point_tolerance_sensitivity.png",
            value="median_f1",
            title="Time-based change-point agreement vs matching tolerance",
        )
        _plot_tolerance_sensitivity(
            boundary_summary,
            output_dir / "change_point_gap_aware_tolerance_sensitivity.png",
            value="gap_aware_median_f1",
            title="Gap-aware change-point agreement vs matching tolerance",
        )

    _plot_change_point_gap_support(
        cp_gap_summary,
        output_dir / "change_point_gap_support.png",
    )
    _plot_event_error_distribution(
        event_matches,
        output_dir / "change_point_event_timing_errors_time_based.png",
        matching_mode="time_tolerance",
    )
    _plot_event_error_distribution(
        event_matches,
        output_dir / "change_point_event_timing_errors_gap_aware.png",
        matching_mode="gap_aware",
    )
    _plot_residual_autocorrelation(
        residual_acf_summary,
        output_dir / "elapsed_time_residual_autocorrelation.png",
    )
    _plot_bootstrap_intervals(
        bootstrap_summary,
        output_dir / "method_metric_bootstrap_confidence_intervals.png",
    )
    _plot_hmm_transition_matrix(
        hmm_transition_summary,
        output_dir / "hmm_decoded_transition_matrix.png",
    )

    _plot_hmm_state_summary(hmm_occupancy, output_dir / "hmm_state_occupancy_and_dwell.png")

    _write_interpretation(
        output_dir,
        method_summary,
        curve_summary,
        boundary_summary,
        event_summary,
        cp_gap_summary,
        residual_acf_summary,
        hmm_source_summary,
        bootstrap_summary,
        representative_sources,
        args.boundary_tolerances,
        args.plot_gap_days,
        args.bootstrap_replicates,
        args.random_seed,
    )

    print(f"\nAnalysis outputs written to: {output_dir}")
    print("No segmentation was rerun. No original raw LCR JSON files were required.")
    if hmm_dwell.empty:
        print("WARNING: no HMM state labels were available; HMM state diagnostics were empty.")
    print("HMM transition probabilities are estimated from saved decoded state runs,")
    print("not read from the fitted model's transition matrix.")


if __name__ == "__main__":
    main()
