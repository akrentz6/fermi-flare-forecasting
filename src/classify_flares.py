"""Apply a shared, exploratory flare definition to fitted and observed flux."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from segment_light_curves import METHODS, read_observations

BIN_DAYS = 3.0
TIME_TOL = 1e-8
REFERENCE = "reference"
EVENT_COLUMNS = ["source", "period", "start_MJD", "end_MJD", "duration_days", "threshold"]
SENSITIVITY_COLUMNS = [
    "source", "method", "period", "scenario", "quantile", "quiet_fraction",
    "sigma_multiplier", "threshold", "event_count", "count_change", "onset_MJD",
]


def read_segments(directory, methods, train_end=None):
    by_method = {}
    for method in methods:
        path = directory / f"segments_{method}.csv"
        frame = pd.read_csv(path, dtype={"train_end_MJD": str})
        if train_end is not None:
            if ("train_end_MJD" not in frame or
                    not frame["train_end_MJD"].map(float).eq(train_end).all()):
                raise ValueError(f"{path} must be fitted with --train-end-mjd {train_end}")
        elif "train_end_MJD" in frame:
            raise ValueError(f"{path} must contain full light-curve fits")
        by_method[method] = dict(tuple(frame.groupby("source", sort=False)))
    return by_method


def fitted_lines(rows, time, source, method, period):
    """Recover line centres from segment membership in the fitted observations."""
    rows = rows.sort_values("start_MJD", kind="stable")
    start, end, level = rows[["start_MJD", "end_MJD", "fitted_flux"]].to_numpy().T
    counts = rows["n_obs"].to_numpy(dtype=int)
    stops = counts.cumsum()
    if (not len(time) or (counts <= 0).any() or stops[-1] != len(time) or
            abs(start[0] - time[0]) > TIME_TOL or abs(end[-1] - time[-1]) > TIME_TOL):
        raise ValueError(f"{source}: {period} {method} segments do not match the input observations")
    cuts = stops[:-1]
    edges = (time[cuts - 1] + time[cuts]) / 2
    if not (np.allclose(start[1:], edges, rtol=0, atol=TIME_TOL) and
            np.allclose(end[:-1], edges, rtol=0, atol=TIME_TOL)):
        raise ValueError(f"{source}: {period} {method} boundaries do not match the input observations")
    if method == "pelt-linear":
        slope = rows["slope_per_day"].to_numpy()
        centre = np.array([chunk.mean() for chunk in np.split(time, cuts)])
        level = level + slope * (start - centre)
    else:
        slope = np.zeros(len(rows))
    return np.column_stack((start, end, level, slope))


def threshold(sorted_flux, quantile, quiet_fraction, sigma_multiplier):
    n_quiet = max(1, int(np.floor(quiet_fraction * len(sorted_flux))))
    quiet = sorted_flux[:n_quiet]
    return max(np.quantile(sorted_flux, quantile), quiet.mean() + sigma_multiplier * quiet.std())


def coverage(time, lower, upper, gap_days):
    """Return coverage blocks bounded by the first and last observed bin edges."""
    if not len(time):
        return []
    breaks = np.flatnonzero(np.diff(time) > gap_days + TIME_TOL) + 1
    start = np.maximum(time[np.r_[0, breaks]] - BIN_DAYS / 2, lower)
    end = np.minimum(time[np.r_[breaks - 1, len(time) - 1]] + BIN_DAYS / 2, upper)
    return list(zip(start, end))


def clip_to_block(intervals, start, end):
    left = np.searchsorted(intervals[:, 1], start, side="right")
    right = np.searchsorted(intervals[:, 0], end, side="left")
    clipped = intervals[left:right].copy()
    clipped[:, 0] = np.maximum(clipped[:, 0], start)
    clipped[:, 1] = np.minimum(clipped[:, 1], end)
    return clipped


def above_threshold(lines, level):
    start, end, initial, slope = lines.T
    final = initial + slope * (end - start)
    keep = (initial > level) | (final > level)
    start, end, initial, slope = start[keep], end[keep], initial[keep], slope[keep]
    rising = initial <= level
    falling = final[keep] <= level
    end[falling] = start[falling] + (level - initial[falling]) / slope[falling]
    start[rising] += (level - initial[rising]) / slope[rising]
    return np.column_stack((start, end))


def join_intervals(intervals, max_gap):
    joined = []
    for start, end in intervals:
        if end <= start:
            continue
        if joined and start <= joined[-1][1] + max_gap + TIME_TOL:
            joined[-1] = (joined[-1][0], max(joined[-1][1], end))
        else:
            joined.append((start, end))
    return joined


def flare_events(lines, level, blocks, min_days, merge_days):
    active = above_threshold(lines, level)
    events = []
    for start, end in blocks:
        runs = join_intervals(clip_to_block(active, start, end), 0)
        qualifying = [(lo, hi) for lo, hi in runs if hi - lo >= min_days - TIME_TOL]
        events.extend(join_intervals(qualifying, merge_days))
    return events


def plot_comparison(source, time, flux, error, level, cutoff, methods, panels, path, dpi):
    fig = Figure(figsize=(11, 2 * len(methods)), layout="constrained")
    axes = fig.subplots(len(methods), sharex=True, sharey=True)
    fig.suptitle(source)
    fig.supylabel("Photon flux (ph/cm²/s)")
    axes[-1].set_xlabel("Time (MJD)")
    for ax, method in zip(axes, methods):
        ax.set_title("Observed-bin reference" if method == REFERENCE else method, loc="left")
        ax.errorbar(time, flux, yerr=np.where(error > 0, error, np.nan),
                    fmt=".", ms=2, lw=0.6, alpha=0.5)
        ax.axhline(level, color="tab:orange", ls="--", lw=1, label="Training threshold")
        if time[0] < cutoff <= time[-1]:
            ax.axvline(cutoff, color="0.35", ls=":", lw=1, label="Training cutoff")
        for lines, blocks, events in panels[method]:
            if method != REFERENCE:
                pieces = []
                for start, end in blocks:
                    lo, hi, value, slope = lines.T
                    keep = (hi > start) & (lo < end)
                    x = np.column_stack((np.maximum(lo[keep], start), np.minimum(hi[keep], end)))
                    y = value[keep, None] + slope[keep, None] * (x - lo[keep, None])
                    pieces.extend(np.stack((x, y), axis=-1))
                ax.add_collection(LineCollection(pieces, colors="tab:red", linewidths=1))
                ax.autoscale_view()
            for start, end in events:
                ax.axvspan(start, end, color="tab:red", alpha=0.18, lw=0)
        if not panels[method]:
            ax.text(0.02, 0.82, "No fit available", transform=ax.transAxes)
    axes[0].legend(loc="upper right")
    fig.savefig(path, dpi=dpi)


def classify(args, methods, full, training):
    observations = read_observations(args.input_csv).sort_values(["source", "time_MJD"])
    names = [*methods, REFERENCE]
    event_rows = {name: [] for name in names}
    sensitivity = []
    observations["period"] = np.where(observations["time_MJD"] < args.train_end_mjd,
                                      "training", "held_out")
    observations["threshold"] = np.nan
    for name in names:
        observations[name] = pd.Series(pd.NA, index=observations.index, dtype="Int8")

    scenarios = [
        ("configured", args.quantile, args.quiet_fraction, args.sigma_multiplier),
        ("lower", max(0, args.quantile - .05), max(args.quiet_fraction / 2, args.quiet_fraction - .05),
         max(0, args.sigma_multiplier - .5)),
        ("higher", min(1, args.quantile + .05), min(1, args.quiet_fraction + .05),
         args.sigma_multiplier + .5),
    ]
    plotted = 0
    for source, obs in observations.groupby("source", sort=False):
        time, flux, error = obs[["time_MJD", "flux", "flux_error"]].to_numpy().T
        fit_times = time[error > 0]
        split = np.searchsorted(time, args.train_end_mjd)
        if split == 0:
            continue
        train_flux = np.sort(flux[:split])
        levels = [threshold(train_flux, *scenario[1:]) for scenario in scenarios]
        observations.loc[obs.index, "threshold"] = levels[0]
        panels = {name: [] for name in names}
        for period, subset, lower, upper, fits, fit_time in (
            ("training", slice(None, split), -np.inf, args.train_end_mjd,
             training, fit_times[fit_times < args.train_end_mjd]),
            ("held_out", slice(split, None), args.train_end_mjd, np.inf, full, fit_times),
        ):
            times, values = time[subset], flux[subset]
            if not len(times):
                continue
            lines_by_method = {
                REFERENCE: np.column_stack((times - BIN_DAYS / 2, times + BIN_DAYS / 2,
                                            values, np.zeros(len(times)))),
            }
            for method in methods:
                rows = fits[method].get(source)
                if rows is not None:
                    lines_by_method[method] = fitted_lines(rows, fit_time, source, method, period)
            observed_blocks = coverage(times, lower, upper, args.gap_days)
            fitted_blocks = coverage(fit_time[(fit_time >= lower) & (fit_time < upper)],
                                     lower, upper, args.gap_days)
            for name, lines in lines_by_method.items():
                blocks = observed_blocks if name == REFERENCE else fitted_blocks
                if not blocks:
                    continue
                results = [flare_events(lines, level, blocks, args.min_days, args.merge_days)
                           for level in levels]
                events = results[0]
                covered = np.zeros(len(times), dtype=bool)
                for start, end in blocks:
                    covered |= (times >= start) & (times <= end)
                labels = np.zeros(len(times), dtype=np.int8)
                for start, end in events:
                    first = np.searchsorted(times, start, side="left")
                    last = np.searchsorted(times, end, side="right")
                    labels[first:last] = 1
                    event_rows[name].append((source, period, start, end, end - start, levels[0]))
                observations.loc[obs.index[subset][covered], name] = labels[covered]
                for scenario, level, alternatives in zip(scenarios, levels, results):
                    onsets = "|".join(f"{start:.6f}" for start, _ in alternatives)
                    sensitivity.append((source, name, period, *scenario, level, len(alternatives),
                                        len(alternatives) - len(events), onsets))
                panels[name].append((lines, blocks, events))
        if plotted < args.plots:
            plot_comparison(source, time, flux, error, levels[0], args.train_end_mjd, names,
                            panels, args.out / "plots" / f"{source}.png", args.dpi)
            plotted += 1

    sensitivity = pd.DataFrame(sensitivity, columns=SENSITIVITY_COLUMNS)
    label_columns = ["source", "time_MJD", "period", "flux", "threshold"]
    for directory, group in [(args.out, methods), (args.out / REFERENCE, [REFERENCE])]:
        for name in group:
            pd.DataFrame(event_rows[name], columns=EVENT_COLUMNS).to_csv(
                directory / f"events_{name}.csv", index=False
            )
        observations[[*label_columns, *group]].to_csv(directory / "observation_labels.csv", index=False)
        sensitivity[sensitivity["method"].isin(group)].to_csv(directory / "sensitivity.csv", index=False)
    print(f"Wrote flare classifications for {len(methods)} methods, a separate observed-bin "
          f"reference, and {plotted} comparison PNGs to {args.out}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", nargs="?", default="cleaned/cleaned_min300.csv")
    parser.add_argument("--segments", type=Path, default=Path("cleaned/segmentation"))
    parser.add_argument("--training-segments", type=Path,
                        help="training-only segment directory (default: training under --segments)")
    parser.add_argument("--out", type=Path, default=Path("cleaned/flares"))
    parser.add_argument("--train-end-mjd", type=float, default=59910.0)
    parser.add_argument("--quantile", type=float, default=.70)
    parser.add_argument("--quiet-fraction", type=float, default=.40)
    parser.add_argument("--sigma-multiplier", type=float, default=3.0)
    parser.add_argument("--min-days", type=float, default=9.0)
    parser.add_argument("--merge-days", type=float, default=9.0)
    parser.add_argument("--gap-days", type=float, default=9.0,
                        help="maximum spacing between observed bin centres in a coverage block")
    parser.add_argument("--plots", type=int, default=10, help="maximum source comparison PNGs")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    if not np.isfinite([args.train_end_mjd, args.sigma_multiplier, args.min_days,
                        args.merge_days, args.gap_days]).all():
        parser.error("the cutoff, sigma multiplier, durations and gap limit must be finite")
    if not 0 <= args.quantile <= 1 or not 0 < args.quiet_fraction <= 1:
        parser.error("--quantile must be in [0, 1] and --quiet-fraction in (0, 1]")
    if min(args.sigma_multiplier, args.merge_days, args.plots) < 0:
        parser.error("--sigma-multiplier, --merge-days and --plots must be nonnegative")
    if min(args.min_days, args.gap_days, args.dpi) <= 0:
        parser.error("--min-days, --gap-days and --dpi must be positive")
    methods = [name for name in METHODS if (args.segments / f"segments_{name}.csv").is_file()]
    if not methods:
        parser.error(f"no segment CSVs found in {args.segments}")
    try:
        full = read_segments(args.segments, methods)
        training = read_segments(args.training_segments or args.segments / "training",
                                 methods, args.train_end_mjd)
        (args.out / REFERENCE).mkdir(parents=True, exist_ok=True)
        if args.plots:
            (args.out / "plots").mkdir(exist_ok=True)
        classify(args, methods, full, training)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
