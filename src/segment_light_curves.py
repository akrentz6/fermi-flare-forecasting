"""Segment each source in the combined cleaned Fermi-LAT light curves."""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.stats import bayesian_blocks
from matplotlib.figure import Figure

PELT_FACTORS = {"linear": 2, "rbf": 1, "normal": 3}
METHODS = ["bayesian-blocks", *(f"pelt-{cost}" for cost in PELT_FACTORS), "window", "hmm"]
COLUMNS = ["source", "method", "start_MJD", "end_MJD", "n_obs", "mean_flux",
           "flux_std", "fitted_flux", "slope_per_day", "state"]


def segment(method, time, flux, error, args):
    n = len(time)
    if method == "bayesian-blocks":
        edges = bayesian_blocks(time, x=flux.copy(), sigma=error, fitness="measures", p0=args.p0)
        cuts = [0, *np.searchsorted(time, edges[1:-1]), n]
        return [
            (edges[k], edges[k + 1], i, j, np.average(flux[i:j], weights=error[i:j]**-2),
             None, None)
            for k, (i, j) in enumerate(zip(cuts, cuts[1:]))
        ]

    center, scale = flux.mean(), flux.std()
    signal = (flux - center) / (scale or 1)
    penalty = args.penalty_scale * np.log(n)
    days = time - time[0]
    states = state_means = None

    if method.startswith("pelt-"):
        if n < args.min_size:
            return []
        import ruptures as rpt

        cost = method.removeprefix("pelt-")
        data = np.column_stack((signal, np.ones(n), days)) if cost == "linear" else signal
        # Keep covariance regularization and suppress its repeated version notice.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"New behaviour in v1\.1\.5:.*",
                                    category=UserWarning, module=r"ruptures\.costs\.costnormal")
            detector = rpt.Pelt(model=cost, min_size=args.min_size, jump=1)
        ends = detector.fit(data).predict(
            pen=PELT_FACTORS[cost] * penalty
        )
    elif method == "window":
        if n <= args.window_width:
            return []
        import ruptures as rpt

        ends = rpt.Window(width=args.window_width, model="l2", jump=1).fit(signal).predict(
            pen=penalty
        )
    else:
        if n < 3 * args.hmm_states or np.unique(flux).size < args.hmm_states:
            return []
        from hmmlearn.hmm import GaussianHMM

        values = signal[:, None]
        # Give each transition a small pseudocount, including from unused states.
        hmm = GaussianHMM(n_components=args.hmm_states, transmat_prior=1.01,
                          n_iter=200, random_state=0).fit(values)
        order = np.argsort(hmm.means_[:, 0])
        states = np.argsort(order)[hmm.predict(values)] + 1
        state_means = center + scale * hmm.means_[order, 0]
        ends = [*(np.flatnonzero(np.diff(states)) + 1), n]

    edges = np.r_[time[0], (time[:-1] + time[1:]) / 2, time[-1]]
    segments = []
    for i, j in zip([0, *ends[:-1]], ends):
        slope = state = None
        level = flux[i:j].mean()
        if method == "pelt-linear":
            slope = np.linalg.lstsq(
                np.column_stack((np.ones(j - i), days[i:j])), signal[i:j], rcond=None
            )[0][1] * scale
        elif states is not None:
            state = states[i]
            level = state_means[state - 1]
        segments.append((edges[i], edges[j], i, j, level, slope, state))
    return segments


def plot_source(source, time, flux, error, fits, path, dpi):
    fig = Figure(figsize=(10, max(4, 2 * len(fits))), layout="constrained")
    axes = fig.subplots(len(fits), sharex=True, sharey=True, squeeze=False)[:, 0]
    fig.suptitle(source)
    fig.supylabel("Photon flux (ph/cm²/s)")
    axes[-1].set_xlabel("Time (MJD)")
    for ax, (method, segments) in zip(axes, fits.items()):
        ax.set_title(method, loc="left")
        ax.errorbar(time, flux, yerr=error, fmt=".", ms=3, lw=0.7, alpha=0.5)
        if not segments:
            ax.text(0.02, 0.9, "Insufficient data for this method", transform=ax.transAxes)
            continue
        if method == "bayesian-blocks":
            ax.stairs([row[4] for row in segments],
                      [segments[0][0], *(row[1] for row in segments)],
                      baseline=None, color="tab:red")
            continue
        for start, end, i, j, level, slope, _ in segments:
            if slope is None:
                ax.hlines(level, start, end, color="tab:red")
            else:
                bounds = time[[i, j - 1]]
                ax.plot(bounds, level + slope * (bounds - time[i:j].mean()), color="tab:red")
            if i:
                ax.axvline(start, color="tab:red", alpha=0.4, lw=0.7)
    fig.savefig(path, dpi=dpi)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_csv", help="combined cleaned light curves")
    ap.add_argument("--mode", default="bayesian-blocks",
                    help="comma-separated methods (default: bayesian-blocks)")
    ap.add_argument("--out", type=Path, default=Path("cleaned/segmentation"))
    ap.add_argument("--plots", type=int, default=0, help="maximum source PNGs (default: 0)")
    ap.add_argument("--dpi", type=int, default=300, help="PNG resolution (default: 300)")
    ap.add_argument("--limit", type=int, help="fit at most this many sources")
    ap.add_argument("--p0", type=float, default=0.05)
    ap.add_argument("--min-size", type=int, default=10, help="minimum PELT segment detections")
    ap.add_argument("--penalty-scale", type=float, default=1, help="multiply PELT and window penalties")
    ap.add_argument("--window-width", type=int, default=40, help="total detections in each window")
    ap.add_argument("--hmm-states", type=int, default=2, help="number of Gaussian HMM states")
    args = ap.parse_args()
    if not 0 < args.p0 < 1:
        ap.error("--p0 must be between 0 and 1")
    if args.min_size < 3 or args.window_width < 4 or args.window_width % 2 or args.hmm_states < 2:
        ap.error("use --min-size >= 3, an even --window-width >= 4, and --hmm-states >= 2")
    if not np.isfinite(args.penalty_scale) or args.penalty_scale <= 0:
        ap.error("--penalty-scale must be positive and finite")
    if args.plots < 0 or (args.limit is not None and args.limit < 0):
        ap.error("--plots and --limit must be nonnegative")
    if args.dpi < 1:
        ap.error("--dpi must be positive")

    mode_groups = {"all": METHODS, "pelt": [f"pelt-{cost}" for cost in PELT_FACTORS]}
    methods = {}
    for mode in args.mode.split(","):
        mode = mode.strip()
        if mode not in METHODS and mode not in mode_groups:
            ap.error(f"unknown mode {mode!r}. Choose from {', '.join([*METHODS, *mode_groups])}")
        methods.update(dict.fromkeys(mode_groups.get(mode, [mode])))

    columns = ["source", "time_MJD", "ts", "flux", "flux_error"]
    df = pd.read_csv(args.input_csv, usecols=columns)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    detections = df[(df["ts"] >= 4) & (df["flux"] > 0) & (df["flux_error"] > 0)]

    args.out.mkdir(parents=True, exist_ok=True)
    rows = {method: [] for method in methods}
    plot_tag = "_".join(methods)
    fitted = 0
    for source, observations in detections.groupby("source", sort=False):
        if args.limit is not None and fitted >= args.limit:
            break
        if len(observations) < 2:
            continue
        time, flux, error = observations.sort_values("time_MJD")[
            ["time_MJD", "flux", "flux_error"]
        ].to_numpy(dtype=float).T
        fits = {}
        for method in methods:
            segments = segment(method, time, flux, error, args)
            fits[method] = segments
            for start, end, i, j, level, slope, state in segments:
                mean = level if method == "bayesian-blocks" else flux[i:j].mean()
                rows[method].append((source, method, start, end, j - i, mean, flux[i:j].std(),
                                     level, slope, state))
        if any(fits.values()):
            if fitted < args.plots:
                path = args.out / f"{source}_{plot_tag}.png"
                plot_source(source, time, flux, error, fits, path, args.dpi)
            fitted += 1

    for method, method_rows in rows.items():
        result = pd.DataFrame(method_rows, columns=COLUMNS)
        result["state"] = result["state"].astype("Int64")
        result.to_csv(args.out / f"segments_{method}.csv", index=False)
    print(f"Fitted {fitted} sources. Wrote {len(rows)} CSVs and "
          f"{min(fitted, args.plots)} PNGs to {args.out}.")


if __name__ == "__main__":
    main()
