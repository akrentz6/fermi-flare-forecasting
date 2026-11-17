"""Clean and normalise raw LCR light curves.

Steps (following arXiv 2605.07680, Shah & Akbar):
  1. parse raw JSON -> tidy table
  2. quality cuts: TS >= 4, flux > 0, finite flux error, flux <= 1e-4,
     (extra) MINUIT convergence == 0
  3. keep gaps visible (gap_days column) instead of imputing
  4. per-source normalisation by quiescent flux F_q = median of the lowest
     30% of TRAINING-period fluxes (only data before --train-end-mjd)

Usage:
  python src/clean_lcr.py data/raw data/processed --train-end-mjd 59910
  python src/clean_lcr.py data/raw --inspect     # print JSON structure

Parsing follows NASA's download.py: each field is a list of
[time_MET, value] pairs (3-element entries use (v[2]-v[1])/2) and
MJD = 51910 + 7.428703703703703e-4 + MET/86400.
Not yet tested on real LCR files: run --inspect on one first.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FLUX_CAP = 1e-4        # ph cm^-2 s^-1, outlier cap used in the paper
TS_MIN = 4.0
QUIESCENT_FRAC = 0.30  # lowest 30% of training fluxes
MIN_TRAIN_POINTS = 30


def met_to_mjd(met):
    return 51910 + 7.428703703703703e-4 + np.asarray(met, dtype=float) / 86400.0

# Convert julian dates for plotting
def julian_convert(dates): 
    return pd.to_datetime(dates+ 2400000.5, unit ='D', origin='julian').dt.strftime('%Y-%m-%d')



def parse_raw(path):
    raw = json.loads(Path(path).read_text())
    rows = {}
    for key, entries in raw.items():
        key = "fit_convergence" if key == "fit_convergance" else key
        if not isinstance(entries, list):
            continue
        for e in entries:
            if isinstance(e, list) and len(e) >= 2:
                met = e[0]
                val = e[1] if len(e) == 2 else (e[2] - e[1]) / 2
                rows.setdefault(met, {})[key] = val
    if not rows:
        raise ValueError(f"{path}: no [time, value] entries found; run --inspect")
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "time_MET"
    df = df.reset_index().sort_values("time_MET").reset_index(drop=True)
    df["time_MJD"] = met_to_mjd(df["time_MET"])
    return df


def quality_cuts(df, keep_nonconverged=False):
    """Return (clean_df, dict of rows dropped per reason)."""
    n0 = len(df)
    ok = pd.Series(True, index=df.index)
    reasons = {}

    def apply(name, mask):
        nonlocal ok
        reasons[name] = int((ok & ~mask).sum())
        ok &= mask

    apply("ts_lt_4_or_missing", df["ts"].ge(TS_MIN))
    apply("flux_not_positive", df["flux"].gt(0))
    apply("flux_error_not_finite", np.isfinite(df["flux_error"]))
    apply("flux_above_cap", df["flux"].le(FLUX_CAP))
    if not keep_nonconverged and "fit_convergence" in df:
        apply("fit_not_converged", df["fit_convergence"].fillna(0).eq(0))
    reasons["n_raw"] = n0
    reasons["n_kept"] = int(ok.sum())
    return df[ok].copy().reset_index(drop=True), reasons


def quiescent_flux(df, train_end_mjd):
    train = df.loc[df["time_MJD"] < train_end_mjd, "flux"].sort_values()
    if len(train) < MIN_TRAIN_POINTS:
        return np.nan
    n_low = max(1, int(np.floor(QUIESCENT_FRAC * len(train))))
    return float(train.iloc[:n_low].median())


def process_source(path, train_end_mjd, keep_nonconverged=False):
    # filename: <source>_<cadence>_<flux_type>_<index_type>_ts<N>.json
    source = Path(path).stem.rsplit("_", 4)[0]
    df = parse_raw(path)
    # bin width check on the RAW bins (before cuts): smallest spacing between
    # time stamps should be ~3 days for 3-day bins
    raw_min_gap = float(df["time_MJD"].diff().min())
    clean, reasons = quality_cuts(df, keep_nonconverged)
    clean.insert(0, "source", source)
    clean["gap_days"] = clean["time_MJD"].diff()
    fq = quiescent_flux(clean, train_end_mjd)
    clean["F_q"] = fq
    clean["flux_norm"] = clean["flux"] / fq
    clean["flux_error_norm"] = clean["flux_error"] / fq
    # sanity check: raw_min_gap_days should be ~3.0 for 3-day bins
    # (median_gap_days can be larger because cuts leave gaps)
    reasons.update(source=source, F_q=fq, raw_min_gap_days=raw_min_gap,
                   median_gap_days=float(clean["gap_days"].median()))
    return clean, reasons


def inspect(path):
    raw = json.loads(Path(path).read_text())
    print(f"{path}: top-level type {type(raw).__name__}")
    for k, v in (raw.items() if isinstance(raw, dict) else []):
        sample = v[:2] if isinstance(v, list) else v
        n = len(v) if isinstance(v, list) else "-"
        print(f"  {k}: n={n}, sample={sample}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    ap.add_argument("out_dir", nargs="?", default="data/processed")
    ap.add_argument("--train-end-mjd", type=float, default=59910.0,
                    help="agree this cut with your teammate before normalising")
    ap.add_argument("--keep-nonconverged", action="store_true")
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()

    files = sorted(Path(a.raw_dir).glob("*.json"))
    if a.inspect:
        inspect(files[0])
        return

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for f in files:
        try:
            clean, reasons = process_source(f, a.train_end_mjd, a.keep_nonconverged)
        except Exception as e:
            report.append({"source": f.name, "error": str(e)})
            print(f"SKIP {f.name}: {e}")
            continue
        clean.to_csv(out / f"{reasons['source']}.csv", index=False)
        report.append(reasons)
    pd.DataFrame(report).to_csv(out / "quality_report.csv", index=False)
    print(f"{len(report)} sources processed; report in {out / 'quality_report.csv'}")


if __name__ == "__main__":
    main()