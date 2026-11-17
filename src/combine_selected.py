"""Select well-sampled sources and combine their cleaned files into one CSV.

Reads quality_report.csv, keeps sources with at least --min-detections kept
bins (and a valid quiet-level flux F_q), and concatenates their cleaned CSVs
(from the same folder as the report) into a single file.

Usage (from the repo root):
  python src/combine_selected.py data/processed/quality_report.csv \
      --min-detections 300 --out cleaned/cleaned_min300.csv

Also writes selected_sources.csv (the report rows for the chosen sources) next
to the output file. Both files are small enough to commit to GitHub; the script
warns if the combined file is close to GitHub's 100 MB limit.
"""
import argparse
from pathlib import Path

import pandas as pd

KEEP_COLUMNS = [
    "source", "time_MJD", "ts", "flux", "flux_error", "photon_index",
    "gap_days", "F_q", "flux_norm", "flux_error_norm",
]

ap = argparse.ArgumentParser()
ap.add_argument("report")
ap.add_argument("--min-detections", type=int, default=300)
ap.add_argument("--out", default="cleaned/cleaned_min300.csv")
a = ap.parse_args()

report_path = Path(a.report)
rep = pd.read_csv(report_path)
if "error" in rep.columns:
    rep = rep[rep["error"].isna()]
chosen = rep[(rep["n_kept"] >= a.min_detections) & rep["F_q"].notna()]

frames = []
for s in chosen["source"]:
    df = pd.read_csv(report_path.parent / f"{s}.csv")
    frames.append(df[[c for c in KEEP_COLUMNS if c in df.columns]])
combined = pd.concat(frames, ignore_index=True)

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(out, index=False, float_format="%.6g")
chosen.to_csv(out.parent / "selected_sources.csv", index=False)

mb = out.stat().st_size / 1e6
print(f"{len(chosen)} of {len(rep)} sources kept (n_kept >= {a.min_detections})")
print(f"{len(combined)} rows written to {out} ({mb:.1f} MB)")
if mb > 90:
    print("WARNING: close to GitHub's 100 MB limit. Use a higher --min-detections.")
