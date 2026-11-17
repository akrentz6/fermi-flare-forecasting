import sys

import pandas as pd

df = pd.read_csv(sys.argv[1])

if "error" in df.columns:
    bad = df[df["error"].notna()]
    print(f"Sources that failed to parse: {len(bad)}")
    print(bad[["source", "error"]].head(10).to_string(index=False))
    df = df[df["error"].isna()]

df = df.copy()
df["frac_kept"] = df["n_kept"] / df["n_raw"]

print(f"\nSources cleaned: {len(df)}")
print("\nBin width check (raw_min_gap_days), should all be 3.0:")
print(df["raw_min_gap_days"].round(2).value_counts().head())

print("\nDetections kept per source (n_kept):")
print(df["n_kept"].describe().round(1).to_string())

print("\nSources by detections kept:")
for t in (30, 100, 300, 600):
    print(f"  n_kept >= {t:>3}: {(df['n_kept'] >= t).sum()}")

print(f"\nMedian fraction of bins kept: {df['frac_kept'].median():.2f}")
print(f"Sources with too few training points for F_q (NaN): {df['F_q'].isna().sum()}")

print("\nRows removed by each rule (total across sources):")
rules = ["ts_lt_4_or_missing", "flux_not_positive", "flux_error_not_finite",
         "flux_above_cap", "fit_not_converged"]
print(df[rules].sum().to_string())