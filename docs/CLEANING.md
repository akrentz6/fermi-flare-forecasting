# Data cleaning record

Script: `src/clean_lcr.py`. Input: raw LCR JSON files (3-day bins, photon flux,
fixed spectral index), one per source. 1,525 sources were requested from the LCR
API on 17 November 2026; 4 failed (listed in `failed.txt`), leaving 1,521. The raw
files are never edited; all cleaning is done in code. Cleaned files and
`quality_report.csv` are written to `data/processed/`.

## 1. Look (what is one row?)
One row is one 3-day time bin for one source. Each raw file holds ~2,200 bins
with TS (detection strength), flux, flux error, flux upper limit, photon index,
fit status and background terms. A flux value exists only when TS >= 4; other
bins have only a 95% upper limit.

## 2. Blanks
Most "blanks" are not errors: a bin with no flux is a non-detection (upper limit).
Decision: drop non-detections (TS < 4), as the reference paper does. We keep no
imputed values and do not fill gaps. We record the gap before each kept bin in
`gap_days`. Removed in total: 2,832,340 bins. About 85% of all bins are removed
(92% for the median source).

## 3. Repeated rows
Rows are keyed by time stamp when the JSON is parsed, so each time stamp appears
once per source. A duplicated time stamp would be overwritten, not kept. We did not
run a separate duplicate check.

## 4. Spelling
Source names come from one catalogue (4FGL-DR2) and are consistent. The API needs
spaces in names (4FGL J0001.5+2113); files are saved with underscores.

## 5. Numbers and dates
- Time: NASA's MET (seconds) is converted to MJD (days):
  MJD = 51910 + 7.4287e-4 + MET / 86400.
- All values are stored as numbers. Flux error comes as [lower, upper] edges, so the
  symmetric error is (upper - lower) / 2.

## 6. Impossible or unusual values
| Rule | Rows removed (all sources) | Source of rule |
|---|---|---|
| TS < 4 | 2,832,340 | Reference paper |
| Flux not positive | 0 | Reference paper |
| Flux error not finite | 0 | Reference paper |
| Flux above 1e-4 ph/cm2/s | 1,236 | Reference paper (outlier cap) |
| Fit did not converge | 926 | Our addition |

Open decision: very bright bins may be real flares, which is what we want to
forecast. The cap may be better as a flag than a deletion.

## 7. Normalisation
Flux and error are divided by each source's quiet-level flux F_q: the median of the
lowest 30% of fluxes in the training period (before MJD 59910, to be agreed).
Needs at least 30 training points; 3 sources had too few and get no value.
Limitation: F_q uses detections only, so it is biased upwards for faint sources.

## 8. Saved outputs and checks
- One CSV per source, plus `quality_report.csv` (rows removed per rule, F_q, bin
  spacing). The smallest raw gap between bins was 3.0 days on the 3 test sources
  checked.
- Results across sources: median 185 detections kept (min 6, max 2,178);
  478 sources have >= 300 detections; median fraction of bins kept is 8%.
- Combined file for sharing: `cleaned/cleaned_min300.csv` (478 sources, 341,129
  rows) and `cleaned/selected_sources.csv`, made by `src/combine_selected.py`.
  The checking notebook is `notebooks/cleaning.ipynb`.

## 9. Not done / open
Sources are selected for the combined file: 478 sources with at least 300
detections and a valid F_q. The 300 threshold and the MJD 59910 train/test cut are
not yet agreed with the team. No imputation, no smoothing, no flare labelling.
