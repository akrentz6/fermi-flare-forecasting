# Forecasting gamma-ray flares in Fermi-LAT sources

Project for UCL Data Science Society Project Labs (2026/27).

## Goal
Predict when a gamma-ray source will become unusually bright (a flare), using
time-series forecasting on Fermi-LAT light curves, and test whether a model works
on sources it has not seen. This extends the approach of arXiv 2605.07680
(Shah & Akbar), which tested one source only.

## Data
Fermi-LAT Light Curve Repository (LCR), NASA: flux in 3-day bins since 2008 for
~1,500 variable sources. Downloaded through the LCR backend interface, using the
request format from NASA's own example script.
https://fermi.gsfc.nasa.gov/ssc/data/access/lat/lcr/about.html

Data are not stored in this repo. Download them with the scripts, or ask the team
for the shared snapshot (downloaded 17 November 2026).

## Status
- Done: source list, download of 1,521 sources, cleaning and normalisation.
- Available: segmentation and exploratory flare labelling.
- Not started: features, forecasting model, evaluation.

## How to run
Python 3.13 with the packages in `requirements.txt`. Run from the repo root.

1. Install packages: `python -m pip install -r requirements.txt`
2. Source list (already provided as `sources.txt`; to rebuild, download
   `gll_psc_v27.fit` from the 4FGL-DR2 catalogue page):
   `python src\make_source_list.py gll_psc_v27.fit sources.txt`
3. Download raw light curves into `data\raw` (about 1-2 hours; safe to stop and
   rerun): `python src\download_lcr.py sources.txt --out data\raw`
   Use `--limit 3` for a quick test.
4. Clean and normalise: `python src\clean_lcr.py data\raw data\processed`
5. Summarise across sources:
   `python src\summarise_report.py data\processed\quality_report.csv`
6. Segment the combined cleaned light curves:
   `python src/segment_light_curves.py cleaned/cleaned_min300.csv --mode all --plots 10`
7. Fit training-only segments and classify flares:

   ```sh
   python src/segment_light_curves.py cleaned/cleaned_min300.csv --mode all --train-end-mjd 59910
   python src/classify_flares.py --train-end-mjd 59910
   ```

   The classifier uses the same raw-flux threshold for all six methods and the
   observed-bin reference. It estimates the threshold from observations before
   MJD 59910 by taking the larger of the 70th percentile and the mean plus three
   population standard deviations of the lowest 40% of training fluxes (rounded
   down to a whole number of observations, with at least one).

   Join continuous above-threshold intervals first, keep runs of at least nine
   days, then merge qualifying events separated by at most nine days. Spacing
   greater than nine days between observation centres breaks coverage and
   prevents merging across that gap. Events are identified separately on each
   side of the cutoff. Training labels use training-only fits. Held-out labels
   use the full-series fits for retrospective evaluation.
   Use `--help` to change the cutoff, threshold parameters,
   timing rules, and plot count.

   Under `cleaned/flares/`, the classifier saves an event CSV per method,
   observation labels, a sensitivity CSV, and comparison plots. Observation
   labels mark membership in the final merged event, with blanks where no
   classification is available. The `reference/` directory contains separate
   outputs for the original 3-day observation bins. This baseline is for
   comparison only. The sensitivity CSV reports flare counts, changes against
   the configured definition, and onset MJDs. The two alternatives vary the
   quantile and quiet fraction by five percentage points and the sigma
   multiplier by 0.5 in each direction, within their valid ranges.

   The threshold is an exploratory definition of a flare. Fix the definition
   independently of forecasting performance and interpret it descriptively.

## Scripts
| Script | Purpose |
|---|---|
| `src/make_source_list.py` | Lists sources with variability index > 21.67 (LCR sample) |
| `src/download_lcr.py` | Downloads raw light curves as JSON |
| `src/clean_lcr.py` | Quality cuts, gap recording, normalisation, quality report |
| `src/summarise_report.py` | Totals across sources from the quality report |
| `src/segment_light_curves.py` | Per-source segmentation |
| `src/classify_flares.py` | Shared flare labels, observed-bin reference, and threshold sensitivity |
| `src/probe_api.py` | Diagnostic: tests which name format and cadence the API accepts |

## Things to know
- The API accepts source names with spaces only. Cadence `daily` returns 3-day bins.
- 4 catalogue sources (names ending in "e", extended sources) have no light curve.
- Most sources are faint: only ~8% of their bins are detections after cleaning.

## Documentation
- `docs/PROJECT_LOG.md`: full project history and open decisions
- `docs/CLEANING.md`: every cleaning decision with row counts

## AI assistance
AI tools (Claude by Anthropic) were used to clarify technical concepts, understand errors, and review code. All scripts, analytical decisions, testing, and final results were independently developed and verified by the team.

## Licence
MIT (see `LICENSE`).
