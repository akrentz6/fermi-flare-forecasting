# Fermi LCR flare forecasting: project log (from the beginning)
 
## 1. Project direction
- Classifying Fermi objects is already well covered by existing projects and papers.
  What has not been done is analysing how objects' activity changes over time.
- Goal: forecast when a gamma-ray source will become unusually bright (a flare).
  This is time-series forecasting, so no physics knowledge is needed.
- Main dataset: Fermi-LAT Light Curve Repository (LCR). It records how brightness
  changes over time for ~1,500 variable sources, in 3-day bins, from 2008 onwards.
- Possible second dataset: FAVA (all-sky variability analysis), which flags unusually
  bright periods. Not used so far.
## 2. Reading the reference paper (arXiv 2605.07680, Shah & Akbar)
- They forecast flares for one blazar (plus 13 helper sources) from 3-day LCR data.
- They do not test on unseen sources. This is the gap our project targets.
- Their preprocessing, which we copied: keep bins with TS >= 4, positive flux and
  finite error; cap flux at 1e-4; no imputation of missing bins; normalise each source
  by its quiet-level flux (median of the lowest 30% of training-period fluxes).
## 3. Decision: use the API, not manual downloads
- ~1,500 sources makes manual download from each source page impractical.
- NASA's LCR pages document scripted access (pyLCR, an example `download.py`, API
  links) and list no bulk download. We used NASA's own request format.
## 4. Setup
- Python 3.13, VS Code, packages: numpy, pandas, astropy, matplotlib.
- Scripts in `src/`: `make_source_list.py`, `download_lcr.py`, `clean_lcr.py`,
  `probe_api.py`, `summarise_report.py`.
- Raw and cleaned data live in `data/`, which is not committed to GitHub.
## 5. Source list
- `make_source_list.py` reads the 4FGL-DR2 catalogue (`gll_psc_v27.fit`) and keeps
  sources with variability index > 21.67 (NASA's rule for LCR inclusion).
- Result: 1,525 sources.
## 6. Test download and what we learned
- First test downloads returned empty data without errors.
- `probe_api.py` showed: the server only accepts source names **with spaces**
  (`4FGL J0001.5+2113`), not underscores; and cadence `daily` returns the 3-day bins
  (456 bins for the test source; weekly 287; monthly 113).
- Check on the 3 test sources: smallest gap between raw bins = 3.0 days.
- JSON structure: for each time bin there are TS, fit status and background terms. A
  flux is given only for **detections (TS >= 4)**; other bins have only a 95% upper
  limit. `flux_error` is stored as [time, lower edge, upper edge], so the symmetric
  error is (upper - lower) / 2.
## 7. Full download
- 1,525 sources attempted, 1,521 saved.
- 4 failed: 4FGL J1036.3-5833e, J1420.3-6046e, J1855.9+0121e, J1923.2+1408e. Names
  ending in "e" are extended sources, which the LCR does not cover. Expected.
## 8. Cleaning (`clean_lcr.py`), run on 1,521 sources
1. Parse raw JSON into a table (time, flux, flux error, TS, photon index, fit status).
   Convert NASA's MET time to MJD.
2. Drop bins with TS < 4.
3. Drop bins with non-positive flux.
4. Drop bins with non-finite flux error.
5. Drop bins with flux above 1e-4 ph cm^-2 s^-1 (outlier cap from the paper).
6. Drop bins where the fit did not converge (our addition).
7. Do not impute gaps; record the gap before each kept bin (`gap_days`).
8. Normalise flux and error by the quiet-level flux (needs >= 30 training points).
   Training period = before MJD 59910 (the paper's cutoff; to be agreed).
- Each run also writes `quality_report.csv` with the rows removed by each rule.
## 9. Results across all 1,521 sources
- Detections kept per source (n_kept): median 185, mean 326, min 6, max 2,178.
- Sources with n_kept >= 30 / 100 / 300 / 600: 1,519 / 1,239 / 478 / 212.
- Median fraction of bins kept: 8%.
- 3 sources have too few training points for a quiet-level flux (NaN).
- Rows removed in total: TS < 4: 2,832,340; flux above cap: 1,236; non-converged:
  926; non-positive flux: 0; non-finite error: 0.
## 10. Findings and issues
- Most sources are faint: ~92% of their bins are upper limits only. After cleaning,
  their series have large gaps.
- The paper needs >= 30 valid points per 365-day window (~122 bins). A typical source
  (8% kept) would rarely meet this, so faint sources are poor candidates.
- The quiet-level flux uses detections only, so it is biased upwards for faint sources.
- The flux cap removes very bright bins, which may be real flares (what we want to
  forecast). Consider flagging instead of deleting.
## 11. Proposed next steps (to agree with team)
- Select a subset of well-sampled sources, e.g. n_kept >= 300 (478 sources); stricter
  option >= 600 (212 sources). Choose the threshold before looking at model results.
- Agree the train/test cut date (default MJD 59910).
- Decide whether to keep the non-convergence cut and how to treat the flux cap.
- Decide how to label flares (paper uses Bayesian Blocks) and whether to use FAVA.
