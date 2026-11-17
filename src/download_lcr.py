"""Download raw LCR light curves (JSON) for a list of sources.

The request URL is taken from NASA's own example script
(https://fermi.gsfc.nasa.gov/ssc/data/access/lat/lcr/download.py).

Usage:
  python src/download_lcr.py data/sources.txt --out data/raw --limit 10

Cadence: the pyLCR README lists 'daily', 'weekly' or 'monthly'. The LCR's
shortest bins are 3 days, so 'daily' is assumed to be the 3-day light curve.
clean_lcr.py reports the median gap between bins per source; it should be
~3 days. If it isn't, change --cadence.
"""
import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = (
    "https://fermi.gsfc.nasa.gov/ssc/data/access/lat/LightCurveRepository/"
    "queryDB.php?typeOfRequest=lightCurveData"
    "&source_name={source}&cadence={cadence}&flux_type={flux_type}"
    "&index_type={index_type}&ts_min={ts_min}&magicWord=130427A"
)


def fetch(source, cadence, flux_type, index_type, ts_min):
    url = BASE.format(
        source=urllib.parse.quote(source),
        cadence=cadence,
        flux_type=flux_type,
        index_type=index_type,
        ts_min=ts_min,
    )
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode())


def has_data(d):
    """True if the reply contains at least one non-empty list of values."""
    return isinstance(d, dict) and any(isinstance(v, list) and v for v in d.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources_file")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--cadence", default="daily", help="'daily' = 3-day bins (see docstring)")
    ap.add_argument("--flux-type", default="photon")
    ap.add_argument("--index-type", default="fixed")
    ap.add_argument("--ts-min", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="only first N sources (test batch)")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between requests")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sources = [s.strip() for s in Path(a.sources_file).read_text().splitlines() if s.strip()]
    if a.limit:
        sources = sources[: a.limit]

    failed = []
    for i, s in enumerate(sources, 1):
        dest = out / f"{s}_{a.cadence}_{a.flux_type}_{a.index_type}_ts{a.ts_min}.json"
        if dest.exists():
            continue  # cached, so reruns only fetch what's missing
        try:
            data = None
            # pyLCR uses spaces in names ('4FGL J0001.2-0747'), the web pages
            # use underscores; try both. The server can return valid JSON
            # with empty lists, so check there is actual data.
            for name in (s, s.replace("_", " ")):
                try:
                    d = fetch(name, a.cadence, a.flux_type, a.index_type, a.ts_min)
                except Exception:
                    continue
                if has_data(d):
                    data = d
                    break
            if data is None:
                raise ValueError("no data returned; check --cadence (run src/probe_api.py)")
            dest.write_text(json.dumps(data))
            print(f"[{i}/{len(sources)}] saved {s}")
        except Exception as e:  # keep going; log the failure
            print(f"[{i}/{len(sources)}] FAILED {s}: {e}")
            failed.append(f"{s}\t{e}")
        time.sleep(a.pause)

    if failed:
        (out / "failed.txt").write_text("\n".join(failed) + "\n")
        print(f"{len(failed)} failures logged to {out / 'failed.txt'}")


if __name__ == "__main__":
    main()