"""Find which source-name format and cadence value the LCR API accepts.

Usage:
  python src/probe_api.py
  python src/probe_api.py 4FGL_J0001.5+2113

Prints how many time bins each combination returns. 0 = empty reply.
Weekly and monthly are controls: if they work but 'daily' doesn't, the
3-day cadence has a different name and we need to find it.
"""
import sys

from download_lcr import fetch

source = sys.argv[1] if len(sys.argv) > 1 else "4FGL_J0001.5+2113"
names = [source, source.replace("_", " ")]
cadences = ["daily", "weekly", "monthly", "threedays", "3days"]

for name in names:
    for cadence in cadences:
        try:
            d = fetch(name, cadence, "photon", "fixed", 4)
            n = len(d.get("flux", [])) if isinstance(d, dict) else "not a dict"
        except Exception as e:
            n = f"error: {e}"
        print(f"name={name!r:24} cadence={cadence:10} -> flux bins: {n}")
