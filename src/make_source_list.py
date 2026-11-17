"""Build the list of LCR sources from the 4FGL-DR2 catalogue.

The LCR covers every 4FGL-DR2 source with variability index > 21.67
(per the LCR FAQ). Download the catalogue file once:
  https://fermi.gsfc.nasa.gov/ssc/data/access/lat/10yr_catalog/gll_psc_v27.fit

Usage:
  python src/make_source_list.py gll_psc_v27.fit data/sources.txt

Some catalogue sources (e.g. extended ones) have no LCR light curve; the
download script logs those as failures and carries on.
"""
import sys

from astropy.table import Table

VAR_INDEX_MIN = 21.67


def main(fits_path, out_path):
    t = Table.read(fits_path, hdu=1)
    names = [str(n).strip() for n in t["Source_Name"]]
    var = t["Variability_Index"]
    keep = [n.replace(" ", "_") for n, v in zip(names, var) if v > VAR_INDEX_MIN]
    with open(out_path, "w") as f:
        f.write("\n".join(keep) + "\n")
    print(f"{len(keep)} variable sources written to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])