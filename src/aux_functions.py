import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def lightcurve_plot(sel,df):    
    src = sel.sort_values("n_kept").iloc[len(sel) // 2]["source"]
    d = df[df["source"] == src]
    fig, ax = plt.subplots(figsize=(10, 4))
    d["time_MJD"] = julian_convert(d["time_MJD"])
    ax.errorbar(d["time_MJD"], d["flux_norm"], yerr=d["flux_error_norm"], fmt=".", ms=3, lw=0.5)
    ax.axhline(1, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("MJD"); ax.set_ylabel("flux / quiescent flux"); ax.set_title(src)
    plt.show()

# Convert julian dates for plotting
def julian_convert(dates): 
    return pd.to_datetime(dates+ 2400000.5, unit ='D', origin='julian').dt.strftime('%Y-%m-%d')


# extract a single lightcurve
def lightcurve_extract(source,out_path='source_file/',in_path='cleaned/cleaned_min300.csv'):
    data = pd.read_csv(in_path)
    data = data[data['source']==source]

    out = Path(out_path)
    out.mkdir(parents=True, exist_ok=True)

    return data.to_csv(out_path+str(source)+'.csv',index=False)

lightcurve_extract('4FGL_J0001.5+2113')