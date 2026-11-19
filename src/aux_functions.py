import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import matplotlib.dates as mdates

def lightcurve_plot(data):    
    
    src = str(data['source'].iloc[0])
    data = data.copy()
    data["datetime"] = julian_convert(data["time_MJD"])
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.errorbar(data["datetime"], data["flux_norm"], yerr=data["flux_error_norm"], fmt=".", ms=3, lw=0.5)
    
    ax.axhline(1, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("Date") 
    ax.set_ylabel("Flux / Quiescent Flux") 
    ax.set_title(src)

    ax.margins(x=0)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.tick_params(axis='x', rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    
    plt.tight_layout()
    plt.show()

# Convert julian dates for plotting
def julian_convert(dates): 
    # shift and convert julian calendar to normal one
    return pd.to_datetime(dates+ 2400000.5, unit ='D', origin='julian')


# extract a single lightcurve
def lightcurve_extract(source,out_path='source_file/',in_path='cleaned/cleaned_min300.csv'):
    # read the data into a variable
    data = pd.read_csv(in_path)

    # Consider only the needed source
    data = data[data['source']==source]

    # Path object
    out = Path(out_path)
    # create the folders needed to save the file there or skip if already exists
    out.mkdir(parents=True, exist_ok=True)

    # save it 
    data.to_csv(out_path+str(source)+'.csv',index=False)

