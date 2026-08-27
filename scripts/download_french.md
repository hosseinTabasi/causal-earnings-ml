# Ken French factors

Intended source: Kenneth R. French data library,
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
Daily FF3 file: F-F_Research_Data_Factors_daily_CSV.zip

This repo will try that URL with a short timeout. If the download fails,
`src/returns/car.py` uses synthetic factors labelled `synthetic_toy`.
Do not scrape CRSP/Compustat. No WRDS credentials.
Place a parsed CSV at `data/french_ff3_daily.csv` if you already have it.
