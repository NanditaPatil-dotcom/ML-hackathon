# Report

`PathSnap_Technical_Report.pdf` is the consolidated technical report. It is generated from the saved
results, so every number in it can be traced to a file in `results/`:

```bash
python report/build_report.py   # writes report/PathSnap_Technical_Report.html
```

Then print the HTML to PDF (open it in a browser and use "Save as PDF").

Before the final build, put these PNGs in `report/screenshots/` (they are embedded automatically):
- a Folium map screenshot (filename containing `map` or `folium`)
- the Streamlit UI showing a predicted route
- the API terminal on Laptop 1 showing the request logs

The Bayesian section reads `results/bayesian/summary.json`.
