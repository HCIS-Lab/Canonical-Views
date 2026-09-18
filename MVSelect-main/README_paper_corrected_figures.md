# Rebuild the September 15 corrected-cohort figures

The accompanying plot_paper_corrected_figures.py generates 18 PNG/SVG figures from the supplied review source_data directory and original object-image assets. It is a CPU plotting utility; it does not load models, train, modify raw results or contact a server. Historical and corrected cohorts remain explicitly separated. The older plot_paper_review_figures.py still reproduces the historical review package.

Requirements: Python 3, NumPy, pandas, Matplotlib and Pillow. From MVSelect-main, with the review archive extracted to /path/to/review:

```bash
python3 plot_paper_corrected_figures.py --data-dir /path/to/review/source_data --assets-dir /path/to/review/assets --output /path/to/rebuilt_figures
```

Use the supplied CSV filenames together; individual raw metadata files are not required. Plotted curves and their model SD are exported as plotted_curve_values.csv. No smoothing or run exclusion occurs. New Extended Data Fig. 8 records the pre-freeze mismatch. Source code and data definitions describe sample sizes, scopes and uncertainty.

This is a reproduction instruction, not a new author action. Figures have already been rebuilt locally for the delivered manuscript. No remote transfer is assumed.
