# Paper review figures — September 14, 2026 polished revision

Rebuilds 17 main, Extended Data and Supplementary figures from the audited CSV files. It performs no model training or inference. Requires Python 3, NumPy, pandas, Matplotlib and Pillow.

Place the polished review package at compare/paper_review_20260914_polished. From MVSelect-main:

```bash
python3 plot_paper_review_figures.py --data-dir compare/paper_review_20260914_polished/source_data --assets-dir compare/paper_review_20260914_polished/assets --output compare/paper_review_20260914_polished/rebuilt_figures
```

The assets argument is optional when the assets folder is beside source_data, or paper_review_assets is beside this script. All three original PNG renders and object_examples.json must be present. The author supplied side-on, end-on and oblique views of the same vehicle. The JSON records original filenames, camera metadata and SHA-256 hashes. The files are copied unchanged; the same blank top/bottom margins are omitted from all three display viewports. The original aspect ratio and relative scale are preserved, with no synthesis or retouching. Numerical selection statistics come from the audited CSVs.

Figure 1 now uses object examples, explicitly depicts multiview recognition, routes feedback upward, and places the lower-panel legends outside the plotting areas. Figure 2 shows epoch ticks in all panels and shares a figure-level legend. Display text uses American English; source CSV field names are preserved.

Outputs are PNG, SVG, figure_build.json and plotted_curve_values.csv. The review package supplies captions, source workbooks and analysis-unit definitions. This plotting command is optional and is not a new experiment. Do not rerun E03/E12 or mix pending E01 pilots into the historical five-run results.
