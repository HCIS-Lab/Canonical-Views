# E18 reporting update — September 16, 2026

No new training is required. The author has returned all 15 jobs, and the manuscript package already includes the completed audit and plots. These commands reproduce the reporting from the existing server outputs. They do not launch GPU training or overwrite models.

Run in MVSelect-main after copying these four reporting files into the repository. Python dependencies are numpy, pandas and matplotlib. The existing E18 core files and paper_mechanism_assets must remain present.

```bash
python3 audit_paper_mechanism_results.py --root compare/paper_e18_20260916 --output compare/paper_e18_reporting_20260916
python3 summarize_paper_mechanism_uncertainty.py --data-root compare/paper_e18_reporting_20260916
python3 plot_paper_e18.py --data-root compare/paper_e18_reporting_20260916 --output compare/paper_e18_reporting_20260916/figures
```

Use the actual parent containing job01–job15 for --root if you used a different output path. Commands run sequentially on CPU. Nothing needs to be returned unless reproducing these checks reveals an error. No scientific result changes during reporting.

The analysis preserves five source models crossed with three recipient seeds, matched conditions within each job, and three repeated endpoint view sets per object. Primary descriptive intervals additionally resample the three endpoint objects within each category. Source-dependent exact sign-flip checks condition on recipient seeds; random-control tests use only three recipient means. Small model counts and repeated random controls constrain inference. The local gradient diagnostic is retained as inconclusive for long-horizon learning.
