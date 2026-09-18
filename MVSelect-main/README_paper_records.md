# Collect the existing paper records

This helper collects existing results and a list of rendered filenames. It does
not train a model, evaluate a model, load model checkpoints, or read image pixels.
It uses the Python standard library; no additional packages are needed.

After transferring `collect_paper_records.py` to the remote `MVSelect-main`
directory, run there:

```bash
python3 collect_paper_records.py
```

The default image directory is the one specified for RGB in `main.py`:
`/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_23`.
If the images actually used are elsewhere, provide that directory with
`--dataset-root /actual/path`. Run against the dataset used for the paper.

Return the newly created `paper_records_<UTC timestamp>.tar.gz`. It contains:

- A compressed list of image filenames, object IDs, categories, split folders,
  camera directions, original filename flags, and labels from the current parser.
- Object counts, train/test object-ID overlaps, and candidate counts reconstructed
  using the current first-25/train and first-5/test rule. These are explicitly
  marked as reconstructed rather than historical run manifests.
- An index of existing result files under `logs`, `meta_logs`, `compare` and
  `results`, including the companion `human_multiview-main` repository if present.
- Copies of selected existing small summary/configuration/metric files, capped at
  5 MiB per file and 64 MiB total. Larger files remain listed in the index so that
  the next request can name their exact paths. Images and checkpoints are not copied.
- Current source snapshots and saved pose tables. A current source snapshot does
  not establish which source version produced an old run.

Existing files are left intact. The collector refuses to reuse an output folder
or archive name. If a previously used dataset is no longer present, do not
substitute a regenerated dataset without identifying it as such.

Validation: checked on a small fixture for train/test overlap detection,
first-five test inclusion, view-type denominators, filename-to-direction
conversion and malformed filenames. Also checked the command's help output.
No research experiment has been run by this validation.
