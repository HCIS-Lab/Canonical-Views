# Mid-Level Visible-Shape Feature Analysis

This analysis asks whether MVSelect learns to select views that expose useful
mid-level shape structure, rather than merely converging to an arbitrary
canonical-looking camera convention.

The features are computed from the already-rendered ModelNet images. No
ShapeNet mesh access or dataset regeneration is required for this analysis.
The measurements are therefore **2D visible-view descriptors**: they describe
what structure is exposed in a particular rendered view, not the object's full
3D ground-truth structure.

## Data Flow

`midlevel_shape_features.py` does three things:

1. Reads every rendered PNG under:

   ```text
   <data_root>/<class_name>/<split>/*.png
   ```

2. Computes silhouette, skeleton, and edge descriptors for every candidate
   view.

3. Joins those descriptors to `*_selection.json`, so each epoch is summarized
   by the views actually selected by the agent.

The default comparison value is:

```text
lift_metric(epoch)
  = mean(metric on agent-selected views at epoch)
    - mean(metric over all candidate views of the same object instances)
```

So a positive lift means:

> For the same test objects, the selector chose views that expose more of this
> mid-level cue than the average available camera view.

This instance-matched baseline is important because it prevents class-level
shape differences from masquerading as selection preferences.

## Shared Preprocessing

All metrics start from the same image-derived mask:

1. Load image as RGB.
2. Convert to grayscale by channel mean.
3. Treat pixels with grayscale `< 0.97` as object pixels.
4. Fill holes.
5. Apply binary opening and closing.
6. Keep only the largest connected component.

This assumes the rendered objects sit on a near-white background, which matches
the current dataset.

## Metric Groups

### 1. Axis Visibility

These metrics ask whether the visible silhouette exposes a clear, stable object
axis.

| Metric | High-Level Interpretation | Implementation |
|---|---|---|
| `ellipse_orientation_deg` | The dominant visible silhouette axis direction in the image plane. Useful for asking whether selected views expose consistent upright/elongated axes, but not a "higher is better" score. | Take all object-mask pixel coordinates `(x, y)`, center them, compute their 2D covariance matrix, eigendecompose it, and use the eigenvector with the largest eigenvalue as the main axis. Convert that eigenvector angle to degrees in `[0, 180)`. |
| `ellipse_aspect_ratio` | How strongly elongated the visible silhouette is. Higher means the object view has a clearer major axis; values near 1 mean the silhouette is more round/square. | From the same covariance eigenvalues, compute `sqrt(lambda_major / lambda_minor)`. |
| `skeleton_length_px` | Raw length of the visible medial skeleton in pixels. Higher usually means the silhouette contains a longer visible shape spine or extended parts. | Thin the binary object mask using Zhang-Suen skeletonization, then count skeleton pixels. |
| `skeleton_length_norm` | Scale-normalized skeleton length. Higher means the visible shape has a longer medial structure relative to object area. | `skeleton_length_px / sqrt(mask_area)`. |
| `skeleton_elongation` | How much the skeleton stretches along the object's visible extent. Higher means the skeleton occupies a long path relative to the bounding box. | `skeleton_length_px / max(bbox_width, bbox_height)`. |

Advisor-facing summary:

> Axis visibility is high when the selected view exposes a long, stable
> silhouette axis or medial spine, rather than a compressed or ambiguous outline.

### 2. Symmetry / Part Organization

These metrics ask whether the visible shape exposes organized bilateral or
medial structure.

| Metric | High-Level Interpretation | Implementation |
|---|---|---|
| `bilateral_symmetry` | How mirror-symmetric the visible silhouette is. Higher means the silhouette is more symmetric under either left-right or up-down reflection. | Crop the object mask to its bounding box. Compute IoU between the crop and its left-right mirror, and IoU between the crop and its up-down mirror. Use the maximum of the two IoUs. Range is approximately `[0, 1]`. |
| `medial_axis_symmetry` | How mirror-symmetric the medial skeleton is. Higher means the visible internal shape spine has a more symmetric organization. | Apply the same mirror-IoU computation to the skeleton image instead of the full object mask. |
| `skeleton_endpoint_count` | Number of visible skeleton tips. Higher can indicate more visible protrusions or part endpoints. | On the skeleton, convolve with a `3x3` all-ones kernel to count neighboring skeleton pixels. Endpoints are skeleton pixels with exactly one skeleton neighbor. |
| `skeleton_branchpoint_count` | Number of visible skeleton branch junctions. Higher can indicate richer visible part structure. | Branchpoints are skeleton pixels with three or more skeleton neighbors. |
| `skeleton_branch_density` | Branching normalized by skeleton size. Higher means the visible skeleton is more branchy per unit length. | `skeleton_branchpoint_count / skeleton_length_px`. |

Advisor-facing summary:

> Symmetry / part organization is high when the selected view exposes a
> coherent silhouette or skeleton organization, especially bilateral structure
> and visible branching that may support stable object representation.

### 3. Edge Organization

These metrics ask whether local image edges are coherent or noisy/isotropic.

| Metric | High-Level Interpretation | Implementation |
|---|---|---|
| `dominant_edge_orientation_deg` | The dominant edge direction in the visible object. Like ellipse orientation, this is a direction descriptor, not a "higher is better" score. | Compute Sobel gradients `dx` and `dy`, then edge orientation `theta = atan2(dy, dx) mod pi`, so opposite directions are treated as the same orientation. Use a magnitude-weighted axial circular mean to get an angle in `[0, 180)`. |
| `edge_entropy` | How spread out edge orientations are. Higher means edges point in many different directions; lower means orientations are concentrated. | Inside the object mask, keep pixels whose gradient magnitude is at least the 75th percentile of masked gradient magnitudes. Build an 18-bin magnitude-weighted orientation histogram over `[0, pi)`, normalize it, then compute Shannon entropy divided by `log(18)`. Range is approximately `[0, 1]`. |
| `edge_anisotropy` | How strongly edge orientations align to a dominant axis. Higher means edges are directionally organized; lower means edge orientations are isotropic. | Compute the magnitude-weighted axial resultant length: `abs(sum(w * exp(2j * theta))) / sum(w)`. Range is approximately `[0, 1]`. |
| `edge_pixel_count` | Number of strong edge pixels used in the edge-orientation summary. | Count pixels passing the masked 75th-percentile gradient threshold. |

Advisor-facing summary:

> Edge organization is high when selected views contain coherent, directionally
> structured edges rather than uniformly distributed or noisy edge orientations.

## Orientation Handling

Orientation metrics are axial, not directional:

```text
0 degrees == 180 degrees
```

So epoch summaries use axial circular averaging:

```text
mean_angle = 0.5 * angle(mean(exp(2j * theta)))
```

Likewise, orientation lift uses the shortest signed axial angular difference in
`[-90, 90)`.

Because orientation does not naturally mean "more" or "less" structure,
`ellipse_orientation_deg` and `dominant_edge_orientation_deg` are saved in the
CSVs but excluded from the default heatmaps. Use them for targeted analyses,
for example whether selected views converge to a consistent image-plane axis.

## Output Columns

For every metric, the selected-view summary contains:

| Column pattern | Meaning |
|---|---|
| `selected_<metric>` | Mean metric value over views selected by the agent at that epoch. |
| `baseline_<metric>` | Mean metric value over all candidate views from the same selected object instances. |
| `lift_<metric>` | `selected_<metric> - baseline_<metric>`, or shortest axial angular difference for orientation metrics. |

The script also records:

| Column | Meaning |
|---|---|
| `n_selected` | Number of selected view records contributing to the epoch summary. |
| `n_baseline_views` | Number of candidate views from the same object instances used for the baseline. |
| `n_missing` | Selected filenames that could not be matched to the rendered-image cache. Should usually be `0`. |
| `prop_expanded`, `prop_Expanded-like`, etc. | Proportion of selected views in each filename-derived view bucket. |

## Commands

Single experiment:

```bash
cd MVSelect-main

python3 midlevel_shape_features.py \
  --selection_dir meta_logs/rgb/resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100 \
  --bin_epochs 10
```

Freeze sweep:

```bash
./run_midlevel_shape_features_pipeline.sh
```

Selector-limit sweep:

```bash
COMPARISON_SET=selector_limit ./run_midlevel_shape_features_pipeline.sh
```

Plot raw selected values instead of selected-minus-baseline lift:

```bash
VALUE=selected STYLE=both ./run_midlevel_shape_features_pipeline.sh
```

Plot orientation columns explicitly:

```bash
METRICS="ellipse_orientation_deg dominant_edge_orientation_deg" \
VALUE=selected STYLE=line \
./run_midlevel_shape_features_pipeline.sh
```

## Caveats

1. These are 2D visible-view proxies, not true 3D shape descriptors. They test
   what structure a rendered camera view exposes.

2. ShapeNet meshes are not needed unless the target claim changes to
   mesh-intrinsic structure: true 3D axes, 3D symmetry planes, 3D medial axes,
   or semantic part annotations.

3. The object mask is derived with a simple white-background threshold. It is
   appropriate for the current rendered dataset but should be revisited if the
   background, lighting, or rendering style changes.

4. Skeleton metrics are sensitive to mask quality and image resolution. They
   should be interpreted as visible skeleton organization, not exact anatomical
   or semantic part decomposition.

5. `skeleton_endpoint_count`, `skeleton_branchpoint_count`, and
   `skeleton_branch_density` measure branch-like topology. They do not identify
   named semantic parts.

6. Orientation columns do not have a simple "higher is better" interpretation.
   They are included in CSVs but excluded from default heatmaps.

7. The selected-view analysis follows `selection.json`, so it measures the
   agent-selected views. It does not add the initial input view unless that view
   is explicitly present in the selection file.

8. A positive lift should be read as an instance-controlled selection bias:
   among the same objects, the agent picked views with more of that visible cue
   than the objects' average available view.

## Suggested Figure Caption

> We quantified mid-level visible shape structure in each rendered view using
> image-derived silhouette, skeleton, and edge descriptors. For each training
> epoch, we compared the agent-selected views against all candidate views of the
> same test objects. Positive values therefore indicate that the selector chose
> views exposing more axis visibility, symmetry/part organization, or edge
> organization than the instance-matched all-view baseline.
