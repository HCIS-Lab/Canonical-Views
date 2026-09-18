#!/usr/bin/env python3
"""Plot the common-five perturbation results; SD is over trained runs."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def read_inputs(inputs):
    """Combine disjoint checkpoint batches only when their test protocols match."""
    all_rows, sources, seen, baseline = [], [], set(), None
    conditions = {'unrestricted', 'freeze_10', 'freeze_20', 'freeze_30'}
    tests = {'clean', 'rotation', 'colour_jitter', 'rotation_and_colour_jitter'}
    settings_keys = ('repeats', 'num_views', 'seed', 'rotation_deg',
                     'jitter_strength', 'hue_strength', 'batch_size')
    run_sets = None
    for directory in inputs:
        plan_path, csv_path = directory / 'evaluation_plan.json', directory / 'run_summary.csv'
        plan = json.loads(plan_path.read_text())
        if not plan.get('completed') or plan.get('smoke_test') or plan.get('completed_scope') != 'full_protocol':
            raise ValueError(f'Completed full evaluation required: {directory}')
        if plan.get('protocol') != 'common_random_five_saved_classifier_checkpoints' or plan['settings']['num_views'] != 5:
            raise ValueError(f'Expected the common-five protocol: {directory}')
        epochs = plan['settings']['epochs']
        if not epochs or len(set(epochs)) != len(epochs) or any(type(e) is not int or e < 1 for e in epochs):
            raise ValueError(f'Invalid or duplicate epochs: {directory}')
        signature = {
            'settings': {k: plan['settings'][k] for k in settings_keys},
            'architecture': plan['architecture'], 'trials': plan['trials'],
            'instances_sha256': plan['instances_sha256'], 'script_sha256': plan['script_sha256'],
            'shared_tests': plan.get('test_images_shared_across_all_models_and_epochs'),
        }
        if baseline is not None and signature != baseline:
            raise ValueError('Batches differ in test images, perturbations, settings, source code or instance manifest; do not pool them.')
        baseline = signature
        jobs = plan['jobs']
        keys = {(j['condition'], j['run'], j['epoch']) for j in jobs}
        if len(keys) != len(jobs) or seen & keys:
            raise ValueError('Repeated model/checkpoint jobs within or between batches; do not count them twice.')
        if {c for c, _, _ in keys} != conditions or {e for _, _, e in keys} != set(epochs):
            raise ValueError('Expected all four training conditions and the listed epochs.')
        current_runs = {c: {r for cc, r, _ in keys if cc == c} for c in conditions}
        if any(len(runs) != 5 for runs in current_runs.values()) or (run_sets is not None and current_runs != run_sets):
            raise ValueError('Expected the same five trained runs per condition in every batch.')
        run_sets = current_runs
        expected_jobs = {(c, r, e) for c, runs in run_sets.items() for r in runs for e in epochs}
        if keys != expected_jobs or any(not j.get('checkpoint_sha256') for j in jobs):
            raise ValueError('Missing checkpoint jobs or checkpoint hashes.')
        rows = list(csv.DictReader(csv_path.open()))
        row_keys = [(r['condition'], r['run'], int(r['epoch']), r['test_condition']) for r in rows]
        if len(set(row_keys)) != len(rows) or set(row_keys) != {(*k, t) for k in keys for t in tests}:
            raise ValueError('Missing, duplicate or unexpected summary rows.')
        n_trials = len(plan['trials'])
        n_objects = len({t['object_id'] for t in plan['trials']})
        clean = {(r['condition'], r['run'], r['epoch']): float(r['accuracy_pct']) for r in rows if r['test_condition'] == 'clean'}
        for row in rows:
            if (int(row['n_trials']) != n_trials or int(row['n_objects']) != n_objects
                    or int(row['n_sampling_repeats']) != plan['settings']['repeats']):
                raise ValueError('Summary denominators do not match the evaluation plan.')
            accuracy, drop = float(row['accuracy_pct']), float(row['accuracy_drop_percentage_points'])
            if not math.isfinite(accuracy) or not math.isfinite(drop) or not 0 <= accuracy <= 100:
                raise ValueError('Invalid accuracy or accuracy drop.')
            if abs(clean[row['condition'], row['run'], row['epoch']] - accuracy - drop) > 1e-8:
                raise ValueError('Reported drop differs from clean minus perturbed accuracy.')
        seen.update(keys)
        all_rows.extend(rows)
        sources.append(dict(directory=str(directory.resolve()), epochs=sorted(epochs),
                            evaluation_plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                            run_summary_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                            registry_sha256=plan['registry_sha256']))
    return all_rows, sorted({int(r['epoch']) for r in all_rows}), sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, nargs='+', required=True,
                        help='One or more completed evaluation directories. Checkpoint batches must not overlap.')
    parser.add_argument('--output', type=Path, required=True, help='New figure directory')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory.')
    try:
        rows, epochs, sources = read_inputs(args.input)
    except (ValueError, KeyError, OSError) as ex:
        parser.error(str(ex))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    styles = [('unrestricted', 'Continued selection', '#222222', 'o'),
              ('freeze_10', 'Freeze after epoch 10', '#0072B2', 's'),
              ('freeze_20', 'Freeze after epoch 20', '#E69F00', '^'),
              ('freeze_30', 'Freeze after epoch 30', '#CC79A7', 'D')]
    panels = [('clean', 'Clean accuracy', 'Accuracy (%)', 'accuracy_pct'),
              ('rotation', 'In-plane rotation (within ±10°)', 'Accuracy drop (percentage points)', 'accuracy_drop_percentage_points'),
              ('colour_jitter', 'Colour jitter', 'Accuracy drop (percentage points)', 'accuracy_drop_percentage_points'),
              ('rotation_and_colour_jitter', 'Rotation and colour jitter', 'Accuracy drop (percentage points)', 'accuracy_drop_percentage_points')]
    fig, axes = plt.subplots(2, 2, figsize=(8.3, 6.5))
    points = []
    for index, (ax, (test, title, ylabel, field)) in enumerate(zip(axes.flat, panels)):
        for condition, label, color, marker in styles:
            means, sds = [], []
            for epoch in epochs:
                group = [r for r in rows if r['condition'] == condition and int(r['epoch']) == epoch and r['test_condition'] == test]
                if len(group) != 5 or len({r['run'] for r in group}) != 5:
                    parser.error(f'Expected five distinct runs for {condition}, epoch {epoch}, {test}.')
                vals = [float(r[field]) for r in group]
                mean, sd = statistics.mean(vals), statistics.stdev(vals)
                means.append(mean)
                sds.append(sd)
                points.append(dict(condition=condition, epoch=epoch, test_condition=test, metric=field,
                                   mean=mean, sample_sd=sd, n_trained_runs=5))
            ax.errorbar(epochs, means, yerr=sds, label=label, color=color, marker=marker,
                        linewidth=1.4, markersize=4.3, capsize=2.6, elinewidth=.8)
        ax.set_title(title, fontsize=10, loc='left', pad=9)
        ax.text(-.17, 1.08, 'abcd'[index], transform=ax.transAxes, fontsize=12, fontweight='bold')
        ax.set_xlabel('Classifier training epoch')
        ax.set_ylabel(ylabel)
        ax.set_xticks(epochs)
        ax.set_xlim(min(epochs) - 5, max(epochs) + 5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.grid(axis='y', alpha=.15)
        if test != 'clean':
            ax.axhline(0, color='#888888', linewidth=.7, linestyle=':')
        # Keep a useful baseline range, expanding to show every mean and SD.
        lower, upper = {'clean': (50, 90), 'rotation': (-3, 5)}.get(test, (-5, 45))
        panel_points = [p for p in points if p['test_condition'] == test]
        observed_low = min(p['mean'] - p['sample_sd'] for p in panel_points)
        observed_high = max(p['mean'] + p['sample_sd'] for p in panel_points)
        padding = max(1, (observed_high - observed_low) * .05)
        ax.set_ylim(min(lower, observed_low - padding), max(upper, observed_high + padding))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .065), ncol=2, frameon=False, fontsize=9)
    fig.text(.5, .027, 'Identical five-image tests across models. Mean ± SD over five trained runs per condition.',
             ha='center', fontsize=8)
    fig.text(.5, .008, f'Panels use different vertical ranges. Lines connect {len(epochs)} measured checkpoints.', ha='center', fontsize=8)
    fig.subplots_adjust(left=.10, right=.98, top=.95, bottom=.22, hspace=.52, wspace=.35)
    args.output.mkdir(parents=True)
    for ext in ['png', 'svg']:
        fig.savefig(args.output / f'perturbation_review.{ext}', dpi=220, facecolor='white')
    plt.close(fig)
    with (args.output / 'plotted_values.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(points[0]))
        w.writeheader()
        w.writerows(points)
    with (args.output / 'combined_run_summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r['condition'], r['run'], int(r['epoch']), r['test_condition'])))
    (args.output / 'plot_provenance.json').write_text(json.dumps(dict(
        sources=sources, epochs=epochs, centre='mean of trained-run values',
        uncertainty='sample SD across five trained runs per condition',
        checkpoints_are_repeated_measurements=True,
        note='Protocol and summary consistency checks do not replace an independent audit of trial_results.csv.'
    ), indent=2) + '\n')
    print(f'Saved review figure and plotted values in {args.output}')


if __name__ == '__main__':
    main()
