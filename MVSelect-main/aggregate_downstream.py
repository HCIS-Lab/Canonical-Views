import os
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = "downstream_logs"
ARCHITECTURES = ["resnet18", "vit", "tinyvit"]
REP_LIST = ['random-fixed-all', 'random-fixed-class', 'random', 'rgb', 'depth', 'edge', 'rgb_depth', 'rgb_edge', 'depth_edge', 'rgb_depth_edge']
VIEW_TYPE_SELECTION = ['0', '4', '04', '01234']
SELECT_OR_NOT = ['random-fixed-all', 'random-fixed-class', 'agent', 'random']
NUM_VIEWS = ['numviews1','numviews3','numviews5']
# NUM_VIEWS = ['numviews3']
SETTINGS = ['train_ins25_lr5e-05base1.0other1.0', 'train_ins10_lr5e-05base1.0other1.0']
# SETTINGS = ['train_ins25_lr5e-05base1.0other1.0']

for arch in ARCHITECTURES:
    arch_root = os.path.join(ROOT, arch)
    # Preserve support for ResNet results written before architecture-scoped
    # downstream paths were introduced.
    roots = [arch_root]
    if arch == "resnet18":
        roots.append(ROOT)
    for active_root in roots:
        if not os.path.isdir(active_root):
            continue
        for REP in REP_LIST:
            print(f"{'-'*10} {arch}")
            REP_ROOT = os.path.join(active_root, REP)
            for view_type in VIEW_TYPE_SELECTION:
                VIEW_ROOT = os.path.join(REP_ROOT, view_type)
                if not os.path.isdir(VIEW_ROOT):
                    continue
                for select in SELECT_OR_NOT:
                    SELECT_ROOT = os.path.join(VIEW_ROOT, select)
                    if not os.path.isdir(SELECT_ROOT):
                        continue
                    for num_views in NUM_VIEWS:
                        num_view_root = os.path.join(SELECT_ROOT, num_views)
                        if not os.path.isdir(num_view_root):
                            continue
                        for setting in SETTINGS:
                            SETTING_ROOT = os.path.join(num_view_root, setting)
                            if not os.path.isdir(SETTING_ROOT):
                                continue
                            runs = []
                            run_names = []
                            for fn in os.listdir(SETTING_ROOT):
                                if fn.endswith("meta.json"):
                                    with open(os.path.join(SETTING_ROOT, fn), "r") as f:
                                        runs.append(json.load(f)['best_accuracy'])
                                        run_names.append(fn)
                            if len(runs) > 0:
                                label = '_'.join([
                                    arch, REP, view_type, select, num_views, setting])
                                print(f"{label}: {sum(runs) / len(runs)}")
