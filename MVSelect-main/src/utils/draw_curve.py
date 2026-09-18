# import matplotlib
#
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
import json

matplotlib.use('Agg')


import pickle

def load_or_create_list(filepath='pose_table.pkl', non_roll=False):
    """
    Load a list from a pickle file if it exists; otherwise, create it and save it.
    """
    if not non_roll:
        if os.path.exists(filepath):
            # print(f"Loading existing list from {filepath}")
            with open(filepath, "rb") as f:
                values = pickle.load(f)
        else:
            # print(f"File not found, creating new list and saving to {filepath}")
            elevs = np.arange(-180.0, 180.0, 22.5).tolist() 
            azims = np.arange(-180.0, 180.0, 22.5).tolist() 
            rolls = np.arange(-180.0, 180.0, 22.5).tolist() 

            ea = []
            for e in elevs:
                for a in azims:
                    ea.append([e, a])
            values = []
            for r in rolls:
                for ea_i in ea:
                    values.append(ea_i[0], ea_i[1], r)

            with open(filepath, "wb") as f:
                pickle.dump(values, f)
        return values

    else:
        filepath = 'non_roll_pose_table.pkl'
        if os.path.exists(filepath):
            # print(f"Loading existing list from {filepath}")
            with open(filepath, "rb") as f:
                new_values = pickle.load(f)
        else:
            # print(f"File not found, creating new list and saving to {filepath}")
            elevs = np.arange(-180.0, 180.0, 22.5).tolist() 
            azims = np.arange(-180.0, 180.0, 22.5).tolist() 
            rolls = np.arange(-180.0, 180.0, 22.5).tolist() 

            ea = []
            for e in elevs:
                for a in azims:
                    ea.append([e, a])
            values = []
            for r in rolls:
                for ea_i in ea:
                    values.append([ea_i[0], ea_i[1],r])

            new_values = []
            for v in values:
                if v[2] == 0.0:
                    new_values.append(v)

            with open(filepath, "wb") as f:
                pickle.dump(new_values, f)
        return new_values

def load_pose_list(filepath='non_like_pose_table.json'):
    with open(filepath) as f:
        pose_dict = json.load(f)
    return pose_list_from_mapping(pose_dict)


def pose_list_from_mapping(pose_dict):
    N = len(pose_dict)
    if set(pose_dict.values()) != set(range(N)):
        raise ValueError('Camera indices must cover 0 to N-1 exactly.')
    pose_list = [None] * N
    for s, idx in pose_dict.items():
        deg = s.split('_')
        e, a, r = deg[0], deg[1], deg[2]
        pose_list[idx] = [e, a, r]
    return pose_list

def draw_curve(path, x_epoch, train_loss, test_loss, train_result, test_result):
    fig = plt.figure()
    ax1 = fig.add_subplot(121, title="loss")
    ax1.plot(x_epoch, train_loss, 'bo-', label='train' + ': {:.3f}'.format(train_loss[-1]))
    ax1.plot(x_epoch, test_loss, 'ro-', label='test' + ': {:.3f}'.format(test_loss[-1]))
    ax1.legend()
    if train_result is not None and None not in train_result:
        ax2 = fig.add_subplot(122, title="result")
        ax2.plot(x_epoch, train_result, 'bo-', label='train' + ': {:.1f}'.format(train_result[-1]))
    else:
        ax2 = fig.add_subplot(122, title="result")
    ax2.plot(x_epoch, test_result, 'ro-', label='test' + ': {:.1f}'.format(test_result[-1]))
    ax2.legend()
    fig.savefig(path)
    plt.close(fig)


def plot(logdir, acc_history, sel_counts_total, sel_counts_hist, total_selected_hist, \
        longest_ratio_hist, short_ratio_hist, longest_like_ratio_hist, short_like_ratio_hist,\
        longest_count_hist, short_count_hist, longest_like_count_hist, short_like_count_hist, \
        eye_deg_bar_hist, inplane_deg_bar_hist, \
        epochs, steps=2, non_roll=False, non_like=False, pose_mapping=None):
    
    # Color-blind friendly palette (Okabe–Ito)
    colors = {
        "expanded":       "#0072B2",  # Blue
        "foreshortened":  "#009E73",  # Bluish green
        "expanded_like":  "#D55E00",  # Vermillion
        "short_like":     "#CC79A7",  # Reddish purple
        "remainder":      "#56B4E9",  # Sky blue
    }

    if non_roll and non_like:
        tabel_path = 'non_roll_non_like_pose_table.json'
    elif not non_roll and non_like:
        tabel_path = 'non_like_pose_table.json'
    elif non_roll and not non_like:
        tabel_path = 'non_roll_pose_table.json'
    elif not non_roll and not non_like:
        tabel_path = 'pose_table.json'

    pose_table = (pose_list_from_mapping(pose_mapping) if pose_mapping is not None
                  else load_pose_list(tabel_path))
    if len(pose_table) != len(sel_counts_total):
        raise ValueError('Camera mapping and selection-count lengths differ.')

    #Accuracy over episodes
    x = range(1, len(acc_history)+1)
    plt.figure()
    plt.plot(range(1, len(acc_history)+1), acc_history, marker='o')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(10, len(x) + 1, 10))
    plt.xlim(0.5, len(x) + 0.5)
    # plt.xticks(ticks=np.arange(epochs), labels=np.arange(1, epochs + 1))
    # plt.xlim(0.5, epochs + 0.5)
    plt.ylabel("Test accuracy")
    plt.title("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, "accuracy_over_epochs.png"), dpi=200); plt.close()



    # Principal/Planar ratios over episodes
    plt.figure();
    # plt.plot(x, longest_ratio_hist, marker='D', linestyle='-', lw=2, alpha=0.85, label='Expanded ratio')
    plt.plot(x, longest_ratio_hist, marker='D', linestyle='-', lw=2, alpha=0.9, color=colors["expanded"], label='Expanded ratio')
    # plt.plot(x, short_ratio_hist, marker='d', linestyle='--', lw=2, alpha=0.85, label='Foreshortened ratio')
    plt.plot(x, short_ratio_hist, marker='d', linestyle='--', lw=2, alpha=0.9, color=colors["foreshortened"], label='Foreshortened ratio')
    if not non_like:
        # plt.plot(x, longest_like_ratio_hist, marker='*', linestyle='-.', lw=2, alpha=0.85, label='Expanded-like ratio')
        plt.plot(x, longest_like_ratio_hist, marker='*', linestyle='-.', lw=2, alpha=0.9, color=colors["expanded_like"], label='Expanded-like ratio')

        # plt.plot(x, short_like_ratio_hist, marker='x', linestyle=':', lw=2, alpha=0.85, label='Foreshortened-like ratio')
        plt.plot(x, short_like_ratio_hist, marker='x', linestyle=':', lw=2, alpha=0.9, color=colors["short_like"], label='Foreshortened-like ratio')
        
    longest_ratio_hist = np.array(longest_ratio_hist, dtype=float)
    short_ratio_hist = np.array(short_ratio_hist, dtype=float)
    if not non_like:
        longest_like_ratio_hist = np.array(longest_like_ratio_hist, dtype=float)
        short_like_ratio_hist = np.array(short_like_ratio_hist, dtype=float)
        total = (longest_ratio_hist 
             + short_ratio_hist 
             + longest_like_ratio_hist 
             + short_like_ratio_hist)
    else:
        total = longest_ratio_hist + short_ratio_hist
    remainder = 1.0 - total
    plt.plot(x, remainder, marker='o', linestyle='-', lw=2, alpha=0.9,
         color=colors["remainder"], label='Remainder')

    plt.ylabel("Ratio of selections")
    plt.xlabel('Epoch')
    plt.xticks(np.arange(10, len(x) + 1, 10))
    # plt.xticks(ticks=np.arange(epochs), labels=np.arange(1, epochs + 1))
    plt.xlim(0.5, epochs + 0.5)
    if non_like:
        plt.title("Expanded/Foreshortened/Expanded-like/Foreshortened-like ratios over episodes")
    else:
        plt.title("Expanded/Foreshortened ratios over episodes")
    

    plt.legend(); plt.tight_layout();
    plt.savefig(os.path.join(logdir, "view_type_ratios_over_episodes.png"), dpi=200); plt.close()

    # # Principal/Planar counts per episode
    # plt.figure();
    # # plt.plot(range(1, len(principal_count_hist)+1), principal_count_hist, marker='o', label='Principal count/ep')
    # plt.plot(x, longest_count_hist, marker='D', label='Expanded count/ep')
    # plt.plot(x, short_count_hist, marker='d', label='Foreshortened count/ep')
    # plt.plot(x, longest_like_count_hist, marker='*', label='Expanded-like count/ep')
    # plt.plot(x, short_like_count_hist, marker='x', label='Foreshortened-like count/ep')
    # plt.xlabel('Epoch')
    # plt.xticks(x)
    # # plt.xticks(ticks=np.arange(epochs), labels=np.arange(1, epochs + 1))
    # plt.xlim(0.5, epochs + 0.5)

    # plt.ylabel("Count of selected views")
    # plt.title("Expanded/Foreshortened/Expanded-like/Foreshortened-like counts per episode")
    # plt.legend(); plt.tight_layout();
    # plt.savefig(os.path.join(logdir, "view_type_counts_over_episodes.png"), dpi=200); plt.close()

    # Simulate your data (100 epochs × 18 densities)
    data = eye_deg_bar_hist  # Each row sums to 1 (density-like)
    # Convert to array: shape (100, 18)
    arr = np.array(data)

    # Example input: arr.shape = (num_epochs, num_bins)
    num_bins = 9
    arr = np.random.dirichlet(np.ones(num_bins), size=epochs)

    # Example custom y labels (you can build your own list dynamically)
    y_labels = [f"[{10*i},{10*(i+1)})" for i in range(num_bins - 1)] + [f"[{10*(num_bins-1)},{10*num_bins}]"]

    plt.figure(figsize=(10, 5))
    plt.imshow(arr.T, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Density')

    # --- X-axis (Epochs) ---
    plt.xlabel('Epoch')
    plt.xticks(ticks=np.arange(10, epochs + 1, 10), labels=np.arange(10, epochs + 1, 10))
    # plt.xticks(ticks=np.arange(epochs), labels=np.arange(1, epochs + 1))
    plt.xlim(-0.5, epochs - 0.5)

    # --- Y-axis (Bin ranges) ---
    plt.ylabel('Value range')
    plt.yticks(ticks=np.arange(num_bins), labels=y_labels)

    plt.title('eye_deg_heatmap over Epochs')
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, 'eye_deg_heatmap.png'), dpi=200)
    plt.close()


    data = inplane_deg_bar_hist  # Each row sums to 1 (density-like)
    # Convert to array: shape (100, 18)
    arr = np.array(data)
    num_bins = 10
    arr = np.random.dirichlet(np.ones(num_bins), size=epochs)

    # Example custom y labels (you can build your own list dynamically)
    y_labels = [f"[{10*i},{10*(i+1)})" for i in range(8)] + ["[80,90]"] + ["Foreshortened"]
    plt.figure(figsize=(10, 5))
    plt.imshow(arr.T, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Density')

    # --- X-axis (Epochs) ---
    plt.xlabel('Epoch')
    # plt.xticks(ticks=np.arange(epochs), labels=np.arange(1, epochs + 1))
    plt.xticks(ticks=np.arange(10, epochs + 1, 10), labels=np.arange(10, epochs + 1, 10))
    plt.xlim(-0.5, epochs - 0.5)

    # --- Y-axis (Bin ranges) ---
    plt.ylabel('Value range')
    plt.yticks(ticks=np.arange(num_bins), labels=y_labels)

    plt.title('inplane_deg over Epochs')
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, 'inplane_deg.png'), dpi=200)
    plt.close()


    # # Heatmaps across episodes
    # if len(sel_counts_hist) > 0:
    #     arr = np.stack(sel_counts_hist, axis=0)  # [E, V]
    #     plt.figure(); plt.imshow(arr.T, aspect='auto', origin='lower'); plt.colorbar(label="Selections per episode")
    #     plt.xlabel("Episode"); plt.ylabel("Pose index"); plt.title("Per-view selection counts over episodes")
    #     plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "selection_counts_heatmap.png"), dpi=200); plt.close()

    #     denom = np.maximum(1, np.array(total_selected_hist, dtype=np.float32))[:, None]
    #     share = arr / denom
    #     plt.figure(); plt.imshow(share.T, aspect='auto', origin='lower', vmin=0.0, vmax=share.max() if share.size else 1.0); plt.colorbar(label="Share per episode")
    #     plt.xlabel("Episode"); plt.ylabel("Pose index"); plt.title("Per-view selection share over episodes")
    #     plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "selection_share_heatmap.png"), dpi=200); plt.close()

    # Heatmaps across episodes (with tagged high-frequency rows)
    if len(sel_counts_hist) > 0:
        arr = np.stack(sel_counts_hist, axis=0) # [E, V]


    def _pose_string(i):
        pose = pose_table[i]
        string = str(i) + '_e' + str(pose[0]) + '_a' +str(pose[1]) + '_r' +str(pose[2])
        return string

    # # Counts heatmap + tagged rows
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    im = ax.imshow(arr.T, aspect='auto', origin='lower')
    cbar = fig.colorbar(im, ax=ax, label="Selections per episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Pose index")
    ax.set_title("Per-view selection counts over episodes")
    mean_counts = arr.mean(axis=0) # [V]
    topk = steps+5
    topk = int(min(max(1, topk), mean_counts.shape[0]))
    top_ids = np.argsort(mean_counts)[-topk:]
    ax.set_yticks(top_ids)
    ax.set_yticklabels([_pose_string(i) for i in top_ids], fontsize=7)
    for y in top_ids:
        ax.axhline(y=y, lw=0.5, alpha=0.4, color='w')
    fig.tight_layout()
    fig.savefig(os.path.join(logdir, "selection_counts_heatmap.png"), dpi=300)
    plt.close(fig)


    # # Share heatmap + tagged rows
    # denom = np.maximum(1, np.array(total_selected_hist, dtype=np.float32))[:, None]
    # share = arr / denom
    # fig, ax = plt.subplots(figsize=(6.4, 4.8))
    # im = ax.imshow(share.T, aspect='auto', origin='lower', vmin=0.0, vmax=share.max() if share.size else 1.0)
    # cbar = fig.colorbar(im, ax=ax, label="Share per episode")
    # ax.set_xlabel("Episode"); ax.set_ylabel("Pose index"); ax.set_title("Per-view selection share over episodes")
    # mean_share = share.mean(axis=0)
    # top_ids_share = np.argsort(mean_share)[-topk:]
    # ax.set_yticks(top_ids_share)
    # ax.set_yticklabels([_pose_string(i) for i in top_ids_share], fontsize=7)
    # for y in top_ids_share:
    #     ax.axhline(y=y, lw=0.5, alpha=0.4, color='w')
    # fig.tight_layout()
    # fig.savefig(os.path.join(logdir, "selection_share_heatmap.png"), dpi=300)
    # plt.close(fig)


    # Share heatmap + tagged rows (fixed x-axis ticks and labels)
    E = len(total_selected_hist)                      # number of episodes
    eps = 1e-9
    denom = np.maximum(1, np.array(total_selected_hist, dtype=np.float32))[:, None]
    share = arr / denom                               # shape: [E, P]
    P = share.shape[1]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    # Use extent so that episode 1..E land at integer x positions
    im = ax.imshow(
        share.T,                                      # shape: [P, E] -> (rows, cols)
        aspect='auto',
        origin='lower',
        vmin=0.0,
        vmax=max(float(share.max()), eps),
        extent=(0.5, E + 0.5, -0.5, P - 0.5)          # x spans 0.5..E+0.5; y spans -0.5..P-0.5
    )
    cbar = fig.colorbar(im, ax=ax, label="Share per episode")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Pose index")
    ax.set_title("Per-view selection share over episodes")
    # x-axis: show 1..E (you can thin ticks if E is large)

    ax.set_xticks(np.arange(1, E + 1, 10))
    ax.set_xticklabels([str(i) for i in range(10, E + 1, 10)])
    # ax.set_xticklabels([str(i) for i in range(1, E + 1)])
    # ----- Tagged rows (top-k by mean share) -----
    mean_share = share.mean(axis=0)                   # [P]
    top_ids_share = np.argsort(mean_share)[-topk:]
    ax.set_yticks(top_ids_share)
    ax.set_yticklabels([_pose_string(i) for i in top_ids_share], fontsize=7)
    # Light guide lines on selected rows
    for y in top_ids_share:
        ax.axhline(y=y, lw=0.5, alpha=0.4, color='w')
    fig.tight_layout()
    fig.savefig(os.path.join(logdir, "selection_share_heatmap.png"), dpi=300)
    plt.close(fig)



    # Trend lines for top views (share)
    episodes = np.arange(1, arr.shape[0]+1)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i in top_ids:
        ax.plot(episodes, share[:, i], label=_pose_string(i), linewidth=1.0)
    ax.set_xlabel("Episode"); ax.set_ylabel("Selection share")
    ax.set_title("Top views selection share trend")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(logdir, "top_views_share_trend.png"), dpi=300)
    plt.close(fig)

    return remainder.tolist()
