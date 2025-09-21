import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import numpy as np
import argparse
import json
import os
import copy
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class_list = ['airplane', 'bathtub', 'bed', 'bin', 'bottle', 'bowl', 'bus', 'can', 'case', 'hat']

def plot_and_save_acc(acc_dict, episodes, folder_path, plot_every):
    # Extract episode numbers (sorted) and corresponding metrics
    episode_list = []
    for episode in list(acc_dict.keys()):
        if not isinstance(episode, str):
            episode_list.append(episode)
    episode_list = episode_list[::plot_every]
    top1_acc = [acc_dict[episode]['top-1'] for episode in episode_list]
    top5_acc = [acc_dict[episode]['top-5'] for episode in episode_list]
    # mae = [acc_dict[episode]['mae'] for episode in episode_list]
    # Create the plot

    plt.figure(figsize=(10, 6))
    plt.plot(episode_list, top1_acc, marker='o', label='Top-1 Accuracy')
    plt.plot(episode_list, top5_acc, marker='o', label='Top-5 Accuracy')
    # plt.plot(episode_list, mae, marker='o', label='MAE')

    # Customize the plot
    plt.xlabel('Episode')
    plt.ylabel('Accuracy (%)')
    plt.title('Episode-wise Top-1 and Top-5 Accuracy')
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(os.path.join(folder_path, 'episode_accuracy.png'), dpi=300, bbox_inches='tight')

    with open(os.path.join(folder_path, 'acc.json'), "w", encoding="utf-8") as f:
        json.dump(acc_dict, f, ensure_ascii=False, indent=4)


def in_angle_range(angle, category, offset):
        lower_bound = (category - offset) 
        upper_bound = (category + offset)
        return lower_bound <= angle <= upper_bound

def check_angles(view):
    x, y, z = view
    offset = 15./180.
    categories = [
        -1.,
        0.5,
        0.,
        0.5,
        1.
    ]
    planar = [False]*3
    for i, angle in enumerate([x, y, z]):
        for cat in categories:
            if in_angle_range(float(angle), cat, offset):
                planar[i] = True
                break

    return all(planar)


def plot_and_save_acc_discrete(episode_dict, folder_path, train=False):
    list_acc = []
    episode_list = list(episode_dict.keys())
    episode_list.sort()
    num_episode = len(episode_list)
    x_axis = []
    for episode in range(num_episode):
        list_acc.append(episode_dict[episode])
        x_axis.append(episode)

    # ----- Smooth: average every 50 episodes -----
    window_size = 5
    smooth_x = []
    smooth_p = []

    for i in range(0, len(x_axis), window_size):
        window = list_acc[i:i+window_size]
        if len(window) > 0:
            smooth_x.append(np.mean(x_axis[i:i+window_size]))
            smooth_p.append(np.mean(window))

    plt.figure(figsize=(10, 6))
    plt.plot(smooth_x, smooth_p, marker='o', label='acc')

    # Add labels and title
    plt.xlabel('Episode')
    plt.ylabel('Acc')
    if train:
        plt.title('Train ACC Over Episodes')
    else:
        plt.title('ACC Over Episodes')

    # Add a legend to distinguish between the metrics
    plt.legend()

    # Optional: add grid for better readability
    plt.grid(True)

    # Save the plot (optional)
    if train:
        plt.savefig(os.path.join(folder_path, 'acc_train_over_episode.png'), dpi=300, bbox_inches='tight')
    else:
        plt.savefig(os.path.join(folder_path, 'acc_over_episode.png'), dpi=300, bbox_inches='tight')

def plot_view_selcetion_discrete(episode_dict, folder_path):

    list_p = []
    list_non_p = []
    episode_list = list(episode_dict.keys())
    episode_list.sort()
    num_episode = len(episode_list)
    x_axis = []
    for episode in range(num_episode):
        num_p = 0
        num_non_p = 0
        for i, view in enumerate(episode_dict[episode]):
            planar = check_angles(view)
            if planar:
                num_p+=1
            else:
                num_non_p+=1
        list_p.append(float(num_p/len(episode_dict[episode])))
        x_axis.append(episode)

    # ----- Smooth: average every 50 episodes -----
    window_size = 5
    smooth_x = []
    smooth_p = []

    for i in range(0, len(x_axis), window_size):
        window = list_p[i:i+window_size]
        if len(window) > 0:
            smooth_x.append(np.mean(x_axis[i:i+window_size]))
            smooth_p.append(np.mean(window))

    plt.figure(figsize=(10, 6))
    plt.plot(smooth_x, smooth_p, marker='o', label='planar')

    # Add labels and title
    plt.xlabel('Episode')
    plt.ylabel('View ratio')
    plt.title('View Over Episodes')

    # Add a legend to distinguish between the metrics
    plt.legend()

    # Optional: add grid for better readability
    plt.grid(True)

    # Save the plot (optional)
    plt.savefig(os.path.join(folder_path, 'view_over_episode.png'), dpi=300, bbox_inches='tight')


def plot_selected_entropy(first_entropy_dict, sec_entropy_dict, folder_path):

    list_first = []
    list_sec = []
    episode_list = list(first_entropy_dict.keys())
    episode_list.sort()
    num_episode = len(episode_list)
    x_axis = []
    for episode in range(num_episode):

        avg_first = float(sum(first_entropy_dict[episode])/len(first_entropy_dict[episode]))
        avg_sec = float(sum(sec_entropy_dict[episode])/len(sec_entropy_dict[episode]))

        list_first.append(avg_first)
        list_sec.append(avg_sec)
        x_axis.append(episode)

    # ----- Smooth: average every 50 episodes -----
    window_size = 5
    smooth_x = []
    smooth_first = []
    smooth_sec = []

    for i in range(0, len(x_axis), window_size):
        window_first = list_first[i:i+window_size]
        window_sec = list_sec[i:i+window_size]
        if len(window_first) > 0:
            smooth_x.append(np.mean(x_axis[i:i+window_size]))
            smooth_first.append(np.mean(window_first))
            smooth_sec.append(np.mean(window_sec))

    plt.figure(figsize=(10, 6))
    plt.plot(smooth_x, smooth_first, marker='o', label='first-order')
    plt.plot(smooth_x, smooth_sec, marker='x', label='second-order')  # change label accordingly
    # Add labels and title
    plt.xlabel('Episode')
    plt.ylabel('Entropy')
    plt.title('Entropy Over Episodes')

    # Add a legend to distinguish between the metrics
    plt.legend()

    # Optional: add grid for better readability
    plt.grid(True)

    # Save the plot (optional)
    plt.savefig(os.path.join(folder_path, 'entropy_over_episode.png'), dpi=300, bbox_inches='tight')

def plot_loss(loss_list, folder_path):

    num_episode = len(loss_list)
    x_axis = []

    # ----- Smooth: average every 50 episodes -----
    window_size = 5
    smooth_x = []
    smooth_loss = []
    for episode in range(num_episode):
        x_axis.append(episode)
    for i in range(0, len(x_axis), window_size):
        window_loss = loss_list[i:i+window_size]
        if len(window_loss) > 0:
            smooth_x.append(np.mean(x_axis[i:i+window_size]))
            smooth_loss.append(float(sum(window_loss)/window_size))

    plt.figure(figsize=(10, 6))
    plt.plot(smooth_x, smooth_loss, marker='o', label='loss')
    
    # Add labels and title
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss Over Episodes')

    # Add a legend to distinguish between the metrics
    plt.legend()

    # Optional: add grid for better readability
    plt.grid(True)

    # Save the plot (optional)
    plt.savefig(os.path.join(folder_path, 'loss_over_episode.png'), dpi=300, bbox_inches='tight')

def plot_view_selcetion(view_selection, episodes, folder_path, split, plot_every, shot):


    list_p = []
    list_non_p = []
    episode_list = list(view_selection.keys())
    episode_list.sort()
    num_episode = len(episode_list)
    num_section = num_episode//plot_every
    total_view_section = shot*plot_every*10
    x_axis = []
    for sec in range(num_section):
        num_p = 0
        num_non_p = 0
        for in_episode in range(plot_every):
            for i, class_name in enumerate(class_list):
                view_list = view_selection[sec*plot_every + in_episode][class_name]
                for data in view_list:
                    planar = data['planar']
                    if planar:
                        num_p+=1
                    else:
                        num_non_p+=1
        list_p.append(float(num_p/total_view_section))
        list_non_p.append(float(num_non_p/total_view_section))
        x_axis.append(str(sec*plot_every))

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, list_p, marker='o', label='planar')
    # plt.plot(x_axis, list_non_p, marker='o', label='non-planar')

    # Add labels and title
    plt.xlabel('Episode')
    plt.ylabel('View ratio')
    plt.title('View Over Episodes')

    # Add a legend to distinguish between the metrics
    plt.legend()

    # Optional: add grid for better readability
    plt.grid(True)

    # Save the plot (optional)
    plt.savefig(os.path.join(folder_path, split+'_view_over_episode.png'), dpi=300, bbox_inches='tight')

def plot_set_selcetion(set_selection, episodes, folder_path, split, plot_every, shot):

    episode_list = list(set_selection.keys())
    episode_list.sort()
    x_axis = []
    list_p = []
    for episode in episode_list:
        list_p.append(float(set_selection[episode]*0.1))
        x_axis.append(str(episode))

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, list_p, marker='o', label='planar')

    # Add labels and title
    plt.xlabel('Episode')
    plt.ylabel('planar ratio')
    plt.title('View Over Episodes')

    # Add a legend to distinguish between the metrics
    plt.legend()

    # Optional: add grid for better readability
    plt.grid(True)

    # Save the plot (optional)
    plt.savefig(os.path.join(folder_path, split+'_view_over_episode.png'), dpi=300, bbox_inches='tight')
    
def plot_view_distribution(view_selection, episode, folder_path, split, plot_every, shot):

    x_list = []
    y_list = []
    z_list = []
    color_list = []

    start = episode - plot_every
    end = episode - 1
    for epi in range(start, end+1):

        for i, class_name in enumerate(class_list):
            view_list = view_selection[epi][class_name]
            for data in view_list:
                rots = data['rot']
                rots = rots.split('_')
                planar = data['planar']
                x_list.append(float(rots[0]))
                y_list.append(float(rots[1]))
                z_list.append(float(rots[2]))
                if planar:
                    color = (1.,0.,0.)
                else:
                    color = (0.,0.,0.)
                color_list.append(color)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x_list, y_list, z_list, c=color_list, alpha=0.7)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    os.makedirs(os.path.join(folder_path, 'view_dist_'+split), exist_ok=True)
    plt.savefig(os.path.join(folder_path, 'view_dist_'+split, str(start)+'~'+str(end)+".png"), dpi=300)

