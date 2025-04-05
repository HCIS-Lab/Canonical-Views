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

