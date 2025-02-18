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
import dataset.dataset as dataset
import plot.plot as plot
import model.model as model

class_list = ['airplane', 'bathtub', 'bed', 'bin', 'bottle', 'bowl', 'bus', 'can', 'case', 'hat']

def select_view(model, data_loader, episodes, folder_path, device, num_classes=10, tau=0.5):
    model.eval()
    all_labels, all_attributes = [], []
    all_scores = {class_id: [] for class_id in range(num_classes)}
    class_indices = {class_id: [] for class_id in range(num_classes)}
    with torch.no_grad():
        for data in data_loader:
            features = data['feature']
            labels = data['label']
            class_names = data['class_name']
            ids = data['id']
            rots = data['rot']
            planars = data['planar']
            rot_input = data['rot_input']

            features = features.to(device)
            rot_input = rot_input.to(device)
            scores = model(features, rot_input)
            for b in range(len(features)):
                all_attributes.append({'class_name': class_names[b],
                                        'id': ids[b],
                                        'rot': rots[b],
                                        'planar': bool(planars[b])})
                c = labels[b].item()
                all_scores[c].append(scores[b].cpu())
                class_indices[c].append(len(all_attributes) - 1)
        
        for c in range(num_classes):
            if len(all_scores[c]) > 0:
                all_scores[c] = torch.stack(all_scores[c])          # shape = [num_samples_in_class]
                all_scores[c] = F.gumbel_softmax(all_scores[c], tau=tau, hard=False)

        view_selection = {}
        for c in range(num_classes):
            dist_c = all_scores[c]  # shape [Nc]
            if dist_c.nelement() == 0:
                continue
            sampled_local_idxs = torch.multinomial(dist_c, args.shot, replacement=False)

            global_idxs = [class_indices[c][loc_id.item()] for loc_id in sampled_local_idxs]
            view_selection[class_list[c]] = []
            for g in global_idxs:
                view_selection[class_list[c]].append(all_attributes[g])
    return view_selection

# Reinforcement learning training loop
def train_rl_selection(train_paths, test_paths, args, num_classes=10):
    seed = random.randint(1, 100)
    root_dir = 'results'
    os.makedirs(root_dir, exist_ok=True)  # Will create the folder if it doesn't exist
    exp_dir = 'rl_lr'+str(args.lr_rl) + '_cls_lr' + str(args.lr_cls) \
    + '_episodes' + str(args.num_episode)+'_epochs'+str(args.epochs) \
    +'_shot'+str(args.shot) + '_sample' + str(args.data_per_class) \
    +'_tau' + str(args.tau) + '_entropy' + str(args.entropy) + '_rot' + str(args.rot) \
    +'_feature' + str(args.feature) + '_pre' + str(args.pretrained)

    stored_folder = os.path.join(root_dir,exp_dir)
    os.makedirs(stored_folder, exist_ok=True)
    store_every_episode = args.num_episode * 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # if args.feature:
    score_network = model.ScoreNetwork(args.input_dim, args.rot).to(device)
    # else:
    #     score_network = model.ScoreRes(args.input_dim, args.rot).to(device)
    score_optimizer = optim.Adam(score_network.parameters(), lr=args.lr_rl)
    
    criterion = nn.CrossEntropyLoss()
    
    train_sampled_paths = []
    train_sampled_labels = []
    test_sampled_paths = []
    test_sampled_labels = []

    for class_name, path_list in train_paths.items():
        sampled_paths = path_list
        sampled_labels = [class_list.index(class_name)]*len(sampled_paths)
        train_sampled_paths += sampled_paths
        train_sampled_labels += sampled_labels
    # if args.feature:
    train_dataset = dataset.FeatureDataset(train_sampled_paths, train_sampled_labels)
    # else:
    #     train_dataset = dataset.ImageDataset(train_sampled_paths, train_sampled_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, pin_memory=True)

    for class_name, path_list in test_paths.items():
        sampled_paths = path_list
        sampled_labels = [class_list.index(class_name)]*len(sampled_paths)
        test_sampled_paths += sampled_paths
        test_sampled_labels += sampled_labels
    if args.feature:
        test_dataset = dataset.FeatureDataset(test_sampled_paths, test_sampled_labels)
    else:
        test_dataset = dataset.ImageDataset(test_sampled_paths, test_sampled_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=False, num_workers=10, pin_memory=True)

    best_episode = 0
    episode_acc = {'best_top-1': 0.0, 'best_top-5': 0.0, 'worst_top-1': float("inf"), 'worst_top-5': float("inf")}

    train_view_selection = {}
    test_view_selection = {}
    baseline_reward = 0  # Running baseline reward for variance reduction
    for episode in range(args.num_episode):
        is_best = False
        # Assign scores to all data
        score_network.train()
        all_features, all_labels, all_attributes, all_images_paths = [], [], [], []
        all_scores = {class_id: [] for class_id in range(num_classes)}
        class_indices = {class_id: [] for class_id in range(num_classes)}
        # with torch.no_grad():
        for data in train_dataloader:
            features = data['feature']
            labels = data['label']
            class_names = data['class_name']
            ids = data['id']
            rots = data['rot']
            rot_input = data['rot_input']
            planars = data['planar']
            image_paths = data['image_path']

            features = features.to(device)
            rot_input = rot_input.to(device)

            scores = score_network(features, rot_input)
            for b in range(len(features)):
                # global append
                all_features.append(features[b])
                all_labels.append(labels[b].item())
                all_attributes.append({'class_name': class_names[b],
                                        'id': ids[b],
                                        'rot': rots[b],
                                        'planar': bool(planars[b])})
                all_images_paths.append(image_paths[b])
                # put this sample’s score into the appropriate class list
                c = labels[b].item()
                all_scores[c].append(scores[b].cpu())
                
                # track that the c-th class’s Nth position corresponds 
                # to global index = len(all_features) - 1
                class_indices[c].append(len(all_features) - 1)
        
        # Apply softmax across the class axis for each sample
        for c in range(num_classes):
            if len(all_scores[c]) > 0:
                all_scores[c] = torch.stack(all_scores[c])          # shape = [num_samples_in_class]
                all_scores[c] = F.gumbel_softmax(all_scores[c], tau=args.tau, hard=False)

        # Select top 5 based on scores per class
        selected_features = []
        selected_labels   = []
        selected_attributes = []
        selected_image_paths = []
        selected_local_indices = []  # to help with policy update
        train_view_selection[episode] = {}
        for c in range(num_classes):
            dist_c = all_scores[c]  # shape [Nc]
            if dist_c.nelement() == 0:
                continue
            # sample, e.g., 5
            sampled_local_idxs = torch.multinomial(dist_c, args.shot, replacement=False)

            # sampled_local_idxs = dist_c.topk(args.shot, dim=0).indices
            # convert those local indices to global
            global_idxs = [class_indices[c][loc_id.item()] for loc_id in sampled_local_idxs]
            train_view_selection[episode][class_list[c]] = []

            for g in global_idxs:
                selected_features.append(all_features[g])
                selected_labels.append(c)
                selected_attributes.append(all_attributes[g])
                selected_image_paths.append(all_images_paths[g])
                # if episode % store_every_episode == 0 or episode==args.num_episode-1:
                train_view_selection[episode][class_list[c]].append(all_attributes[g])

            # We also store the local indices so we can do log_prob later
            # (because log_prob = log( all_scores[c][loc_id] ))
            selected_local_indices.extend(
                [(c, loc_id.item()) for loc_id in sampled_local_idxs]
            )

        # Train classifier on selected data
        classifier = model.initialize_fixed_model(seed, args.feature, args.pretrained).to(device)
        classifier_optimizer = optim.Adam(classifier.parameters(), lr=args.lr_cls, weight_decay=args.wd_cls)
        if args.pretrained:
            selected_dataloader = dataset.SelectedDataset(selected_features, selected_labels, batch_size=args.batch_size)
        else:
            selected_dataloader = dataset.SelectedImageDataset(selected_image_paths, selected_labels, batch_size=args.batch_size)
        episode_acc[episode] = {'top-1': 0.0, 'top-5': 0.0}
        for epoch in range(args.epochs):
            classifier.train()
            epoch_loss = 0.0
            selected_dataloader.shuffle()
            all_outputs = []
            all_labels = []
            for data in selected_dataloader:
                features, targets = data['feature'], data['label']
                if args.feature:
                    features, targets = torch.stack(features).to(device), torch.tensor(targets).to(device)
                else:
                    features, targets = torch.stack(list(features)).to(device), torch.stack(list(torch.tensor(targets))).to(device)
                    # features, targets = features.to(device), torch.stack((torch.tensor(targets)).to(device)
                classifier_optimizer.zero_grad()
                outputs = classifier(features)
                loss = criterion(outputs, targets)
                loss.backward()
                epoch_loss += loss.item()
                classifier_optimizer.step()

            classifier.eval()
            all_outputs = []
            all_labels = []
            correct = 0
            if epoch % args.val_every == 0 or epoch ==args.epochs-1:
                with torch.no_grad():
                    for data in test_dataloader:
                        features = data['feature']
                        targets = data['label']
                        features, targets = features.to(device), targets.to(device)
                        outputs = classifier(features)
                        all_outputs.append(outputs.cpu())
                        all_labels.append(targets.cpu())
                        correct += (outputs.argmax(dim=1) == targets).sum().item()
                # Concatenate all outputs and labels into tensors
                all_outputs = torch.cat(all_outputs, dim=0)  # Shape: [num_samples, num_classes]
                all_labels = torch.cat(all_labels, dim=0)    # Shape: [num_samples]

                top1_predictions = all_outputs.argmax(dim=1)  # Get top-1 predictions
                top1_correct = (top1_predictions == all_labels).sum().item()
                top1_accuracy = top1_correct / len(all_labels)

                top5_predictions = all_outputs.topk(5, dim=1).indices  # Get top-5 class indices
                top5_correct = (top5_predictions == all_labels.view(-1, 1)).any(dim=1).sum().item()  # Check if true label is in top 5
                top5_accuracy = top5_correct / len(all_labels)
                if top1_accuracy > episode_acc[episode]['top-1']:
                    episode_acc[episode]['top-1'] = top1_accuracy
                if top5_accuracy > episode_acc[episode]['top-5']:
                    episode_acc[episode]['top-5'] = top5_accuracy

                if episode_acc[episode]['top-1'] > episode_acc['best_top-1']:
                    episode_acc['best_top-1'] = episode_acc[episode]['top-1']
                elif top1_accuracy < episode_acc['worst_top-1']:
                    episode_acc['worst_top-1'] = top1_accuracy
                if episode_acc[episode]['top-5'] > episode_acc['best_top-5']:
                    episode_acc['best_top-5'] = episode_acc[episode]['top-5']
                elif top5_accuracy < episode_acc['worst_top-5']:
                    episode_acc['worst_top-5'] = top5_accuracy

        reward = episode_acc[episode]['top-1'] + 0.1 * episode_acc[episode]['top-5']

        # ===============
        # Policy Gradient Update
        # ===============
        # We want to maximize reward => minimize -reward =>
        # typical REINFORCE: L = - sum( log pi(a) * R )
        score_optimizer.zero_grad()

        probs = []
        for k, v in all_scores.items():
            probs.append(v)

        probs = torch.stack(probs)
        log_probs = torch.log(probs)
        entropy = -(probs * log_probs).sum()

        baseline_reward = 0.9 * baseline_reward + 0.1 * reward if episode > 0 else reward
        advantage = reward - baseline_reward + 1e-8
        loss = (log_probs * advantage).mean() - args.entropy * entropy
        print(f"Episode {episode}: Top-1 = {episode_acc[episode]['top-1']:.4f} Top-5 = {episode_acc[episode]['top-5']:.4f} loss = {loss.item():.4f}")
        loss.backward()
        score_optimizer.step()

        score_network.eval()

        test_view_selection[episode] = select_view(score_network, test_dataloader, episode, stored_folder, device, num_classes=10, tau=args.tau)        
        if (episode % (args.num_episode // 10) == 0 and episode != 0) or episode == args.num_episode - 1:
            plot_every = args.num_episode // 10
        # if (episode % args.plot_every == 0 and episode!=0) or episode==args.num_episode-1:
            plot.plot_and_save_acc(episode_acc, episode, stored_folder, args.plot_every)
            plot.plot_view_selcetion(train_view_selection, episode, stored_folder, 'train', plot_every, args.shot)
            plot.plot_view_selcetion(test_view_selection, episode, stored_folder, 'test', plot_every, args.shot)
            # plot.plot_view_distribution(train_view_selection, episode, stored_folder, 'train', args.plot_every, args.shot)
            # plot.plot_view_distribution(test_view_selection, episode, stored_folder, 'test', args.plot_every, args.shot)
            with open(os.path.join(stored_folder,'train_view.json'), 'w', encoding='utf-8') as f:
                json.dump(train_view_selection, f, ensure_ascii=False, indent=4)
            with open(os.path.join(stored_folder,'test_view.json'), 'w', encoding='utf-8') as f:
                json.dump(test_view_selection, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Five-Shot Training")
    # parser.add_argument("--data_folder", type=str, default='/nfs/wattrel/data/md0/kung/ShapeNet/feature', help="Path to saved features")
    parser.add_argument("--data_folder", type=str, default='/N/project/ego4d_vlm/ShapeNet/feature', help="Path to saved features")
    parser.add_argument("--num_episode", type=int, default=3000)
    parser.add_argument("--input_dim", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=20, help="Number of examples per class in each run")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--lr_rl", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_cls", type=float, default=7e-5, help="Learning rate")
    parser.add_argument("--wd_cls", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--tau", type=float, default=0.5, help="Learning rate")
    parser.add_argument("--entropy", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--view", type=str, default='random')
    parser.add_argument("--shot", type=int, default=10)
    parser.add_argument("--rot", type=bool, default=True)
    parser.add_argument("--data_per_class", type=int, default=500)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--plot_every", type=int, default=100)
    parser.add_argument("--feature", type=bool, default=True)
    parser.add_argument("--pretrained", type=bool, default=True)
    
    args = parser.parse_args()
    # if args.feature:
    data_folder = '../ShapeNet/feature'
    train_paths = dataset.load_feature_paths(data_folder, 'train', args.data_per_class)
    if args.feature:
        test_paths = dataset.load_feature_paths(data_folder, 'test', args.data_per_class)
    else:
        data_folder = '../ShapeNet/'
    #     train_paths = dataset.load_image_paths(data_folder, 'train', args.data_per_class)
        test_paths = dataset.load_image_paths(data_folder, 'test', args.data_per_class)

    train_rl_selection(train_paths, test_paths, args)
