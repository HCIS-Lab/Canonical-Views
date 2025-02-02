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

class_list = ['airplane', 'bathtub', 'bed', 'bin', 'bottle', 'bowl', 'bus', 'can', 'case', 'hat']
# Define a scoring network
class ScoreNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ScoreNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x.squeeze()

# Define a more complex classification model
class ComplexClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ComplexClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)

        return x

# Dataset class for loading features
def load_feature_paths(data_folder, split='train', view='random'):
    feature_paths = {}
    class_list = ['airplane', 'bathtub', 'bed', 'bin', 'bottle', 'bowl', 'bus', 'can', 'case', 'hat']
    for class_idx, class_name in enumerate(class_list):
        folder_path = os.path.join(data_folder, class_name, class_name, split)
        instances = os.listdir(folder_path)
        feature_paths[class_name] = []
        for ins in instances:
            instances_path = os.path.join(folder_path, ins, 'screenshot')
            for _, _, files in os.walk(instances_path):
                for file in files:
                    if file.endswith(".pt"):
                        feature_paths[class_name].append(os.path.join(instances_path, file))
    return feature_paths

class FeatureDataset(Dataset):
    def __init__(self, feature_paths, labels):
        self.feature_paths = feature_paths
        self.labels = labels
        
    def __len__(self):
        return len(self.feature_paths)
    
    def __getitem__(self, idx):
        feature = torch.load(self.feature_paths[idx]).squeeze()
        label = self.labels[idx]
        file_name = os.path.basename(self.feature_paths[idx])
        attrbutes = file_name.split('_')
        class_name, id, x, y, z, view = attrbutes[0], attrbutes[1], attrbutes[2], attrbutes[3], attrbutes[4], attrbutes[5].split('.')[0]
        rot = x+'_'+y+'_'+z
        return {'feature': feature,
                'label': label,
                'class_name': class_name,
                'id': id,
                'rot': rot,
                'view': view}

class SelectedDataset:
    def __init__(self, features, labels, batch_size):
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle()
    
    def shuffle(self):
        combined = list(zip(self.features, self.labels))
        random.shuffle(combined)
        self.features, self.labels = zip(*combined)
    
    def __len__(self):
        return len(self.features)
    
    def __iter__(self):
        for i in range(0, len(self.features), self.batch_size):
            batch_features = self.features[i:i + self.batch_size]
            batch_labels = self.labels[i:i + self.batch_size]
            yield {'feature': batch_features, 'label': batch_labels}

def plot_and_save_acc(acc_dict, episodes, folder_path):
    # Extract episode numbers (sorted) and corresponding metrics
    top1_acc = [acc_dict[episode]['top-1'] for episode in range(episodes+1)]
    top5_acc = [acc_dict[episode]['top-5'] for episode in range(episodes+1)]
    # Create the plot
    episode_list = list(range(1, episodes + 2))
    plt.figure(figsize=(10, 6))
    plt.plot(episode_list, top1_acc, marker='o', label='Top-1 Accuracy')
    plt.plot(episode_list, top5_acc, marker='o', label='Top-5 Accuracy')

    # Customize the plot
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Epoch-wise Top-1 and Top-5 Accuracy')
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(os.path.join(folder_path, 'epoch_accuracy.png'), dpi=300, bbox_inches='tight')

    with open(os.path.join(folder_path, 'acc.json'), "w", encoding="utf-8") as f:
        json.dump(acc_dict, f, ensure_ascii=False, indent=4)

def plot_view_selcetion(view_selection, episodes, folder_path, split='train'):

    list_1p = []
    list_2p = []
    list_3p = []
    list_non_p = []

    for episode in range(episodes+1):
        num_1p = 0
        num_2p = 0
        num_3p = 0
        num_non_p = 0
        for i, class_name in enumerate(class_list):
            view_list = view_selection[str(episode)][class_name]
            for data in view_list:
                view = data['view']
                count_E = view.count('E')
                if count_E == 0:
                    num_3p+=1
                elif count_E == 1:
                    num_2p+=1
                elif count_E ==2:
                    num_1p +=1
                else:
                    num_non_p+=1
        list_1p.append(num_1p)
        list_2p.append(num_2p)
        list_3p.append(num_3p)
        list_non_p.append(num_non_p)

    episode_list = list(range(1, episodes + 2))
    plt.figure(figsize=(10, 6))
    plt.plot(episode_list, list_1p, marker='o', label='1p')
    plt.plot(episode_list, list_2p, marker='o', label='2p')
    plt.plot(episode_list, list_3p, marker='o', label='3p')
    plt.plot(episode_list, list_non_p, marker='o', label='non_p')

    # Add labels and title
    plt.xlabel('Episode')
    plt.ylabel('View numbers')
    plt.title('View Over Episodes')

    # Add a legend to distinguish between the metrics
    plt.legend()

    # Optional: add grid for better readability
    plt.grid(True)

    # Save the plot (optional)
    plt.savefig(os.path.join(folder_path, split+'_view_over_episode.png'), dpi=300, bbox_inches='tight')
    
def plot_view_distribution(view_selection, episode, folder_path, split='train'):

    x_list = []
    y_list = []
    z_list = []
    color_list = []

    for i, class_name in enumerate(class_list):
        view_list = view_selection[str(episode)][class_name]
        for data in view_list:
            rots = data['rot']
            rots = rots.split('_')
            view_type = data['view']
            x_list.append(float(rots[0]))
            y_list.append(float(rots[1]))
            z_list.append(float(rots[2]))
            count_E = view_type.count('E')
            if count_E == 0:
                color = (1.,0.,0.)
            elif count_E == 1:
                color = (0.,1.,0,)
            elif count_E ==2:
                color = (0.,0.,1.)
            else:
                color = (0.,0.,0.)
            color_list.append(color)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x_list, y_list, z_list, c=color_list, alpha=0.8)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    os.makedirs(os.path.join(folder_path, 'view_dist_'+split), exist_ok=True)
    plt.savefig(os.path.join(folder_path, 'view_dist_'+split, str(episode)+".png"), dpi=300)


def select_view(model, data_loader, episodes, folder_path, device, num_classes=10):
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
            views = data['view']

            features = features.to(device)
            scores = model(features)
            for b in range(len(features)):
                all_attributes.append({'class_name': class_names[b],
                                        'id': ids[b],
                                        'rot': rots[b],
                                        'view': views[b]})
                c = labels[b].item()
                all_scores[c].append(scores[b].cpu())
                class_indices[c].append(len(all_attributes) - 1)
        
        for c in range(num_classes):
            if len(all_scores[c]) > 0:
                all_scores[c] = torch.stack(all_scores[c])          # shape = [num_samples_in_class]
                all_scores[c] = F.gumbel_softmax(all_scores[c], tau=1.0, hard=False)

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
def train_rl_selection(train_feature_paths, test_feature_paths, args, num_classes=10):

    root_dir = 'results'
    os.makedirs(root_dir, exist_ok=True)  # Will create the folder if it doesn't exist
    exp_dir = 'rl_lr'+str(args.lr_rl) + '_cls_lr' + str(args.lr_cls) + '_episodes' + str(args.num_episode)+'_epochs'+str(args.epochs)+'_shot'+str(args.shot)
    stored_folder = os.path.join(root_dir,exp_dir)
    os.makedirs(stored_folder, exist_ok=True)
    store_every_episode = args.num_episode * 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score_network = ScoreNetwork(args.input_dim).to(device)
    score_optimizer = optim.Adam(score_network.parameters(), lr=args.lr_rl)
    
    
    criterion = nn.CrossEntropyLoss()
    
    train_sampled_paths = []
    train_sampled_labels = []
    test_sampled_paths = []
    test_sampled_labels = []

    for class_name, path_list in train_feature_paths.items():
        sampled_paths = path_list
        sampled_labels = [class_list.index(class_name)]*args.data_per_class
        train_sampled_paths += sampled_paths
        train_sampled_labels += sampled_labels
    train_dataset = FeatureDataset(train_sampled_paths, train_sampled_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=20, pin_memory=True)

    for class_name, path_list in test_feature_paths.items():
        sampled_paths = path_list
        sampled_labels = [class_list.index(class_name)]*args.data_per_class
        test_sampled_paths += sampled_paths
        test_sampled_labels += sampled_labels
    test_dataset = FeatureDataset(test_sampled_paths, test_sampled_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=20, pin_memory=True)
    



    best_episode = 0
    episode_acc = {'best_top-1': 0.0, 'best_top-5': 0.0, 'worst_top-1': float("inf"), 'worst_top-5': float("inf")}

    train_view_selection = {}
    test_view_selection = {}
    baseline_reward = 0  # Running baseline reward for variance reduction
    for episode in range(args.num_episode):
        is_best = False
        # Assign scores to all data
        score_network.train()
        all_features, all_labels, all_attributes = [], [], []
        all_scores = {class_id: [] for class_id in range(num_classes)}
        class_indices = {class_id: [] for class_id in range(num_classes)}
        # with torch.no_grad():
        for data in train_dataloader:
            features = data['feature']
            labels = data['label']
            class_names = data['class_name']
            ids = data['id']
            rots = data['rot']
            views = data['view']

            features = features.to(device)
            scores = score_network(features)
            for b in range(len(features)):
                # global append
                all_features.append(features[b])
                all_labels.append(labels[b].item())
                all_attributes.append({'class_name': class_names[b],
                                        'id': ids[b],
                                        'rot': rots[b],
                                        'view': views[b]})

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
                all_scores[c] = F.gumbel_softmax(all_scores[c], tau=1.0, hard=False)

        # Select top 5 based on scores per class
        selected_features = []
        selected_labels   = []
        selected_attributes = []
        selected_local_indices = []  # to help with policy update
        train_view_selection[str(episode)] = {}
        for c in range(num_classes):
            dist_c = all_scores[c]  # shape [Nc]
            if dist_c.nelement() == 0:
                continue
            # sample, e.g., 5
            sampled_local_idxs = torch.multinomial(dist_c, args.shot, replacement=False)
            # sampled_local_idxs = dist_c.topk(args.shot, dim=0).indices
            # convert those local indices to global
            global_idxs = [class_indices[c][loc_id.item()] for loc_id in sampled_local_idxs]
            train_view_selection[str(episode)][class_list[c]] = []

            for g in global_idxs:
                selected_features.append(all_features[g])
                selected_labels.append(c)
                selected_attributes.append(all_attributes[g])
                # if episode % store_every_episode == 0 or episode==args.num_episode-1:
                train_view_selection[str(episode)][class_list[c]].append(all_attributes[g])
            # We also store the local indices so we can do log_prob later
            # (because log_prob = log( all_scores[c][loc_id] ))
            selected_local_indices.extend(
                [(c, loc_id.item()) for loc_id in sampled_local_idxs]
            )

        # Train classifier on selected data
        classifier = ComplexClassifier(args.input_dim, num_classes).to(device)
        classifier_optimizer = optim.Adam(classifier.parameters(), lr=args.lr_cls, weight_decay=args.wd_cls)
        # selected_dataset = SelectedDataset(copy.deepcopy(selected_features), copy.deepcopy(selected_labels))
        # selected_dataloader = DataLoader(selected_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=False)
        selected_dataloader = SelectedDataset(selected_features, selected_labels, batch_size=args.batch_size)
        
        episode_acc[episode] = {'top-1': 0.0, 'top-5': 0.0}
        for epoch in range(args.epochs):
            classifier.train()
            epoch_loss = 0.0
            selected_dataloader.shuffle()
            all_outputs = []
            all_labels = []
            for data in selected_dataloader:
                features, targets = data['feature'], data['label']
                features, targets = torch.stack(features).to(device), torch.tensor(targets).to(device)
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
            episode_acc[episode] = {'top-1': top1_accuracy, 'top-5': top5_accuracy}
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
        reward = top1_accuracy + 0.1 * top5_accuracy

        # ===============
        # Policy Gradient Update
        # ===============
        # We want to maximize reward => minimize -reward =>
        # typical REINFORCE: L = - sum( log pi(a) * R )
        score_optimizer.zero_grad()

        probs = []
        for k, v in all_scores.items():
            probs.append(v)
        # for (class_id, local_id) in selected_local_indices:
        #     prob = all_scores[class_id][local_id]  # the softmax prob
        #     probs.append(prob)

        probs = torch.stack(probs)
        log_probs = torch.log(probs)
        entropy = -(probs * log_probs).sum()

        baseline_reward = 0.9 * baseline_reward + 0.1 * reward if episode > 0 else reward
        advantage = reward - baseline_reward + 1e-8
        # loss = -(log_probs * advantage).mean() - 0.01 * entropy
        loss = -(log_probs * advantage).mean() - 0.01 * entropy
        print(f"Episode {episode + 1}: Top-1 = {episode_acc[episode]['top-1']:.4f} Top-5 = {episode_acc[episode]['top-5']:.4f} loss = {loss.item():.4f}")
        loss.backward()
        score_optimizer.step()

        score_network.eval()

        plot_and_save_acc(episode_acc, episode, stored_folder)
        test_view_selection[str(episode)] = select_view(score_network, test_dataloader, episode, stored_folder, device, num_classes=10)
        plot_view_selcetion(train_view_selection, episode, stored_folder, 'train')
        plot_view_selcetion(test_view_selection, episode, stored_folder, 'test')
        plot_view_distribution(test_view_selection, episode, stored_folder)
        with open(os.path.join(stored_folder,'train_view.json'), 'w', encoding='utf-8') as f:
            json.dump(train_view_selection, f, ensure_ascii=False, indent=4)
        with open(os.path.join(stored_folder,'test_view.json'), 'w', encoding='utf-8') as f:
            json.dump(test_view_selection, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Five-Shot Training")
    parser.add_argument("--data_folder", type=str, default='/N/project/ego4d_vlm/ShapeNet/feature', help="Path to saved features")
    parser.add_argument("--num_episode", type=int, default=3000)
    parser.add_argument("--input_dim", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=20, help="Number of examples per class in each run")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--lr_rl", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--lr_cls", type=float, default=7e-5, help="Learning rate")
    parser.add_argument("--wd_cls", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--view", type=str, default='random')
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--data_per_class", type=int, default=500)
    
    args = parser.parse_args()
    
    train_feature_paths = load_feature_paths(args.data_folder, 'train', args.view)
    test_feature_paths = load_feature_paths(args.data_folder, 'test', args.view)

    train_rl_selection(train_feature_paths, test_feature_paths, args)
