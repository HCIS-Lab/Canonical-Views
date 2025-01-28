import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import numpy as np
import argparse
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Define a scoring network
class ScoreNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ScoreNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x.squeeze()

# Define a more complex classification model
class ComplexClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ComplexClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
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
        return feature, label, os.path.basename(self.feature_paths[idx])

# Reinforcement learning training loop
def train_rl_selection(train_feature_paths, test_feature_paths, args, num_classes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score_network = ScoreNetwork(args.input_dim).to(device)
    classifier = ComplexClassifier(args.input_dim, num_classes).to(device)
    
    score_optimizer = optim.Adam(score_network.parameters(), lr=args.lr_rl)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=args.lr_cls)
    
    criterion = nn.CrossEntropyLoss()
    
    class_list = ['airplane', 'bathtub', 'bed', 'bin', 'bottle', 'bowl', 'bus', 'can', 'case', 'hat']
    train_sampled_paths = []
    train_sampled_labels = []
    test_sampled_paths = []
    test_sampled_labels = []
    # for class_name, path_list in train_feature_paths.items():
    #     sampled_indices = random.sample(range(len(path_list)), args.shot)
    #     sampled_paths = [path_list[i] for i in sampled_indices]
    #     sampled_labels = [class_list.index(class_name)]*args.shot
    #     train_sampled_paths += sampled_paths
    #     train_sampled_labels += sampled_labels
    # train_dataset = FeatureDataset(train_sampled_paths, train_sampled_labels)
    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)

    for class_name, path_list in train_feature_paths.items():
        sampled_paths = path_list
        sampled_labels = [class_list.index(class_name)]*args.data_per_class
        train_sampled_paths += sampled_paths
        train_sampled_labels += sampled_labels
    train_dataset = FeatureDataset(train_sampled_paths, train_sampled_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)

    for class_name, path_list in test_feature_paths.items():
        sampled_paths = path_list
        sampled_labels = [class_list.index(class_name)]*args.data_per_class
        test_sampled_paths += sampled_paths
        test_sampled_labels += sampled_labels
    test_dataset = FeatureDataset(test_sampled_paths, test_sampled_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    
    best_t1 = 0
    best_t5 = 0
    best_view = []
    for episode in range(args.num_episode):
        # Assign scores to all data
        all_features, all_labels, all_names = [], [], []
        all_scores = {class_id: [] for class_id in range(num_classes)}
        class_indices = {class_id: [] for class_id in range(num_classes)}
        # with torch.no_grad():
        for features, labels, names in train_dataloader:
            features = features.to(device)
            scores = score_network(features)
            for b in range(len(features)):
                # global append
                all_features.append(features[b])
                all_labels.append(labels[b].item())
                all_names.append(names[b])

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
                all_scores[c] = torch.softmax(all_scores[c], dim=0) # same shape

        # Select top 5 based on scores per class
        selected_features = []
        selected_labels   = []
        selected_names    = []
        selected_local_indices = []  # to help with policy update
        for c in range(num_classes):
            dist_c = all_scores[c]  # shape [Nc]
            if dist_c.nelement() == 0:
                continue
            # sample, e.g., 5
            sampled_local_idxs = torch.multinomial(dist_c, args.shot, replacement=False)
            # convert those local indices to global
            global_idxs = [class_indices[c][loc_id.item()] for loc_id in sampled_local_idxs]
            
            for g in global_idxs:
                selected_features.append(all_features[g])
                selected_labels.append(c)
                selected_names.append(all_names[g])
            # We also store the local indices so we can do log_prob later
            # (because log_prob = log( all_scores[c][loc_id] ))
            selected_local_indices.extend(
                [(c, loc_id.item()) for loc_id in sampled_local_idxs]
            )
        # for class_id in range(num_classes):
        #     class_indices = [i for i, label in enumerate(all_labels) if label.item() == class_id]
        #     if class_indices and len(all_scores[class_id]) > 0:
        #         class_scores = all_scores[class_id]
        #         sampled_indices = torch.multinomial(class_scores, args.shot, replacement=False)
        #         class_samples = [all_features[i] for i in sampled_indices]
        #         selected_features.extend(class_samples)
        #         selected_labels.extend([class_id] * len(class_samples))

        # Train classifier on selected data
        classifier.train()
        for epoch in range(args.epochs):
            indices = list(range(len(selected_features)))
            random.shuffle(indices)

            selected_features = [selected_features[i] for i in indices]
            selected_labels   = [selected_labels[i]   for i in indices]

            classifier_optimizer.zero_grad()
            outputs = classifier(torch.stack(selected_features).to(device))
            loss = criterion(outputs, torch.tensor(selected_labels).to(device))
            loss.backward()
            classifier_optimizer.step()
        
        # Evaluate classifier
        classifier.eval()
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for features, targets, _ in test_dataloader:
                features, targets = features.to(device), targets.to(device)
                outputs = classifier(features)
                all_outputs.append(outputs.cpu())
                all_labels.append(targets.cpu())
                # correct += (outputs.argmax(dim=1) == targets).sum().item()
        top1_correct = (all_outputs.argmax(dim=1) == all_labels).sum().item()
        top5_correct = sum([all_labels[i] in all_outputs[i].topk(5).indices for i in range(len(all_labels))])
        t1 = top1_correct / len(all_labels)
        t5 = top5_correct / len(all_labels)
        num_test_samples = len(test_dataset)  # or len(test_sampled_paths)
        accuracy = correct / num_test_samples
        reward = accuracy

        # ===============
        # Policy Gradient Update
        # ===============
        # We want to maximize reward => minimize -reward =>
        # typical REINFORCE: L = - sum( log pi(a) * R )
        score_network.train()
        score_optimizer.zero_grad()

        log_probs = []
        for (class_id, local_id) in selected_local_indices:
            prob = all_scores[class_id][local_id]  # the softmax prob
            log_probs.append(torch.log(prob))

        log_probs = torch.stack(log_probs)
        loss = - log_probs.mean() * reward  # basic REINFORCE
        loss.backward()
        score_optimizer.step()

        # log_probs = torch.log(torch.stack([all_scores[label][i] for i, label in enumerate(selected_labels)]))
        # log_probs = torch.log(torch.stack([all_scores[label][i] for i, label in enumerate(selected_labels)]))
        # loss = -log_probs.mean() * reward  # Reinforce loss
        # loss.backward()
        # score_optimizer.step()

        print(f"Episode {episode + 1}: Top-1 = {t1:.4f} Top-5 = {t5:.4f}")
        if accuracy > best_t1:
            best_t1 = accuracy
            best_t5 = t5
            best_view = selected_names

    print(f"Best: Top-1 = {t1:.4f} Top-5 = {t5:.4f}")
    print(best_view.sort())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Five-Shot Training")
    parser.add_argument("--data_folder", type=str, default='/N/project/ego4d_vlm/ShapeNet/feature', help="Path to saved features")
    parser.add_argument("--num_episode", type=int, default=2000)
    parser.add_argument("--input_dim", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=10, help="Number of examples per class in each run")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--lr_rl", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--lr_cls", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--view", type=str, default='random')
    parser.add_argument("--shot", type=int, default=10)
    parser.add_argument("--data_per_class", type=int, default=500)
    
    args = parser.parse_args()
    
    train_feature_paths = load_feature_paths(args.data_folder, 'train', args.view)
    test_feature_paths = load_feature_paths(args.data_folder, 'test', args.view)

    train_rl_selection(train_feature_paths, test_feature_paths, args)
