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
                    if view == 'planar' and 'E' not in file:
                        continue
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
        return feature, label

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
    for class_name, path_list in train_feature_paths.items():
        sampled_indices = random.sample(range(len(path_list)), args.shot)
        sampled_paths = [path_list[i] for i in sampled_indices]
        sampled_labels = [class_list.index(class_name)]*args.shot
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
    score_network.eval()
    for episode in range(args.num_episode):
        # Assign scores to all data
        all_features, all_labels = [], []
        all_scores = {class_id: [] for class_id in range(num_classes)}
        with torch.no_grad():
            for features, targets in train_dataloader:
                features = features.to(device)
                scores = score_network(features)
                for i, label in enumerate(targets):
                    all_features.append(features[i].cpu())
                    all_labels.append(label.cpu())
                    all_scores[label.item()].append(scores[i].cpu())
        
        # Apply softmax across the class axis for each sample
        for class_id in all_scores:
            if all_scores[class_id]:
                all_scores[class_id] = torch.stack(all_scores[class_id])
                all_scores[class_id] = torch.softmax(all_scores[class_id], dim=1)  # Normalize across class axis
                print(all_scores[class_id].shape)
        # Select top 5 based on scores per class
        selected_features, selected_labels = [], []
        for class_id in range(num_classes):
            class_indices = [i for i, label in enumerate(all_labels) if label.item() == class_id]
            if class_indices and len(all_scores[class_id]) > 0:
                class_scores = all_scores[class_id][:, class_id]
                # top_indices = torch.argsort(class_scores, descending=True)[:batch_size]
                sampled_indices = torch.multinomial(class_scores, args.shot, replacement=False)
                class_samples = [all_features[i] for i in sampled_indices]
                selected_features.extend(class_samples)
                selected_labels.extend([class_id] * len(class_samples))

                # selected_features.extend([all_features[class_indices[i]] for i in top_indices])
                # selected_labels.extend([all_labels[class_indices[i]] for i in top_indices])

        # Train classifier on selected data
        classifier.train()
        for epoch in range(args.epochs):
            # Generate a random permutation of indices
            indices = list(range(len(selected_features)))
            random.shuffle(selected_features)
            # Apply the same permutation to both lists
            selected_features = [selected_features[i] for i in indices]
            selected_labels = [selected_labels[i] for i in indices]

            classifier_optimizer.zero_grad()
            outputs = classifier(torch.stack(selected_features).to(device))
            loss = criterion(outputs, torch.tensor(selected_labels).to(device))
            loss.backward()
            classifier_optimizer.step()
        
        # Evaluate classifier
        classifier.eval()
        correct = 0
        with torch.no_grad():
            for features, targets in test_dataloader:
                features, targets = features.to(device), targets.to(device)
                outputs = classifier(features)
                correct += (outputs.argmax(dim=1) == targets).sum().item()
        
        accuracy = correct / len(all_labels)
        reward = accuracy

         # ===============
        # Policy Gradient Update
        # ===============
        # We want to maximize reward => minimize -reward =>
        # typical REINFORCE: L = - sum( log pi(a) * R )
        score_network.train()
        score_optimizer.zero_grad()

        log_probs = torch.log(torch.stack([all_scores[label][i][label] for i, label in enumerate(selected_labels)]))
        loss = -log_probs.mean() * reward  # Reinforce loss
        loss.backward()
        score_optimizer.step()
        
        print(f"Episode {episode + 1}: Accuracy = {accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Five-Shot Training")
    parser.add_argument("--data_folder", type=str, default='/N/project/ego4d_vlm/ShapeNet/feature', help="Path to saved features")
    parser.add_argument("--num_episode", type=int, default=1000)
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
