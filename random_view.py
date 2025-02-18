import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import numpy as np
import argparse
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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
                    if split == 'train':
                        if view == 'E' and 'E' not in file:
                            continue
                        if view == 'EE' and 'EE' not in file:
                            continue
                        if view == 'EEE' and 'EEE' not in file:
                            continue
                    if file.endswith(".pt"):
                        feature_paths[class_name].append(os.path.join(instances_path, file))
    
    return feature_paths

def load_feature_paths_with_view(data_folder, split='train', view='random'):
    feature_paths_non_planar = {}
    feature_paths_planar = {}
    class_list = ['airplane', 'bathtub', 'bed', 'bin', 'bottle', 'bowl', 'bus', 'can', 'case', 'hat']
    for class_idx, class_name in enumerate(class_list):
        folder_path = os.path.join(data_folder, class_name, split)
        instances = os.listdir(folder_path)
        feature_paths_non_planar[class_name] = []
        feature_paths_planar[class_name] = []
        for ins in instances:
            instances_path = os.path.join(folder_path, ins, 'screenshot')
            for _, _, files in os.walk(instances_path):
                for file in files:
                    if file.endswith(".pt"):
                        if not 'EEE' in file:
                            feature_paths_planar[class_name].append(os.path.join(instances_path, file))
                        else:
                            feature_paths_non_planar[class_name].append(os.path.join(instances_path, file))
    
    return [feature_paths_non_planar, feature_paths_planar]

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

# Training function
def train_few_shot(train_feature_paths, \
                    test_feature_paths, \
                    args):
    num_classes = 10
    batch_size = args.batch_size
    input_dim = args.input_dim
    epochs = args.epochs
    num_runs = args.num_runs
    shot = args.shot
    data_per_class = args.data_per_class
    lr = args.lr
    top1_results = []
    top5_results = []
    
    class_list = ['airplane', 'bathtub', 'bed', 'bin', 'bottle', 'bowl', 'bus', 'can', 'case', 'hat']
    train_sampled_paths = []
    train_sampled_labels = []
    test_sampled_paths = []
    test_sampled_labels = []
    for run in range(num_runs):
        random.seed(None)  # Ensures randomness across runs
        if args.view == 'random':
            for class_name, path_list in train_feature_paths.items():
                random.seed(None)
                sampled_indices = random.sample(range(len(path_list)), shot)
                sampled_paths = [path_list[i] for i in sampled_indices]
                sampled_labels = [class_list.index(class_name)]*shot
                train_sampled_paths += sampled_paths
                train_sampled_labels += sampled_labels
        else:
            feature_paths_non_planar, feature_paths_planar = train_feature_paths[0], train_feature_paths[1]
            num_non_planar = int((1.0-args.view_ratio)*shot)
            num_planar = int(args.view_ratio*shot)
            for class_name, path_list in feature_paths_non_planar.items():
                random.seed(None)
                sampled_indices = random.sample(range(len(path_list)), num_non_planar)
                sampled_paths = [path_list[i] for i in sampled_indices]
                sampled_labels = [class_list.index(class_name)]*num_non_planar
                train_sampled_paths += sampled_paths
                train_sampled_labels += sampled_labels
            for class_name, path_list in feature_paths_planar.items():
                sampled_indices = random.sample(range(len(path_list)), num_planar)
                sampled_paths = [path_list[i] for i in sampled_indices]
                sampled_labels = [class_list.index(class_name)]*num_planar
                train_sampled_paths += sampled_paths
                train_sampled_labels += sampled_labels

        train_dataset = FeatureDataset(train_sampled_paths, train_sampled_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)

        for class_name, path_list in test_feature_paths.items():
            sampled_paths = path_list
            sampled_labels = [class_list.index(class_name)]*data_per_class
            test_sampled_paths += sampled_paths
            test_sampled_labels += sampled_labels
        test_dataset = FeatureDataset(test_sampled_paths, test_sampled_labels)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
        
        model = ComplexClassifier(input_dim, num_classes).to("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            for features, targets in train_dataloader:
                features, targets = features.to("cuda" if torch.cuda.is_available() else "cpu"), targets.to("cuda" if torch.cuda.is_available() else "cpu")
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        # Evaluation
        model.eval()
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for features, targets in test_dataloader:
                features = features.to("cuda" if torch.cuda.is_available() else "cpu")
                outputs = model(features)
                all_outputs.append(outputs.cpu())
                all_labels.append(targets.cpu())
        
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        top1_correct = (all_outputs.argmax(dim=1) == all_labels).sum().item()
        top5_correct = sum([all_labels[i] in all_outputs[i].topk(5).indices for i in range(len(all_labels))])
        
        top1_results.append(top1_correct / len(all_labels))
        top5_results.append(top5_correct / len(all_labels))
    
    avg_top1 = np.mean(top1_results)
    std_top1 = np.std(top1_results)
    avg_top5 = np.mean(top5_results)
    std_top5 = np.std(top5_results)
    
    print(f"Average Top-1 Accuracy: {avg_top1:.4f} ± {std_top1:.4f}")
    print(f"Average Top-5 Accuracy: {avg_top5:.4f} ± {std_top5:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Five-Shot Training")
    parser.add_argument("--data_folder", type=str, default='/N/project/ego4d_vlm/ShapeNet/feature', help="Path to saved features")
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--input_dim", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=10, help="Number of examples per class in each run")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--view", type=str, default='random')
    parser.add_argument("--view_ratio", type=float, default=0.5)
    parser.add_argument("--shot", type=int, default=10)
    parser.add_argument("--data_per_class", type=int, default=500)
    
    args = parser.parse_args()
    if args.view == 'random':
        train_feature_paths = load_feature_paths(args.data_folder, 'train', args.view)
    else:
        train_feature_paths = load_feature_paths_with_view(args.data_folder, 'train', args.view)
    test_feature_paths = load_feature_paths(args.data_folder, 'test', args.view)
    train_few_shot(train_feature_paths, test_feature_paths, args)
