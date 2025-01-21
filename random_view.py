import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import numpy as np
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
def load_feature_paths(data_folder):
    feature_paths = []
    labels = []
    class_to_idx = {}
    class_idx = 0
    
    for root, _, files in os.walk(data_folder):
        if files:
            class_name = os.path.basename(root)
            if class_name not in class_to_idx:
                class_to_idx[class_name] = class_idx
                class_idx += 1
            
            for file in files:
                if file.endswith(".pt"):
                    feature_paths.append(os.path.join(root, file))
                    labels.append(class_to_idx[class_name])
    
    return feature_paths, labels, len(class_to_idx)

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
def train_five_shot(feature_paths, labels, num_classes, input_dim, num_runs=100, batch_size=5, epochs=10, lr=0.01):
    top1_results = []
    top5_results = []
    
    for run in range(num_runs):
        sampled_indices = random.sample(range(len(feature_paths)), batch_size * num_classes)
        sampled_paths = [feature_paths[i] for i in sampled_indices]
        sampled_labels = [labels[i] for i in sampled_indices]
        
        dataset = FeatureDataset(sampled_paths, sampled_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model = ComplexClassifier(input_dim, num_classes).to("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            for features, targets in dataloader:
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
            for features, targets in dataloader:
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
    avg_top5 = np.mean(top5_results)
    
    print(f"Average Top-1 Accuracy: {avg_top1:.4f}")
    print(f"Average Top-5 Accuracy: {avg_top5:.4f}")

if __name__ == "__main__":
    data_folder = "path/to/saved/features"
    feature_paths, labels, num_classes = load_feature_paths(data_folder)
    input_dim = torch.load(feature_paths[0]).numel()
    train_five_shot(feature_paths, labels, num_classes, input_dim)
