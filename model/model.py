import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

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


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def initialize_fixed_model(seed=42, feature=True, pretrained=True, early_exit=False):
    set_seed(seed)  # Ensure deterministic behavior for this model
    if feature:
        fixed_model = FeatureClassifier()
    else:
        fixed_model = ImageClassifier(pretrained=pretrained, early_exit=early_exit)
    return fixed_model

class ScoreRes(nn.Module):
    def __init__(self, input_dim, rot=False, pretrained=True):
        super(ScoreRes, self).__init__()
        model = models.resnet18(pretrained=pretrained)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        self.resnet = model
        self.fc1 = nn.Linear(512, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 1)


    def forward(self, x, rot=None):
        x = self.resnet(x).squeeze()
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        return x.squeeze()

class ScoreNetwork(nn.Module):
    def __init__(self, input_dim, rot=False, pretrained=True):
        super(ScoreNetwork, self).__init__()
        # First 1x1 convolution to reduce dimensionality (2048 -> 1024)
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(1024)
        
        # 3x3 convolution for feature extraction
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(1024)
        
        # Second 1x1 convolution (1024 -> 512)
        self.conv3 = nn.Conv2d(1024, 512, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(512)

        # Global Average Pooling (GAP)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        if rot:
            self.fc1_rot = nn.Linear(3, 64)
            self.bn1_rot = nn.BatchNorm1d(64)
            self.fc2_rot = nn.Linear(64, 128)
            self.bn2_rot = nn.BatchNorm1d(128)
            self.fc3_rot = nn.Linear(128, 256)
            self.bn3_rot = nn.BatchNorm1d(256)
            self.fc = nn.Linear(768, 1)
        # Fully Connected Layer (512 -> 1)
        else:
            self.fc = nn.Linear(512, 1)

    def forward(self, x, rot=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)

        x = self.global_avg_pool(x)  # Shape: [batch_size, 512, 1, 1]
        x = torch.flatten(x, 1)      # Shape: [batch_size, 512]
        
        if rot != None:
            rot = self.fc1_rot(rot)
            rot = self.bn1_rot(rot)
            rot = torch.relu(rot)
            rot = self.fc2_rot(rot)
            rot = self.bn2_rot(rot)
            rot = torch.relu(rot)
            rot = self.fc3_rot(rot)
            rot = self.bn3_rot(rot)
            rot = torch.relu(rot)
            x = torch.cat((x,rot), dim=-1)
        # Fully Connected Output
        x = self.fc(x)  # Shape: [batch_size, 1]


        return x.squeeze()

class ImageClassifier(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, early_exit=False):
        super(ImageClassifier, self).__init__()
        self.early_exit = early_exit
        model = models.resnet18(pretrained=pretrained)
        if not self.early_exit:
            model = torch.nn.Sequential(*list(model.children())[:-1])
            self.resnet = model
            self.fc1 = nn.Linear(512, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.fc2 = nn.Linear(128, num_classes)
        else:
            model = torch.nn.Sequential(*list(model.children())[:-5])
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc1 = nn.Linear(64, 32)
            self.bn1 = nn.BatchNorm1d(32)
            self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x, rot=None):
        if not self.early_exit:
            x = self.resnet(x).squeeze()
            x = self.fc1(x)
            x = self.bn1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            x = x.squeeze()
        else:
            x = self.resnet(x)
            x = self.global_avg_pool(x)
            x = x.squeeze()
            x = self.fc1(x)
            x = self.bn1(x)
            x = torch.relu(x)
            x = self.fc2(x)

        return x

class FeatureClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(FeatureClassifier, self).__init__()
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(1024)
        
        # 3x3 convolution for feature extraction
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(1024)
        
        # Second 1x1 convolution (1024 -> 512)
        self.conv3 = nn.Conv2d(1024, 512, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(512)

        # Global Average Pooling (GAP)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layer (512 -> 1)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)

        x = self.global_avg_pool(x)  # Shape: [batch_size, 512, 1, 1]
        x = torch.flatten(x, 1)      # Shape: [batch_size, 512]
        
        # Fully Connected Output
        x = self.fc(x)  # Shape: [batch_size, 1]

        return x.squeeze()