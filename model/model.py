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

def initialize_fixed_model(seed=42, feature=True, pretrained=True, depth=False, classifier=False):
    set_seed(seed)  # Ensure deterministic behavior for this model
    if feature:
        if depth:
            fixed_model = FeatureClassifierDepth()
        else:
            fixed_model = FeatureClassifier()
    else:
        if depth:
            fixed_model = ImageClassifierandDepth(pretrained=pretrained)
        else:
            fixed_model = ImageClassifier(pretrained=pretrained)
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
    def __init__(self, input_dim, s_input='r50'):
        super(ScoreNetwork, self).__init__()
        # First 1x1 convolution to reduce dimensionality (2048 -> 1024)
        if s_input =='r50':
            self.conv1 = nn.Conv2d(2048, 1024, kernel_size=1)
            self.bn1 = nn.BatchNorm2d(1024)
            # 3x3 convolution for feature extraction
            self.conv2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(1024)
            # Second 1x1 convolution (1024 -> 512)
            self.conv3 = nn.Conv2d(1024, 512, kernel_size=1)
            self.bn3 = nn.BatchNorm2d(512)            
            self.fc = nn.Linear(512, 1)

        elif s_input == 'r18_1conv':
            self.conv1 = nn.Conv2d(64, 32, kernel_size=1)
            self.bn1 = nn.BatchNorm2d(32)
            # 3x3 convolution for feature extraction
            self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            # Second 1x1 convolution (1024 -> 512)
            self.conv3 = nn.Conv2d(32, 16, kernel_size=1)
            self.bn3 = nn.BatchNorm2d(16)
            self.fc = nn.Linear(16, 1)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

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

class ImageClassifier(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(ImageClassifier, self).__init__()
        #self.early_exit = early_exit
        model = models.resnet18(pretrained=pretrained)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        self.resnet = model
        self.fc1 = nn.Linear(512, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.resnet(x).squeeze()
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = x.squeeze()
        return x

class ImageClassifierDepth(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(ImageClassifierDepth, self).__init__()
        #self.early_exit = early_exit
        model = models.resnet18(pretrained=pretrained)
        model = torch.nn.Sequential(*list(model.children())[:-2])
        self.resnet = model
        self.fc1 = nn.Linear(512, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.depth_decoder = DepthDecoder(512)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        depth = self.depth_decoder(x)
        depth = self.sigmoid(depth)
        x = self.global_avg_pool(x)
        x = x.squeeze()
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = x.squeeze()

        return x, depth

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

class FeatureClassifierDepth(nn.Module):
    def __init__(self, num_classes=10):
        super(FeatureClassifierDepth, self).__init__()
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
    
        self.depth_decoder = DepthDecoder(512)
        self.sigmoid = nn.Sigmoid()

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

        depth = self.depth_decoder(x)
        depth = self.sigmoid(depth)
        x = self.global_avg_pool(x)  # Shape: [batch_size, 512, 1, 1]
        x = torch.flatten(x, 1)      # Shape: [batch_size, 512]
        
        # Fully Connected Output
        x = self.fc(x)  # Shape: [batch_size, 1]

        return x.squeeze(), depth


class DepthDecoder(nn.Module):
    def __init__(self, in_channels, out_size=(224, 224)):
        """
        Args:
            in_channels (int): Number of channels from the encoder (e.g., 512 for ResNet-18).
            out_size (tuple): Desired spatial output size (H, W) of the final depth map.
        """
        super(DepthDecoder, self).__init__()
        self.out_size = out_size

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, kernel_size=1),  # output depth map (1 channel)
        )

    def forward(self, x):
        x = self.decoder(x)
        x = F.interpolate(x, size=self.out_size, mode='bilinear', align_corners=False)
        return x  # shape: (B, 1, H, W)
