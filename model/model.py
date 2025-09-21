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

from monst3r.model import build_model
from monst3r.utils import load_ckpt 

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

def initialize_fixed_edge_model(seed, depth, feature):
    set_seed(seed)  # Ensure deterministic behavior for this model
    if not feature:
        fixed_model = EdgeClassifier(depth)
    else:
        fixed_model = FeatureClassifierDepth(depth=depth)
    return fixed_model

def initialize_fixed_model(seed=42, feature=True, pretrained=True, depth=False, classifier=False, rep=''):
    set_seed(seed)  # Ensure deterministic behavior for this model
    if feature:
        if rep == 'monst3r':
            fixed_model = MonST3RWithMLP(encoder_ckpt_path="checkpoints/monst3r_vitb16.pth")
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

def initialize_fixed_set_model(seed):
    set_seed(seed)  # Ensure deterministic behavior for this model
    fixed_model = SetFeatureClassifier()
    return fixed_model


class MonST3RWithMLP(nn.Module):
    def __init__(self, encoder_ckpt_path, encoder_arch="ViT-B/16", hidden_dim=512, num_classes=10):
        super().__init__()
        self.encoder = build_model(encoder_arch, pretrained=False)
        load_ckpt(self.encoder, encoder_ckpt_path)
        for param in self.encoder.parameters():
            param.requires_grad = False  # freeze encoder if desired

        self.encoder = self.encoder.to(DEVICE)
        self.encoder.eval()

        # Classifier MLP (MonST3R ViT-B output dim is 768)
        self.classifier = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, views):  # views: (B, N_views, C, H, W)
        B, N, C, H, W = views.shape
        views = views.view(B * N, C, H, W)
        with torch.no_grad():
            feats = self.encoder.encode_image(views)  # (B * N, D)
        feats = feats.view(B, N, -1)
        pooled_feats = feats.mean(dim=1)  # average pooling over views
        return self.classifier(pooled_feats)


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

class ScoreSetNetwork(nn.Module):
    def __init__(self, input_dim, s_input='r50'):
        super(ScoreSetNetwork, self).__init__()
        # First 1x1 convolution to reduce dimensionality (2048 -> 1024)
        if s_input =='r50':
            self.fc1 = nn.Linear(2048, 512)
            self.bn1 = nn.BatchNorm1d(512)
            self.fc2 = nn.Linear(512, 64)
            self.bn2 = nn.BatchNorm1d(64)
            self.fc3 = nn.Linear(64, 8)
            self.bn3 = nn.BatchNorm1d(8)

            self.agg_fc1 = nn.Linear(800, 200)
            # self.agg_bn1 = nn.BatchNorm1d(200)
            self.agg_fc2 = nn.Linear(200, 64)
            # self.agg_bn2 = nn.BatchNorm1d(64)
            self.agg_fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # [1, 100, 2048]
        b, n, c = x.shape
        x = x.reshape(b*n, c)
        # [100, 2048]
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        # [1, 100, 8]
        x = x.reshape(1, -1)
        # [1, 800]
        x = self.agg_fc1(x) 
        # x = self.agg_bn1(x)
        x = torch.relu(x)
        x = self.agg_fc2(x)
        # x = self.agg_bn2(x)
        x = torch.relu(x)
        x = self.agg_fc3(x)
        return x.squeeze(dim=0)

class EdgeScoreNetwork(nn.Module):
    def __init__(self, input_dim, s_input='r50'):
        super(EdgeScoreNetwork, self).__init__()
        # First 1x1 convolution to reduce dimensionality (2048 -> 1024)
        # resnet18 = models.resnet18(pretrained=False)

        # # Modify the first conv layer to take 1 input channel (Canny edges)
        # resnet18.conv1 = nn.Conv2d(
        #     in_channels=1,             # single-channel input
        #     out_channels=64,
        #     kernel_size=7,
        #     stride=2,
        #     padding=3,
        #     bias=False
        # )
        # resnet18.fc = nn.Linear(resnet18.fc.in_features, 1)
        # self.model = resnet18

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # input: (1, H, W)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (16, H/2, W/2)
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (32, H/4, W/4)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> (64, 1, 1)
        )

        self.classifier = nn.Linear(64, 1)  # Output: dim=1


    def forward(self, x):
        # x = self.model(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
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

class SetFeatureClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(SetFeatureClassifier, self).__init__()

        # Fully Connected Layer (512 -> 1)
        self.fc1 = nn.Linear(2048, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Fully Connected Output
        x = self.fc1(x)  # Shape: [batch_size, 1]
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

class FeatureClassifierDepth(nn.Module):
    def __init__(self, num_classes=10, depth=False):
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

class EdgeClassifier(nn.Module):
    def __init__(self, s_input='r50', depth=False):
        super(EdgeClassifier, self).__init__()
        self.depth = depth
        if depth:
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # input: (1, H, W)
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),  # -> (16, H/2, W/2)
                
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),  # -> (32, H/4, W/4)
                
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
            self.decoder = DepthDecoder(64, out_size=(128, 128))
            self.pool = nn.AdaptiveAvgPool2d((1, 1)),  # -> (64, 1, 1)
        else:
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # input: (1, H, W)
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),  # -> (16, H/2, W/2)
                
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),  # -> (32, H/4, W/4)
                
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),  # -> (64, 1, 1)
            )
        self.classifier = nn.Linear(64, 10)  # Output: dim=1

    def forward(self, x):
        # x = self.model(x)
        if not self.depth:
            x = self.features(x)
            x = x.view(x.size(0), -1)  # flatten
            x = self.classifier(x)
            depth = None
        else:
            x = self.features(x)
            depth = self.depth_decoder(x)
            depth = self.sigmoid(depth)
            x = self.global_avg_pool(x)  # Shape: [batch_size, 512, 1, 1]
            x = torch.flatten(x, 1)      # Shape: [batch_size, 512]
            x = self.classifier(x)  # Shape: [batch_size, 1]
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
