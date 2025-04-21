import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
import random
import numpy as np
import argparse
import dataset.dataset as dataset
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class_list = ['airplane', 'bathtub', 'bed', 'bin', 'bottle', 'bowl', 'bus', 'can', 'case', 'hat']
transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

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

def initialize_fixed_model(seed=50):
    set_seed(seed)  # Ensure deterministic behavior for this model
    fixed_model = ResNet18Classifier(10)
    return fixed_model

# ==== Rotation Policy with CNN + LSTM ====
class RotationPolicy(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, batch_first=True)
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_size, 3),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(1, 3))

    def forward(self, images, hidden):
        batch_size = images.size(0)
        x = self.cnn(images).view(batch_size, 1, -1)  # [B, 1, 64]
        lstm_out, hidden = self.lstm(x, hidden)
        h = lstm_out.squeeze(1)
        mu = self.mu_head(h)
        std = self.log_std.exp().expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()
        action = action.clamp(-1, 1)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, hidden, log_prob

# ==== Dummy Classifier (Replace with ResNet or other CNN) ====
from torchvision.models import resnet18

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = resnet18(pretrained=True)  # Load with ImageNet weights

        # Replace final fully-connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def random_view(image_dict):
    images = []
    rot = []
    for i, class_name in enumerate(class_list):
        all_images = image_dict[class_name]['image']
        random_idx = random.randint(0, len(all_images)-1)
        image = Image.open(all_images[random_idx]).convert('RGB')
        image = transform(image)
        images.append(image)
        rot.append(image_dict[class_name]['rot'][random_idx])
    # [10, 3, h, w], 
    return torch.stack(images), rot

def wrap_angle(x):
    if x > 1.0:
        x = -(2.0-x)
    elif x < -1.0:
        x = -(-2-x)
    return x

def angle_distance(a, b):
    return min(abs(a - b), 2 - abs(a - b))

def total_angle_distance(rot1, rot2):
    return sum(angle_distance(rot1[i], rot2[i]) for i in range(3))

def next_view(current_view, class_idx, image_dict, current_rot, action):
    new_rot = [wrap_angle(current_rot[i] + action[i].item()) for i in range(3)]

    class_name = class_list[class_idx]
    rot_list = image_dict[class_name]['rot']
    image_list = image_dict[class_name]['image']

    min_dist = float('inf')
    closest_view = None
    matched_rot = None
    for i, rot in enumerate(rot_list):
        dist = total_angle_distance(new_rot, rot)
        if dist < min_dist:
            min_dist = dist
            closest_view = image_list[i]
            matched_rot = rot

    current_view = Image.open(closest_view).convert('RGB')
    current_view = transform(current_view)
    return current_view, matched_rot

def train_action(train_dict, test_dict, args, num_classes=10):

    # ==== Main RL Loop ====
    policy = RotationPolicy().cuda()
    policy.train()
    policy_optim = optim.Adam(policy.parameters(), lr=args.lr_rl)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    policy_optim, mode='max', factor=0.5, patience=5
)
    baseline_reward = 0

    NUM_OBJECTS = 10
    NUM_STEPS = args.shot
    NUM_CLASSES = 10  # Change depending on task
    rot_episode = {}

    test_x = []
    test_y = []
    for i, class_name in enumerate(class_list):
        test_x = test_x + test_dict[class_name]['image']
        test_y = test_y + [torch.tensor(i, dtype=torch.int)]*args.data_per_class

    test_dataset = dataset.ActionDataset(test_x, test_y, test=True)
    test_dataset = DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=10, pin_memory=True)

    for episode in range(args.num_episode):
        # all_images = [[] for _ in range(NUM_OBJECTS)]
        all_images = []
        # all_rot = [[] for _ in range(NUM_OBJECTS)]
        all_rot = []
        hidden = (torch.zeros(1, NUM_OBJECTS, 128).cuda(), torch.zeros(1, NUM_OBJECTS, 128).cuda())
        current_views, current_rot = random_view(train_dict)
        for i in range(NUM_OBJECTS):
            all_images.append(current_views[i])
        current_views = current_views.cuda()
        all_actions = []  # Store actions over all steps

        log_probs = []
        for step in range(NUM_STEPS):
            actions, hidden, log_prob = policy(current_views, hidden)
            log_probs.append(log_prob)

            next_views = []
            next_rot = []
            for i in range(NUM_OBJECTS):
                delta_rot = (actions[i])
                new_image, new_rot = next_view(current_views[i], i, train_dict, current_rot[i], delta_rot)
                all_images.append(new_image)
                next_views.append(new_image)
                all_rot.append(new_rot)
                next_rot.append(new_rot)
            current_views = torch.stack(next_views).cuda()
            current_rot = next_rot
        rot_episode[episode] = all_rot

        # Train classifier on collected views
        classifier = initialize_fixed_model().cuda()
        clf_opt = optim.Adam(classifier.parameters(), lr=args.lr_cls, weight_decay=args.wd_cls)
        criterion = nn.CrossEntropyLoss()

        train_labels = []
        for i in range(NUM_OBJECTS):
            for j in range(args.shot):
                train_labels.append(float(i))
        combined = list(zip(all_images, train_labels))
        random.shuffle(combined)
        train_x, train_y = zip(*combined)
        train_dataset = dataset.ActionDataset(train_x, train_y)
        train_dataset = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=10, pin_memory=True)
        classifier.train()
        for i in range(args.epochs):  # few epochs
            for x, y in train_dataset:
                y = y.type(torch.LongTensor)
                x = x.cuda()
                y = y.cuda()
                logits = classifier(x)
                loss = criterion(logits, y)
                clf_opt.zero_grad()
                loss.backward()
                clf_opt.step()

        classifier.eval()
        with torch.no_grad():
            all_preds = []
            all_labels = []
            for x, y in train_dataset:
                x = x.cuda()
                y = y.cuda()
                logits = classifier(x)
                preds = logits.argmax(dim=1)
                all_preds.append(preds)
                all_labels.append(y)

            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            acc = (all_preds == all_labels).float().mean().item()
            print(f"Accuracy: {acc:.3f}")

        # Simulate reward as test accuracy (replace with real test set)
        classifier.eval()
        with torch.no_grad():
            all_preds = []
            all_labels = []
            for x, y in test_dataset:
                x = x.cuda()
                y = y.cuda()
                logits = classifier(x)
                preds = logits.argmax(dim=1)
                all_preds.append(preds)
                all_labels.append(y)

            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            acc = (all_preds == all_labels).float().mean().item()

        reward = torch.tensor(acc).cuda()
        baseline_reward = 0.9 * baseline_reward + 0.1 * reward if episode > 0 else reward
        advantage = reward - baseline_reward + 1e-8

        # Aggregate all actions for policy gradient loss
        log_probs_tensor = torch.stack(log_probs)  # [NUM_STEPS, NUM_OBJECTS]
        total_log_prob = log_probs_tensor.sum()
        loss = -advantage * total_log_prob
        policy_optim.zero_grad()
        loss.backward()
        policy_optim.step()
        scheduler.step(reward)  # reward is the metric to monitor

        print(f"Episode {episode}, Accuracy: {acc:.3f}, Loss: {loss:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Five-Shot Training")
    parser.add_argument("--num_episode", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=10, help="Number of examples per class in each run")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--lr_rl", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_cls", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--wd_cls", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--shot", type=int, default=10)
    parser.add_argument("--data_per_class", type=int, default=500)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--plot_every", type=int, default=100)
    parser.add_argument("--log_name", type=str, default='')
    args = parser.parse_args()
    # if args.feature:
    print(args)
    data_folder = '../ShapeNet/'
    
    train_image_dict = dataset.load_image_dict(data_folder, 'train', args.data_per_class)
    test_image_dict = dataset.load_image_dict(data_folder, 'test', 100)
    train_action(train_image_dict, test_image_dict, args, 10)
