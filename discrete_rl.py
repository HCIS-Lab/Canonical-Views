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
import time
import plot.plot as plot
import os
class_list = ['airplane', 'bathtub', 'bed', 'bin', 'bottle', 'bowl', 'bus', 'can', 'case', 'hat']
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        resnet = models.resnet18(pretrained=False)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 27)
        # Predefined action lookup table: 4 options per axis -> 4^3 = 64
        step_options = [-15., 0., 15.]
        import itertools
        self.action_table = torch.tensor(list(itertools.product(step_options, repeat=3)), dtype=torch.float32).cuda()

    def forward(self, images, hidden):
        batch_size = images.size(0)
        x = self.cnn(images).view(batch_size, 1, -1)  # [B, 1, 64]
        lstm_out, hidden = self.lstm(x, hidden)
        h = lstm_out.squeeze(1)
        logits = self.fc(h)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()  # [B]

        selected_rot = self.action_table[action_idx] / 180.  # degrees to radians
        log_prob = dist.log_prob(action_idx)
        entropy = dist.entropy().mean()

        return selected_rot.to(images.device), hidden, log_prob, entropy

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
        random.seed(time.time())  # or time.time_ns()
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
        # skip the current frame
        if rot[0].item() == current_rot[0].item() and rot[1].item() ==current_rot[1].item() and rot[2].item() ==current_rot[2].item():
            continue
        dist = total_angle_distance(new_rot, rot)
        if dist < min_dist:
            min_dist = dist
            closest_view = image_list[i]
            matched_rot = rot

    current_view = Image.open(closest_view).convert('RGB')
    current_view = transform(current_view)
    return current_view, matched_rot

def train_action(train_dict, test_dict, args, num_classes=10):

    root_dir = 'results'
    os.makedirs(root_dir, exist_ok=True)  # Will create the folder if it doesn't exist
    if args.log_name == '':
        exp_dir = 'discrete_'+str(args.lr_rl) + '_cls_lr' + str(args.lr_cls) \
        + '_episodes' + str(args.num_episode)+'_epochs'+str(args.epochs) \
        + '_wd' + str(args.wd_cls) +'_shot'+str(args.shot) 
    else:
        exp_dir = args.log_name
    exp_dir = os.path.join(root_dir, exp_dir)

    i = 0
    while os.path.isdir(exp_dir):
        i+=1
        exp_dir = exp_dir + '_' + str(i)
        
    os.makedirs(exp_dir, exist_ok=True)
    # ==== Main RL Loop ====
    policy = RotationPolicy().cuda()
    policy.train()
    policy_optim = optim.Adam(policy.parameters(), lr=args.lr_rl)
    scheduler = torch.optim.lr_scheduler.StepLR(policy_optim, step_size=10, gamma=0.9)
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)

    baseline_reward = 0

    NUM_OBJECTS = 10
    NUM_STEPS = args.shot
    NUM_CLASSES = 10  # Change depending on task
    rot_episode = {}
    all_adv = []
    test_x = []
    test_y = []
    for i, class_name in enumerate(class_list):
        test_x = test_x + test_dict[class_name]['image']
        test_y = test_y + [torch.tensor(i, dtype=torch.int)]*args.data_per_class

    test_dataset = dataset.ActionDataset(test_x, test_y, test=True)
    test_dataset = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=10, pin_memory=True)

    for episode in range(args.num_episode):
        # all_images = [[] for _ in range(NUM_OBJECTS)]
        all_images = []
        all_labels = []
        all_entropy = []
        # all_rot = [[] for _ in range(NUM_OBJECTS)]
        all_rot = []
        hidden = (torch.zeros(1, NUM_OBJECTS, 128).cuda(), torch.zeros(1, NUM_OBJECTS, 128).cuda())
        current_views, current_rot = random_view(train_dict)
        for i in range(NUM_OBJECTS):
            all_images.append(current_views[i])
            all_rot.append(current_rot[i])
            all_labels.append(float(i))
        current_views = current_views.cuda()
        all_actions = []  # Store actions over all steps

        log_probs = []
        for step in range(NUM_STEPS-1):
            actions, hidden, log_prob, entropy = policy(current_views, hidden)
            log_probs.append(log_prob)
            all_entropy.append(entropy)

            next_views = []
            next_rot = []
            for i in range(NUM_OBJECTS):
                # rot = (actions[i] * torch.pi).detach().cpu().numpy()
                delta_rot = (actions[i])
                new_image, new_rot = next_view(current_views[i], i, train_dict, current_rot[i], delta_rot)
                all_images.append(new_image)
                next_views.append(new_image)
                all_rot.append(new_rot)
                next_rot.append(new_rot)
                all_labels.append(float(i))
            current_views = torch.stack(next_views).cuda()
            current_rot = next_rot
        rot_episode[episode] = all_rot

        # Train classifier on collected views
        classifier = initialize_fixed_model().cuda()
        clf_opt = optim.Adam(classifier.parameters(), lr=args.lr_cls, weight_decay=args.wd_cls)
        criterion = nn.CrossEntropyLoss()

        combined = list(zip(all_images, all_labels))
        random.shuffle(combined)
        train_x, train_y = zip(*combined)
        train_dataset = dataset.ActionDataset(train_x, train_y)
        train_dataset = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, pin_memory=True)
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
        log_probs_tensor = torch.stack(log_probs)

        entropy = torch.stack(all_entropy).sum(dim=-1).mean()
        # total_log_prob = log_probs_tensor.sum()
        loss = -(log_probs_tensor.sum() * advantage.detach()) - 0.01*entropy
        # loss = -advantage * total_log_prob
        policy_optim.zero_grad()
        loss.backward()
        policy_optim.step()
        scheduler.step(reward)  # reward is the metric to monitor

        print(f"Episode {episode}, Accuracy: {acc:.3f}, Loss: {loss:.3f}")

        plot.plot_view_selcetion_discrete(rot_episode, exp_dir)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Five-Shot Training")
    parser.add_argument("--num_episode", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=10, help="Number of examples per class in each run")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--lr_rl", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_cls", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--wd_cls", type=float, default=5e-2, help="Learning rate")
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--data_per_class", type=int, default=500)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--plot_every", type=int, default=100)
    parser.add_argument("--log_name", type=str, default='')
    args = parser.parse_args()
    # if args.feature:
    print(args)
    data_folder = '../ShapeNet/'
    
    train_image_dict = dataset.load_image_dict(data_folder, 'train', args.data_per_class)
    test_image_dict = dataset.load_image_dict(data_folder, 'test', args.data_per_class)
    train_action(train_image_dict, test_image_dict, args, 10)
