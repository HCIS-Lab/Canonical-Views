import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random

def load_feature_paths(data_folder, split, num_sample):
    feature_paths = {}
    class_list = ['airplane', 'bathtub', 'bed', 'bin', 'bottle', 'bowl', 'bus', 'can', 'case', 'hat']
    for class_idx, class_name in enumerate(class_list):
        folder_path = os.path.join(data_folder, class_name, class_name, split)
        instances = os.listdir(folder_path)
        feature_paths[class_name] = []
        # for ins in instances:
        ins = instances[0]
        instances_path = os.path.join(folder_path, ins, 'screenshot')
        files = [os.path.join(instances_path, f) for f in os.listdir(instances_path) if os.path.isfile(os.path.join(instances_path, f)) and f.endswith(".pt")]
        files = files[:num_sample]
        feature_paths[class_name] = files

    return feature_paths

def load_image_paths(data_folder, split, num_sample):
    image_paths = {}
    class_list = ['airplane', 'bathtub', 'bed', 'bin', 'bottle', 'bowl', 'bus', 'can', 'case', 'hat']
    for class_idx, class_name in enumerate(class_list):
        folder_path = os.path.join(data_folder, class_name, split)
        instances = os.listdir(folder_path)
        image_paths[class_name] = []
        # for ins in instances:
        ins = instances[0]
        instances_path = os.path.join(folder_path, ins, 'screenshot')
        files = [os.path.join(instances_path, f) for f in os.listdir(instances_path) if os.path.isfile(os.path.join(instances_path, f)) and f.endswith(".jpg")]
        files = files[:num_sample]
        image_paths[class_name] = files

    return image_paths

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color augmentation
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def in_angle_range(self, angle, category, offset):
        lower_bound = (category - offset) % 360
        upper_bound = (category + offset) % 360
        if lower_bound < upper_bound:
            return lower_bound <= angle <= upper_bound
        else:
            return angle >= lower_bound or angle <= upper_bound  # Wraps around 360

    def check_angles(self, x, y, z):
        offset = 15
        categories = [
            0,
            90,
            180,
            270
        ]
        planar = [False]*3
        for i, angle in enumerate([x, y, z]):
            for cat in categories:
                if self.in_angle_range(int(angle), cat, offset):
                    planar[i] = True
                    break

        return all(planar)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]

        file_name = os.path.basename(img_path)
        attrbutes = file_name.split('_')
        class_name, id, x, y, z = attrbutes[0], attrbutes[1], attrbutes[2], attrbutes[3], attrbutes[4]

        rot_input = [float(x)/180., float(y)/180., float(z)/180.]
        rot_input = torch.tensor(rot_input)

        rot = x+'_'+y+'_'+z
        return {'feature': image,
                'label': label,
                'class_name': class_name,
                'id': id,
                'rot': rot,
                'rot_input': rot_input,
                'planar': bool(self.check_angles(x, y, z))}
        

class FeatureDataset(Dataset):
    def __init__(self, feature_paths, labels):
        self.feature_paths = feature_paths
        self.labels = labels
        
    def in_angle_range(self, angle, category, offset):
        lower_bound = (category - offset) % 360
        upper_bound = (category + offset) % 360
        if lower_bound < upper_bound:
            return lower_bound <= angle <= upper_bound
        else:
            return angle >= lower_bound or angle <= upper_bound  # Wraps around 360

    def check_angles(self, x, y, z):
        offset = 15
        categories = [
            0,
            90,
            180,
            270
        ]
        planar = [False]*3
        for i, angle in enumerate([x, y, z]):
            for cat in categories:
                if self.in_angle_range(int(angle), cat, offset):
                    planar[i] = True
                    break

        return all(planar)

    def __len__(self):
        return len(self.feature_paths)
    
    def __getitem__(self, idx):
        feature = torch.load(self.feature_paths[idx]).squeeze()
        label = self.labels[idx]
        file_name = os.path.basename(self.feature_paths[idx])
        attrbutes = file_name.split('_')
        class_name, id, x, y, z, view = attrbutes[0], attrbutes[1], attrbutes[2], attrbutes[3], attrbutes[4], attrbutes[5].split('.')[0]

        rot_input = [float(x)/180., float(y)/180., float(z)/180.]
        rot_input = torch.tensor(rot_input)

        rot = x+'_'+y+'_'+z
        # /nfs/wattrel/data/md0/kung/ShapeNet/airplane/train/1021a0914a7207aff927ed529ad90a11/screenshot/.jpg
        # /nfs/wattrel/data/md0/kung/ShapeNet/feature/airplane/airplane/train/1021a0914a7207aff927ed529ad90a11/screenshot/xxx.pt
        image_file_path = self.feature_paths[idx].split('/')
        image_file_path[-1] = image_file_path[-1].replace('.pt', '')
        image_file_path = image_file_path.pop(-6)
        image_file_path = image_file_path.pop(-6)
        image_file_path = os.path.join(*image_file_path)

        return {'feature': feature,
                'label': label,
                'class_name': class_name,
                'id': id,
                'rot': rot,
                'rot_input': rot_input,
                'planar': bool(self.check_angles(x, y, z)),
                'image_path': os.path.join(image_file_path)}

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

class SelectedImageDataset:
    def __init__(self, image_paths, labels, batch_size):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color augmentation
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
        self.shuffle()
    
    def shuffle(self):
        combined = list(zip(self.image_paths, self.labels))
        random.shuffle(combined)
        self.image_paths, self.labels = zip(*combined)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __iter__(self):
        for i in range(0, len(self.image_paths), self.batch_size):
            batch_images = []
            batch_labels = self.labels[i:i + self.batch_size]

            for img_path in self.image_paths[i:i + self.batch_size]:
                image = Image.open(img_path).convert('RGB')  # Ensure it's in RGB mode
                image = self.transform(image)  # Apply the given transformation
                batch_images.append(image)
            
            batch_images = torch.stack(batch_images)  # Stack tensors to form a batch
            batch_labels = torch.tensor(batch_labels)  # Convert labels to tensor
            
            yield {'feature': batch_images, 'label': batch_labels}
            
            # batch_images = self.image_paths[i:i + self.batch_size]
            # batch_labels = self.labels[i:i + self.batch_size]
            # yield {'feature': batch_images, 'label': batch_labels}

