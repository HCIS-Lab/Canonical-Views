import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
import cv2
import numpy as np
import torch.nn.functional as F

def in_angle_range(angle, category, offset):
        lower_bound = (category - offset) % 360
        upper_bound = (category + offset) % 360
        if lower_bound < upper_bound:
            return lower_bound <= angle <= upper_bound
        else:
            return angle >= lower_bound or angle <= upper_bound  # Wraps around 360

def check_angles(x, y, z):
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
            if in_angle_range(int(angle), cat, offset):
                planar[i] = True
                break

    return all(planar)

def load_feature_paths(data_folder, split, num_sample, planar_ratio=0.0):
    feature_paths = {}
    class_list = ['airplane', 'bathtub', 'bed', 'bin', 'bottle', 'bowl', 'bus', 'can', 'case', 'hat']
    for class_idx, class_name in enumerate(class_list):
        folder_path = os.path.join(data_folder, class_name, split)
        instances = os.listdir(folder_path)
        feature_paths[class_name] = []
        # for ins in instances:
        ins = instances[0]
        instances_path = os.path.join(folder_path, ins, 'screenshot')

        planar = []
        non_planar = []
        
        for f in os.listdir(instances_path):
            if os.path.isfile(os.path.join(instances_path, f)) and f.endswith(".pt"):
                rot = f.split('.')[0]
                rot = rot.split('_')
                x, y, z = rot[2], rot[3], rot[4]
                if check_angles(x, y, z):
                    planar.append(os.path.join(instances_path, f))
                else:
                    non_planar.append(os.path.join(instances_path, f))
        
        # total_num = int(num_planar/planar_ratio)
        # num_non_planar = int((1-planar_ratio)*total_num)
        if planar_ratio != 0.0:
            planar = random.sample(planar, 10)
            num_planar = len(planar)
            total_num = int(num_planar/planar_ratio)
            num_non_planar = int((1-planar_ratio)*total_num)
            non_planar = random.sample(non_planar, num_non_planar)
        files = planar + non_planar
        # files = [os.path.join(instances_path, f) for f in os.listdir(instances_path) if os.path.isfile(os.path.join(instances_path, f)) and f.endswith(".pt")]
        # files = files[:num_sample]
        feature_paths[class_name] = files

    return feature_paths


def load_set_feature_paths(data_folder, split, shot):
    feature_paths = []
    planar_ratio = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    class_list = ['airplane', 'bathtub', 'bed', 'bin', 'bottle', 'bowl', 'bus', 'can', 'case', 'hat']
    
    for i in range(11):
        p_ratio = int(i*0.1)
        num_p = int(shot*p_ratio)
        num_np = int(shot*(1-p_ratio))
        feature_paths.append({})
        for class_idx, class_name in enumerate(class_list):
            folder_path = os.path.join(data_folder, class_name, split)
            instances = os.listdir(folder_path)
            feature_paths[-1][class_name] = []
            # for ins in instances:
            ins = instances[0]
            instances_path = os.path.join(folder_path, ins, 'screenshot')

            planar = []
            non_planar = []
            
            for f in os.listdir(instances_path):
                if os.path.isfile(os.path.join(instances_path, f)) and f.endswith(".pt"):
                    rot = f.split('.')[0]
                    rot = rot.split('_')
                    x, y, z = rot[2], rot[3], rot[4]
                    if check_angles(x, y, z):
                        planar.append(os.path.join(instances_path, f))
                    else:
                        non_planar.append(os.path.join(instances_path, f))
            
            planar = random.sample(planar, num_p)
            non_planar = random.sample(non_planar, num_np)
            feature_paths[-1][class_name] = planar + non_planar

    return feature_paths

class SetFeatureDataset(Dataset):
    def __init__(self, feature_paths, split='train'):
        self.feature_paths = feature_paths
        self.split = split

        if self.split == 'test':
            class_list = ['airplane', 'bathtub', 'bed', 'bin', 'bottle', 'bowl', 'bus', 'can', 'case', 'hat']
            path_list = []
            label_list = []
            for img_set in self.feature_paths:
                for i, class_name in enumerate(class_list):
                    for image_path in img_set[class_name]:
                        path_list.append(image_path)
                        label_list.append(i)
            self.feature_paths = path_list
            self.labels = label_list



    def __len__(self):
        return len(self.feature_paths)
    
    def __getitem__(self, idx):
        if self.split == 'train':
            class_list = ['airplane', 'bathtub', 'bed', 'bin', 'bottle', 'bowl', 'bus', 'can', 'case', 'hat']
            path_set = self.feature_paths[idx]
            feature = []
            label = []
            for i, class_name in enumerate(class_list):
                for img_path in path_set[class_name]:
                    feature.append(torch.load(img_path).squeeze())
                    label.append(i)
            feature = torch.stack(feature)
            label = torch.tensor(label)
            return {'feature': feature,
                    'label': label,
                    }
        else:
            feature = torch.load(self.feature_paths[idx]).squeeze()
            label = torch.tensor(self.labels[idx])
            return {'feature': feature,
                    'label': label,
                    }

class SelectedSetDataset:
    def __init__(self, features, labels, batch_size):
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle()
    
    def shuffle(self):
        combined = list(zip(self.features, self.labels))
        random.shuffle(combined)
        # self.features, self.labels = zip(*combined)
        self.features, self.labels = map(list, zip(*combined))

    def __len__(self):
        return len(self.features)
    
    def __iter__(self):
        for i in range(0, len(self.features), self.batch_size):
            batch_features = self.features[i:i + self.batch_size]
            batch_labels = self.labels[i:i + self.batch_size]
            yield {'feature': batch_features, 'label': batch_labels}

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

def load_image_dict(data_folder, split, num_sample):
    image_dict = {}
    class_list = ['airplane', 'bathtub', 'bed', 'bin', 'bottle', 'bowl', 'bus', 'can', 'case', 'hat']
    for class_idx, class_name in enumerate(class_list):
        folder_path = os.path.join(data_folder, class_name, split)
        instances = os.listdir(folder_path)
        image_dict[class_name] = {}
        # for ins in instances:
        ins = instances[0]
        instances_path = os.path.join(folder_path, ins, 'screenshot')
        files = [os.path.join(instances_path, f) for f in os.listdir(instances_path) if os.path.isfile(os.path.join(instances_path, f)) and f.endswith(".jpg")]
        files.sort()
        files = files[:num_sample]
        image_dict[class_name]['image'] = files
        rot = []
        for file in files:
            file_name = os.path.basename(file)
            attrbutes = file_name.split('_')
            class_name, id, x, y, z = attrbutes[0], attrbutes[1], attrbutes[2], attrbutes[3], attrbutes[4]

            rot_input = [(float(x)-180)/180., (float(y)-180)/180., (float(z)-180)/180.]
            rot_input = torch.tensor(rot_input)
            rot.append(rot_input)
        image_dict[class_name]['rot'] = rot

    return image_dict

class ActionDataset(Dataset):
    def __init__(self, feature, labels, test=False):
        self.feature = feature
        self.labels = labels
        self.test = test
        self.transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self, idx):
        if self.test:
            feature = Image.open(self.feature[idx]).convert("RGB")
            feature = self.transform(feature)
        else:
            feature = self.feature[idx]
        label = self.labels[idx]
        return feature, label



def load_edge_paths(data_folder, split, num_sample):
    image_paths = {}
    class_list = ['airplane', 'bathtub', 'bed', 'bin', 'bottle', 'bowl', 'bus', 'can', 'case', 'hat']
    for class_idx, class_name in enumerate(class_list):
        folder_path = os.path.join(data_folder, class_name, split)
        instances = os.listdir(folder_path)
        image_paths[class_name] = []
        # for ins in instances:
        ins = instances[0]
        instances_path = os.path.join(folder_path, ins, 'screenshot')
        files = [os.path.join(instances_path, f) for f in os.listdir(instances_path) if os.path.isfile(os.path.join(instances_path, f)) and f.endswith(".png")]
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


class DepthDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, load_depth=False):
        transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color augmentation
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
        
class EdgeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, load_depth=False, load_feature=False):
        transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to 224x224
        transforms.Grayscale(num_output_channels=1),  # just in case
        transforms.ToTensor(),  # Convert to tensor
    ])
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.load_depth = load_depth
        self.load_feature = load_feature

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
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]

        feature_path = img_path.replace("/edge/", "/feature/")
        feature_path = feature_path.replace(".png", ".jpg.pt")
        feature_path = feature_path.replace("_edge", "")

        depth_path = img_path.replace("/edge/", "/depth/")
        depth_path = depth_path.replace("_edge", "")

        file_name = os.path.basename(img_path)
        attrbutes = file_name.split('_')
        class_name, id, x, y, z = attrbutes[0], attrbutes[1], attrbutes[2], attrbutes[3], attrbutes[4]

        rot_input = [float(x)/180., float(y)/180., float(z)/180.]
        rot_input = torch.tensor(rot_input)

        rot = x+'_'+y+'_'+z
        out = {'edge': image,
                'label': label,
                'class_name': class_name,
                'id': id,
                'rot': rot,
                'rot_input': rot_input,
                'planar': bool(self.check_angles(x, y, z))
                }
        if self.load_feature:
            out['feature'] = torch.load(feature_path).squeeze()
        else:
            out['feature'] = feature_path
        if self.load_depth:
            depth = Image.open(depth_path).convert('L')  # Ensure it's in RGB mode
            depth_transform = GrayscaleDepthToTensor(depth_min=0.0, depth_max=10.0, out_size=(224, 224))
            depth = depth_transform(depth)  # shape: (1, H, W)
            out['depth'] = depth
        else:
            out['depth'] = depth_path
        return out

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
        
        # image_file_path.pop(-6)
        image_file_path.pop(-6)
        image_file_path = os.path.join(*image_file_path)

        return {'feature': feature,
                'label': label,
                'class_name': class_name,
                'id': id,
                'rot': rot,
                'rot_input': rot_input,
                'planar': bool(self.check_angles(x, y, z)),
                'image_path': os.path.join(image_file_path)}



class DepthFeatureDataset(Dataset):
    def __init__(self, feature_paths, labels, load_depth=False):
        self.feature_paths = feature_paths
        self.labels = labels
        self.load_depth = load_depth
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
        # /nfs/wattrel/data/md0/kung/ShapeNet/feature/airplane/train/1021a0914a7207aff927ed529ad90a11/screenshot/xxx.pt
        image_file_path = self.feature_paths[idx].split('/')
        image_file_path[-1] = image_file_path[-1].replace('.pt', '')
        
        image_file_path.pop(-6)
        image_path = os.path.join(*image_file_path)

        image_file_path[-1] = image_file_path[-1].replace('jpg', 'png')
        image_file_path.insert(-5, "depth")
        depth = os.path.join(*image_file_path)
        if self.load_depth:
            depth = Image.open(depth).convert('L')  # Ensure it's in RGB mode
            depth_transform = GrayscaleDepthToTensor(depth_min=0.0, depth_max=10.0, out_size=(224, 224))
            depth = depth_transform(depth)  # shape: (1, H, W)

            return {'feature': feature,
                    'label': label,
                    'class_name': class_name,
                    'id': id,
                    'rot': rot,
                    'rot_input': rot_input,
                    'planar': bool(self.check_angles(x, y, z)),
                    'image_path': os.path.join(image_path),
                    'depth': depth
                    }
        else:
            return {'feature': feature,
                    'label': label,
                    'class_name': class_name,
                    'id': id,
                    'rot': rot,
                    'rot_input': rot_input,
                    'planar': bool(self.check_angles(x, y, z)),
                    'image_path': os.path.join(image_path),
                    'depth_path': depth
                    }
class GrayscaleDepthToTensor(object):
    def __init__(self, depth_min=0.0, depth_max=10.0, out_size=(224, 224)):
        """
        Args:
            depth_min (float): Minimum depth value expected in meters.
            depth_max (float): Maximum depth value expected in meters.
            out_size (tuple): (height, width) to resize depth to (default 224x224)
        """
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.out_size = out_size

    def __call__(self, depth_pil):
        """
        Args:
            depth_pil (PIL.Image): 8-bit grayscale image loaded with convert('L')

        Returns:
            torch.Tensor: Tensor of shape (1, H, W) with depth values in meters
        """
        # Convert PIL grayscale [0,255] → float tensor [0.0, 1.0]
        depth_tensor = transforms.ToTensor()(depth_pil).squeeze(0)  # shape: (H, W)

        # Rescale to real-world depth range [depth_min, depth_max]
        depth_tensor = depth_tensor * (self.depth_max - self.depth_min) + self.depth_min

        # Resize to (1, H_out, W_out)
        depth_tensor = depth_tensor.unsqueeze(0)  # (1, H, W)
        depth_tensor = F.interpolate(depth_tensor.unsqueeze(0), size=self.out_size, mode='bilinear', align_corners=False)
        return depth_tensor.squeeze(0)  # (1, H_out, W_out)


class SelectedDataset:
    def __init__(self, features, labels, batch_size, depth=None):
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.depth = depth
        self.shuffle()
    
    def shuffle(self):
        if self.depth == None:
            combined = list(zip(self.features, self.labels))
            random.shuffle(combined)
            self.features, self.labels = zip(*combined)
        else:
            combined = list(zip(self.features, self.labels, self.depth))
            random.shuffle(combined)
            self.features, self.labels, self.depth = zip(*combined)
    
    def __len__(self):
        return len(self.features)
    
    def __iter__(self):
        for i in range(0, len(self.features), self.batch_size):
            batch_features = self.features[i:i + self.batch_size]
            batch_labels = self.labels[i:i + self.batch_size]
            if self.depth == None:
                yield {'feature': batch_features, 'label': batch_labels}
            else:
                batch_depth = []
                for depth_path in self.depth[i:i + self.batch_size]:
                    depth = Image.open(depth_path).convert('L')  # Ensure it's in RGB mode
                    depth_transform = GrayscaleDepthToTensor(depth_min=0.0, depth_max=10.0, out_size=(224, 224))

                    depth_tensor = depth_transform(depth)  # shape: (1, H, W)

                    # depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

                    # depth = self.transform(depth)  # Apply the given transformation
                    batch_depth.append(depth_tensor)
                
                batch_depth = torch.stack(batch_depth)  # Stack tensors to form a batch
                batch_labels = torch.tensor(batch_labels)  # Convert labels to tensor

                yield {'feature': batch_features, 'label': batch_labels, 'depth': batch_depth}


class SelectedImageDataset:
    def __init__(self, image_paths, labels, batch_size):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
    #     self.transform = transforms.Compose([
    #     transforms.Resize((224, 224)),  # Resize to 224x224
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color augmentation
    #     transforms.ToTensor(),  # Convert to tensor
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    # ])
        self.transform = transforms.Compose([
                        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),  # Random crop and resize
                        transforms.RandomHorizontalFlip(p=0.5),               # Random horizontal flip
                        transforms.RandomRotation(degrees=10),                # Random rotation
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # More intense color jitter
                        transforms.RandomAffine(
                            degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),  # Small affine transforms
                        transforms.GaussianBlur(kernel_size=3),               # Slight blur
                        transforms.ToTensor(),                                # Convert to tensor
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                             std=[0.229, 0.224, 0.225])        # Normalize
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


class SelectedEdgeDataset:
    def __init__(self, edges, labels,  
                batch_size,
                depth_paths=None, feature_paths=None, 
                load_depth=False, load_feature=False
                ):
        
        self.edges = edges
        self.labels = labels
        self.depth_paths = depth_paths
        self.feature_paths = feature_paths
        self.batch_size = batch_size
        if not load_depth:
            self.depth_paths = [None]*len(self.edges)
        if not load_feature:
            self.feature_paths = [None]*len(self.edges)
        self.shuffle()
    
    def shuffle(self):
        combined = list(zip(self.edges, self.labels, self.depth_paths, self.feature_paths))
        random.shuffle(combined)
        self.edges, self.labels, self.depth_paths, self.feature_paths = zip(*combined)
    
    def __len__(self):
        return len(self.edges)
    
    def __iter__(self):
        for i in range(0, len(self.edges), self.batch_size):
            batch_edges = []
            batch_depths = []
            batch_features = []
            batch_labels = self.labels[i:i + self.batch_size]

            batch_labels = torch.tensor(batch_labels)  # Convert labels to tensor
            
            out = {'label': batch_labels}

            if self.depth_paths[0] != None:
                for depth in self.depth_paths[i:i + self.batch_size]:
                    depth = Image.open(depth).convert('L')  # Ensure it's in RGB mode
                    depth_transform = GrayscaleDepthToTensor(depth_min=0.0, depth_max=10.0, out_size=(224, 224))
                    depth = depth_transform(depth)  # shape: (1, H, W)
                    batch_depths.append(depth)
                batch_depths = torch.stack(batch_depths)
                out['depth'] = batch_depths
            else:
                out['depth'] = None
            if self.feature_paths[0] != None:
                for feature in self.feature_paths[i:i + self.batch_size]:
                    batch_features.append(torch.load(feature).squeeze())
                out['feature'] = batch_features
                out['edge'] = None
            else:
                for img in self.edges[i:i + self.batch_size]:
                    batch_edges.append(img)
                batch_edges = torch.stack(batch_edges)  # Stack tensors to form a batch
                out['edge'] = batch_edges
                out['feature'] = None
            yield out

            # batch_images = self.image_paths[i:i + self.batch_size]
            # batch_labels = self.labels[i:i + self.batch_size]
            # yield {'feature': batch_images, 'label': batch_labels}

