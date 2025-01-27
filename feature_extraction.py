import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

def get_resnet50_model():
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the final classification layer
    model = torch.nn.DataParallel(model)  # Enable multi-GPU support
    model.eval()
    return model

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, image_path

def save_features_for_folder(input_folder, output_folder, model, device, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    os.makedirs(output_folder, exist_ok=True)
    
    image_paths = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith((".jpg")):
                print('root', root)
                print('file', file)
                image_paths.append(os.path.join(root, file))
    
    dataset = ImageDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model.to(device)
    
    for images, paths in tqdm(dataloader, desc="Processing images"):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            features = model(images).squeeze()
        
        for feature, path in zip(features, paths):
            rel_path = os.path.relpath(path, input_folder)
            save_path = os.path.join(output_folder, rel_path + ".pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(feature.cpu(), save_path)

if __name__ == "__main__":
    input_folder = "/N/project/ego4d_vlm/ShapeNet"
    output_folder = "/N/project/ego4d_vlm/ShapeNet/feature"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet50_model().to(device)
    save_features_for_folder(input_folder, output_folder, model, device, batch_size=64)
