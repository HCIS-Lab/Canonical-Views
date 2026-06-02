
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from net_canny import Net

# ============================
# Configuration
# ============================
IMAGE_ROOT = '/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_23'
EDGE_ROOT = '/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_edge_1_23'

BATCH_SIZE = 2048
NUM_WORKERS = 8
THRESHOLD = 10.0

EDGE_ROOT = EDGE_ROOT + '_' + str(THRESHOLD)
# ============================
# Dataset
# ============================
class RecursiveImageDataset(Dataset):
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.paths = []

        for dirpath, _, filenames in os.walk(self.root):
            for f in filenames:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    full_path = os.path.join(dirpath, f)
                    rel_path = os.path.relpath(full_path, self.root)
                    self.paths.append(rel_path)

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        rel_path = self.paths[idx]
        full_path = os.path.join(self.root, rel_path)

        img = cv2.imread(full_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        return img, rel_path

# ============================
# Model Loader
# ============================
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = Net(threshold=THRESHOLD)
    net.eval()

    if torch.cuda.device_count() > 1:
        print(f"🚀 Using {torch.cuda.device_count()} GPUs")
        net = torch.nn.DataParallel(net)

    net.to(device)
    return net, device

# ============================
# Main Processing Loop
# ============================
def main():
    dataset = RecursiveImageDataset(IMAGE_ROOT)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    net, device = load_model()

    pbar = tqdm(total=len(dataset), desc="Generating edge images", unit="img")

    with torch.no_grad():
        for batch, rel_paths in loader:
            batch = batch.to(device, non_blocking=True)

            edges = net(batch)
            edges = edges.cpu().numpy()

            for edge, rel_path in zip(edges, rel_paths):
                edge = edge.squeeze()
                edge = (edge * 255).astype(np.uint8)

                out_path = os.path.join(EDGE_ROOT, rel_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                cv2.imwrite(out_path, edge)

            # Update progress by actual batch size
            pbar.update(len(rel_paths))

    pbar.close()
    print("✅ Edge generation complete.")

# ============================
# Entry Point
# ============================
if __name__ == "__main__":
    main()
