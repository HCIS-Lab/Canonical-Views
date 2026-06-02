import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, threshold=3.0, use_cuda=True):
        super().__init__()
        self.threshold = threshold
        self.use_cuda = use_cuda

        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [0,  0,  0],
             [1,  2,  1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, x):
        """
        x: [B, 3, H, W]
        """
        B, _, H, W = x.shape
        device = x.device

        # grayscale
        x = x.mean(dim=1, keepdim=True)

        # sobel gradients
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)

        mag = torch.sqrt(gx ** 2 + gy ** 2)

        # normalize per image
        mag = mag.view(B, -1)
        mag = mag / (mag.max(dim=1, keepdim=True).values + 1e-6)
        mag = mag.view(B, 1, H, W)

        # threshold
        edges = (mag > (self.threshold / 255.0)).float()

        return edges
