# net_canny.py  (rewritten to produce DEPTH, while keeping the class name "Net")
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthNet(nn.Module):
    """
    Depth estimation network wrapper (MiDaS via torch.hub).

    Input:
        img: float tensor in [0, 1], shape (B, 3, H, W), RGB

    Output:
        depth: float tensor, shape (B, 1, H, W), relative depth (scale not metric)
    """

    def __init__(
        self,
        model_type: str = "DPT_Hybrid",
        input_size: int = 384,
        use_cuda: bool = False,
    ):
        super().__init__()
        self.model_type = model_type
        self.input_size = int(input_size)

        self.device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

        # Try common MiDaS hub repos (names have varied historically)
        repo_candidates = ["isl-org/MiDaS", "intel-isl/MiDaS"]
        last_err = None
        midas = None
        for repo in repo_candidates:
            try:
                midas = torch.hub.load(repo, model_type, trust_repo=True)
                break
            except Exception as e:
                last_err = e

        if midas is None:
            raise RuntimeError(
                "Failed to load MiDaS model via torch.hub. "
                "Tried repos: %s. Last error: %r"
                % (repo_candidates, last_err)
            )

        self.midas = midas.to(self.device).eval()

        # Normalization used by MiDaS examples (ImageNet mean/std)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() != 4 or img.size(1) != 3:
            raise ValueError(f"Expected img shape (B,3,H,W), got {tuple(img.shape)}")

        img = img.to(self.device)

        b, c, h, w = img.shape

        # Resize to a fixed square input (simple + reliable)
        inp = F.interpolate(img, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)

        # Normalize
        inp = (inp - self.mean.cuda()) / self.std.cuda()

        pred = self.midas(inp)

        # Some models return (B,H,W); others may return (B,1,H,W)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        elif pred.dim() == 4 and pred.size(1) != 1:
            # If something unexpected, try to reduce to 1 channel
            pred = pred.mean(dim=1, keepdim=True)

        # Upsample back to original resolution
        depth = F.interpolate(pred, size=(h, w), mode="bicubic", align_corners=False)

        return depth


if __name__ == "__main__":
    Net()
