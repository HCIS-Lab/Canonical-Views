import os
import argparse
import cv2
import numpy as np
import torch
from typing import List, Tuple
from net_depth import DepthNet


def depth_to_uint16(depth: np.ndarray, invert: bool = False) -> np.ndarray:
    """
    Convert float depth map to 16-bit PNG-friendly array.
    Uses robust normalization (2–98 percentile) to reduce outliers.
    """
    d = depth.astype(np.float32)

    d_min = np.percentile(d, 2.0)
    d_max = np.percentile(d, 98.0)
    if not np.isfinite(d_min) or not np.isfinite(d_max) or (d_max - d_min) < 1e-6:
        d_min = float(np.min(d))
        d_max = float(np.max(d))

    if (d_max - d_min) < 1e-6:
        norm = np.zeros_like(d, dtype=np.float32)
    else:
        norm = (d - d_min) / (d_max - d_min)
        norm = np.clip(norm, 0.0, 1.0)

    if invert:
        norm = 1.0 - norm

    return (norm * 65535.0).round().astype(np.uint16)


def rgb_uint8_to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """
    rgb: HxWx3 uint8
    returns: 3xHxW float32 in [0,1]
    """
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float().div_(255.0)


@torch.no_grad()
def run_depth_batch(
    rgbs: List[np.ndarray],
    net: DepthNet,
) -> np.ndarray:
    """
    rgbs: list of HxWx3 RGB uint8, all same H,W
    returns: depth batch as numpy array (B,H,W) float32
    """
    if len(rgbs) == 0:
        return np.zeros((0,), dtype=np.float32)

    # Stack into (B,3,H,W)
    tensors = [rgb_uint8_to_tensor(img) for img in rgbs]
    batch = torch.stack(tensors, dim=0)  # (B,3,H,W)

    depth = net(batch)  # (B,1,H,W)
    depth = depth.squeeze(1).detach().cpu().numpy().astype(np.float32)  # (B,H,W)
    return depth


def main():
    parser = argparse.ArgumentParser(description="Generate MiDaS depth images for a folder of RGB images (batched).")
    parser.add_argument("--input_folder", type=str, required=False,
                        default="/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_4")
    parser.add_argument("--output_folder", type=str, required=False,
                        default="/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_depth_1_4")
    parser.add_argument("--ext", type=str, default=".png", help="Only process files with this extension (e.g. .png)")
    parser.add_argument("--model_type", type=str, default="DPT_Hybrid",
                        help="MiDaS model type (e.g., DPT_Hybrid, DPT_Large, MiDaS_small)")
    parser.add_argument("--input_size", type=int, default=384, help="Network input size (square)")
    parser.add_argument("--invert", action="store_true", help="Invert visualization (near=bright vs far=bright)")
    parser.add_argument("--save_npy", action="store_true", help="Also save raw float depth as .npy next to png")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (only batches same-resolution images)")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    net = DepthNet(model_type=args.model_type, input_size=args.input_size, use_cuda=args.use_cuda)

    processed = 0
    skipped = 0
    failed = 0

    # Accumulators for batching
    batch_rgbs: List[np.ndarray] = []
    batch_save_pngs: List[str] = []
    batch_save_npys: List[str] = []
    batch_hw: Tuple[int, int] = (-1, -1)

    def flush_batch():
        nonlocal processed, batch_rgbs, batch_save_pngs, batch_save_npys, batch_hw
        if len(batch_rgbs) == 0:
            return

        depths = run_depth_batch(batch_rgbs, net)  # (B,H,W)
        for i in range(depths.shape[0]):
            depth_map = depths[i]
            depth_u16 = depth_to_uint16(depth_map, invert=args.invert)

            cv2.imwrite(batch_save_pngs[i], depth_u16)
            if args.save_npy:
                np.save(batch_save_npys[i], depth_map)

        processed += len(batch_rgbs)

        # Reset
        batch_rgbs = []
        batch_save_pngs = []
        batch_save_npys = []
        batch_hw = (-1, -1)

        if processed % 200 == 0:
            print(f"Processed {processed} images... (skipped {skipped}, failed {failed})")

    for root, _, files in os.walk(args.input_folder):
        for file in files:
            if not file.lower().endswith(args.ext.lower()):
                continue

            input_path = os.path.join(root, file)
            rel_path = os.path.relpath(input_path, args.input_folder)
            save_path = os.path.join(args.output_folder, rel_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            base, _ = os.path.splitext(save_path)
            save_png = base + ".png"
            save_npy = base + ".npy"

            if os.path.exists(save_png) and (not args.save_npy or os.path.exists(save_npy)):
                skipped += 1
                continue

            img_bgr = cv2.imread(input_path)
            if img_bgr is None:
                print(f"Warning: Failed to read {input_path}")
                failed += 1
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]

            # If this image has a different resolution than current batch, flush first
            if batch_hw != (-1, -1) and (h, w) != batch_hw:
                flush_batch()

            # Start/continue batch
            if batch_hw == (-1, -1):
                batch_hw = (h, w)

            batch_rgbs.append(img_rgb)
            batch_save_pngs.append(save_png)
            batch_save_npys.append(save_npy)

            # Flush when batch is full
            if len(batch_rgbs) >= max(1, args.batch_size):
                flush_batch()

    # Flush leftovers
    flush_batch()

    print(f"Done. Processed={processed}, Skipped={skipped}, Failed={failed}, Output={args.output_folder}")


if __name__ == "__main__":
    main()
