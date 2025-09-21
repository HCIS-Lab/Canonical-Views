# #!/usr/bin/env python
# # coding: utf-8

import os
import sys
import torch
import json
import pytorch3d

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from pytorch3d.datasets import (
    ShapeNetCore,
)
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    look_at_view_transform
)

# add path for demo utils functions 
import sys
import os
sys.path.append(os.path.abspath(''))


taxonomy_list = ["airplane,aeroplane,plane",
"ashcan,trash can,garbage can,wastebin,ash bin,ash-bin,ashbin,dustbin,trash barrel,trash bin",
"bag,traveling bag,travelling bag,grip,suitcase",
"basket,handbasket",
"bathtub,bathing tub,bath,tub",
"bed",
"bench",
"birdhouse",
"bookshelf",
"bottle",
"bowl",
"bus,autobus,coach,charabanc,double-decker,jitney,motorbus,motorcoach,omnibus,passenger vehi",
"cabinet",
"camera,photographic camera",
"can,tin,tin can",
"cap",
"car,auto,automobile,machine,motorcar",
"chair",
"clock",
"computer keyboard,keypad",
"dishwasher,dish washer,dishwashing machine",
"display,video display",
"earphone,earpiece,headphone,phone",
"faucet,spigot",
"file,file cabinet,filing cabinet",
"guitar",
"helmet",
"jar",
"knife",
"lamp",
"laptop,laptop computer",
"loudspeaker,speaker,speaker unit,loudspeaker system,speaker system",
"mailbox,letter box",
"motorcycle,bike",
"mug",
"grand piano,grand",
"pistol,handgun,side arm,shooting iron",
"pot,flowerpot",
"table",
"telephone,phone,telephone set",
"tower",
"train,railroad train",
"vessel,watercraft",
"washer,automatic washer,washing machine"
]

with open('/nfs/wattrel/data/md0/kung/ShapeNetCore.v2/taxonomy.json', 'r') as file:
    taxonomy_json = json.load(file)



# -----parallel------
def _chunk_indices(n, k):
    sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    out, s = [], 0
    for sz in sizes:
        if sz > 0:
            out.append(torch.arange(s, s+sz))
        s += sz
    return out

def render_multigpu_by_views(
    dataset, model_id, cameras, devices, *, raster_settings, lights, **kwargs
):
    """
    Returns a tensor [V, H, W, 4] with the same order as `cameras`.
    """
    devices = [torch.device(d) for d in devices]
    V = len(cameras)  # batch size of cameras
    chunks = _chunk_indices(V, len(devices))

    # worker function runs on one GPU
    def worker(dev, idx):
        sub_cam = cameras[idx].to(dev)             # subset cameras to device
        sub_lights = lights.to(dev)                # move lights to device
        imgs = dataset.render(
            model_ids=[model_id],
            device=dev,
            cameras=sub_cam,
            raster_settings=raster_settings,
            lights=sub_lights,
            **kwargs,                              # e.g., blend_params, etc.
        )                                          # -> [len(idx), H, W, 4] on GPU
        return idx, imgs.detach().cpu()

    # You can run sequentially (simplest & safest):
    parts = [worker(dev, idx) for dev, idx in zip(devices, chunks)]

    # stitch back in order
    parts.sort(key=lambda p: p[0][0].item())       # sort by first index for stability
    H, W, C = parts[0][1].shape[1:]
    out = torch.empty((V, H, W, C), dtype=parts[0][1].dtype)
    for idx, imgs in parts:
        out[idx] = imgs
    return out
# -----

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    
SHAPENET_PATH = "/nfs/wattrel/data/md0/kung/ShapeNetCore.v2"
shapenet_dataset = ShapeNetCore(SHAPENET_PATH, version=2)

import math
import torch
from pytorch3d.renderer.cameras import look_at_view_transform, OpenGLPerspectiveCameras

def build_view_set(dist=1.5, step_deg=30, device="cpu"):
    """
    Build uniformly distributed camera views over the sphere using elevation and azimuth grids.
    Returns:
      R, T: tensors for all views (N, 3, 3) and (N, 3)
      meta: list[dict] with labels, flags, filename_stub, zenith, and Cartesian coordinates (x, y, z)
    """
    principals = [
        {"label": "front(+Z)",  "elev": 0.0,   "azim":   0.0,   "axis": "+Z", "group": "principal"},
        {"label": "right(+X)",  "elev": 0.0,   "azim":  90.0,   "axis": "+X", "group": "principal"},
        {"label": "back(-Z)",   "elev": 0.0,   "azim": 180.0,   "axis": "-Z", "group": "principal"},
        {"label": "left(-X)",   "elev": 0.0,   "azim": 270.0,   "axis": "-X", "group": "principal"},
        {"label": "top(+Y)",    "elev": 90.0,  "azim":   0.0,   "axis": "+Y", "group": "principal"},
        {"label": "bottom(-Y)", "elev": -90.0, "azim":   0.0,   "axis": "-Y", "group": "principal"},
    ]

    elevs = torch.arange(-90, 91, step_deg, dtype=torch.float32)
    sphere_views = []
    for elev in elevs:
        azim_step = step_deg if abs(elev) < 90 else 360
        azims = torch.arange(0, 360, azim_step, dtype=torch.float32)
        for azim in azims:
            sphere_views.append({
                "label": f"sphere_e{int(elev.item()):+03d}_a{int(azim.item()):03d}",
                "elev": float(elev.item()),
                "azim": float(azim.item()),
                "axis": None,
                "group": "sphere_uniform",
            })

    def key(elev, azim):
        az = (azim % 360.0 + 360.0) % 360.0
        return (round(elev, 6), round(az, 6))

    seen, merged = set(), []
    for v in principals + sphere_views:
        k = key(v["elev"], v["azim"])
        if k in seen:
            continue
        seen.add(k)
        v["is_principal"] = (v["group"] == "principal")
        v["is_planar"] = (abs(v["elev"]) in (0.0, 90.0))
        v["dist"] = float(dist)
        # zenith (polar angle from +Y axis)
        v["zenith"] = float(90.0 - v["elev"])
        # Cartesian coordinates
        elev_rad = math.radians(v["elev"])
        azim_rad = math.radians(v["azim"])
        x = dist * math.cos(elev_rad) * math.sin(azim_rad)
        y = dist * math.sin(elev_rad)
        z = dist * math.cos(elev_rad) * math.cos(azim_rad)
        v["x"], v["y"], v["z"] = x, y, z
        elev_int = int(round(v["elev"]))
        azim_int = int(round(v["azim"]))
        zenith_int = int(round(v["zenith"]))
        v["filename_stub"] = f"z{zenith_int:03d}_e{elev_int:+03d}_a{azim_int:03d}_x{x:+.2f}_y{y:+.2f}_z{z:+.2f}"
        merged.append(v)

    dists = torch.tensor([m["dist"] for m in merged], dtype=torch.float32, device=device)
    elevs = torch.tensor([m["elev"] for m in merged], dtype=torch.float32, device=device)
    azims = torch.tensor([m["azim"] for m in merged], dtype=torch.float32, device=device)

    R, T = look_at_view_transform(dist=dists, elev=elevs, azim=azims, device=device)
    return R, T, merged

def sample_uniform(meta, R, T, k, generator=None):
    N = len(meta)
    idx = torch.randperm(N, generator=generator)[:k]
    return R[idx], T[idx], [meta[i] for i in idx.tolist()]

def sample_stratified(meta, R, T, per_group=None, generator=None):
    buckets = {}
    for i, m in enumerate(meta):
        buckets.setdefault(m["group"], []).append(i)

    if per_group is None:
        total = sum(len(buckets[g]) for g in buckets)
        share = max(1, total // len(buckets))
        per_group = {g: min(share, len(buckets[g])) for g in buckets}

    chosen = []
    gen = generator or torch.Generator()
    for g, need in per_group.items():
        pool = torch.tensor(buckets[g])
        if need >= len(pool):
            chosen.extend(pool.tolist())
        else:
            idx = pool[torch.randperm(len(pool), generator=gen)[:need]].tolist()
            chosen.extend(idx)

    chosen = torch.tensor(chosen)
    return R[chosen], T[chosen], [meta[i] for i in chosen.tolist()]


# ---------------------------
# Example usage
# ---------------------------
R, T, meta = build_view_set(dist=1.0, step_deg=30, device=device)

# Example A: cameras for ALL views (principal + every 10° on equator)
cameras = OpenGLPerspectiveCameras(R=R, T=T, device=device)

raster_settings = RasterizationSettings(image_size=224,
                        faces_per_pixel=10,        # >1 for soft blending
                        perspective_correct=True,
                        cull_backfaces=True,
                        # Optional quality knobs:
                        bin_size=0,              # set to 0/None for highest quality on GPU with enough memory
                        max_faces_per_bin=200000,)
lights = PointLights(location=torch.tensor([0.0, 1.0, -2.0], device=device)[None],device=device)

devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]  # e.g., 4 gpus

for cat in taxonomy_list:
    model_exist = 0
    synsetId = ''
    for item in taxonomy_json:
        if item["name"] == cat:
            synsetId = item["synsetId"]
            break

    if os.path.exists(os.path.join(SHAPENET_PATH, synsetId)):
        model_list = os.listdir(os.path.join(SHAPENET_PATH, synsetId))
        print(f"Contents of '{os.path.join(SHAPENET_PATH, synsetId)}':", model_list)
    else:
        print(f"Directory '{os.path.join(SHAPENET_PATH, synsetId)}' does not exist.")

    for model_idx, model_id in enumerate(model_list):
        if model_exist > 299:
            break
        try:
            images = render_multigpu_by_views(
                shapenet_dataset,
                model_id=model_id,
                cameras=cameras,                      # batched cameras
                devices=devices,
                raster_settings=raster_settings,
                lights=lights,
                # blend_params=...,  # pass any extra kwargs your render() accepts
            )
        except:
            continue

        model_exist+=1
        print(cat)
        print(str(model_exist)+'/300')

        os.makedirs('shapenet_dataset', exist_ok=True)
        os.makedirs(os.path.join('shapenet_dataset', cat), exist_ok=True)
        os.makedirs(os.path.join('shapenet_dataset', cat, model_id), exist_ok=True)
        for i in range(images.shape[0]):
            pil_image = transforms.ToPILImage()(images[i,:,:,:].permute(2,0,1)).convert('RGBA')
            file_name = f"e{int(meta[i]['elev']):+03d}_a{int(meta[i]['azim']):03d}"

            if meta[i]["is_principal"]:
                file_name = file_name + '_principal'
            if meta[i]["is_planar"]:
                file_name = file_name + '_planar'
            file_name = file_name + '.png'

            pil_image.save(os.path.join('shapenet_dataset',cat, model_id, file_name))
