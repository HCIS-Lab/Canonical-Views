# #!/usr/bin/env python
# # coding: utf-8
import argparse
import os
import sys
import torch
import json
import csv
import pytorch3d

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from pytorch3d.datasets import (
    ShapeNetCore,
)

from pytorch3d.renderer import (
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    RasterizationSettings,
    OpenGLPerspectiveCameras,
    PointLights,
    look_at_view_transform
)
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.renderer import Materials

# add path for demo utils functions 
import sys
import os
sys.path.append(os.path.abspath(''))

from typing import Optional, Tuple


# 32
taxonomy_list = [
'airplane,aeroplane,plane', 
'ashcan,trash can,garbage can,wastebin,ash bin,ash-bin,ashbin,dustbin,trash barrel,trash bin', 
'basket,handbasket', 
'bathtub,bathing tub,bath,tub', 
'bed', 
'bench',
'bookshelf', 
'bottle', 
'bowl', 
'bus,autobus,coach,charabanc,double-decker,jitney,motorbus,motorcoach,omnibus,passenger vehi', 
'cabinet', 
'camera,photographic camera', 
'car,auto,automobile,machine,motorcar', 
'chair', 
'clock', 
'display,video display', 
'faucet,spigot', 
'guitar', 
'helmet', 
'knife', 
'lamp',
'laptop,laptop computer', 
'loudspeaker,speaker,speaker unit,loudspeaker system,speaker system',
'motorcycle,bike',
'mug',
'pistol,handgun,side arm,shooting iron',
'table', 
'telephone,phone,telephone set',
'tower',
'train,railroad train', 
'vessel,watercraft', 
'washer,automatic washer,washing machine'
]


# -----parallel------
def _chunk_indices(n, k):
    sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    out, s = [], 0
    for sz in sizes:
        if sz > 0:
            out.append(torch.arange(s, s+sz))
        s += sz
    return out


def texture_has_color(meshes, eps=0.01):
    if meshes.textures is None or not hasattr(meshes.textures, "maps"):
        return False
    tex = meshes.textures.maps_padded()[0]
    return (
        (tex[...,0] - tex[...,1]).abs().mean() > eps or
        (tex[...,0] - tex[...,2]).abs().mean() > eps
    )

def render_singlegpu_by_views(
    obj_path,
    dataset,
    model_id,
    cameras,
    device="cuda:0",
    *,
    raster_settings,
    lights,
    materials=None,   # if you use materials, pass them in
    **kwargs,
):
    """
    Renders one pre-chunked batch of cameras on a single GPU and returns [V,H,W,4] on CPU.
    """
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)  # pin current device

    # Move inputs to the target device
    cameras = cameras.to(dev)
    lights  = lights.to(dev)
    if materials is not None:
        try:
            materials = materials.to(dev)
        except AttributeError:
            pass  # some material structs might not have .to()
    else:
        materials = Materials(
        device=device,
        ambient_color=((1.0, 1.0, 1.0),),
        diffuse_color=((1.0, 1.0, 1.0),),
        specular_color=((0.0, 0.0, 0.0),),
        shininess=64.0,
    )
    # Optional: stricter debugging (use once to pinpoint)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    with torch.no_grad():
        # Defensive: tell the dataset which device to use
        imgs = dataset.render(
            model_ids=[model_id],
            device=dev,
            cameras=cameras,
            raster_settings=raster_settings,
            lights=lights,
            materials=materials,
            **kwargs,
        )

        # renderer = MeshRenderer(
        #     rasterizer=MeshRasterizer(
        #         cameras=cameras,
        #         raster_settings=raster_settings
        #     ),
        #     shader=HardPhongShader(
        #         device=device,
        #         cameras=cameras,
        #         lights=lights,
        #         materials=materials,   # ← REQUIRED
        #     )
        # )
        

        # depth
        # rasterizer = MeshRasterizer(
        #     cameras=cameras,
        #     raster_settings=raster_settings,
        # )
        # shader = HardPhongShader(
        #     device=device,
        #     cameras=cameras,
        #     lights=lights,
        # )
        meshes = load_objs_as_meshes(
            [obj_path],  # must be a LIST
            device=device,
            load_textures=True
        )

        # if not texture_has_color(meshes):
        #     # print("⚠️ Grayscale or missing texture → using vertex colors")
        #     verts = meshes.verts_packed()
        #     verts_rgb = torch.rand_like(verts)
        #     meshes.textures = TexturesVertex(verts_rgb[None])

        # verts = meshes.verts_packed()
        # textures = TexturesVertex(
        #     verts_features=torch.ones_like(verts)[None]
        # )

        # meshes.textures = textures

        num_views=57
        meshes = meshes.extend(num_views)  # num_views = 57

        

        # images = renderer(meshes)
        # rgb = images[..., :3]   # ✅ real color restored

        # print(type(meshes.textures))

        # fragments = rasterizer(meshes)

        # images = shader(
        #     fragments,
        #     meshes,
        #     cameras=cameras
        # )
        # fragments = renderer.rasterizer(meshes)
        fragments = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        )(meshes)
        depth = fragments.zbuf[..., 0]

        # rgb = images[..., :3]
        # depth = fragments.zbuf[..., 0]
        depth[depth == float("inf")] = 0.0
        mask = depth > 0
        d_inv = torch.zeros_like(depth)
        d_inv[mask] = 1.0 / (depth[mask] + 1e-6)
        d_inv[mask] = (d_inv[mask] - d_inv[mask].min()) / (
            d_inv[mask].max() - d_inv[mask].min()
        )

        # Safety sync to surface kernel errors here
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    return imgs.detach().cpu(), d_inv


import math
import torch
from pytorch3d.renderer.cameras import look_at_view_transform, OpenGLPerspectiveCameras

def build_view_set(dist=1.8, step_deg=30, device="cpu", roll_step_deg=22.5, rolls=None):
    """
    Build uniformly distributed camera views over the sphere using elevation and azimuth grids.
    Returns:
      R, T: tensors for all views (N, 3, 3) and (N, 3)
      meta: list[dict] with labels, flags, filename_stub, zenith, Cartesian (x,y,z), and roll (deg)
    """

    step_deg = 22.5

    step_deg = 22.5
    elevs = torch.arange(-180.0, 180.0, step_deg)
    azims = torch.arange(-180.0, 180.0, step_deg)
    all_meta = []
    dirs = []

    for elev in elevs:
        for azim in azims:
            e = math.radians(float(elev))
            a = math.radians(float(azim))
            x = math.cos(e) * math.sin(a)
            y = math.sin(e)
            z = math.cos(e) * math.cos(a)

            all_meta.append({
                "elev": float(elev.item()),
                "azim": float(azim.item()),
            })
            dirs.append((x, y, z))

    # --- Deduplicate directions ---
    seen = {}  # map from rounded (x,y,z) to first index
    for i, (x, y, z) in enumerate(dirs):
        key = (round(x, 6), round(y, 6), round(z, 6))  # avoid FP noise
        if key not in seen:
            seen[key] = i

    unique_indices = list(seen.values())
    unique_views = [all_meta[i] for i in unique_indices]

    print("Total angle pairs:", len(all_meta))
    print("Unique directions:", len(unique_views))


    # # elevs = torch.arange(-90, 91, step_deg, dtype=torch.float32)
    # elevs = torch.arange(-180.0, 180.0, step_deg, dtype=torch.float32)
    # sphere_views = []

    # for elev in elevs:
    #     # azim_step = step_deg if abs(elev) < 90 else 360
    #     azim_step = 22.5
    #     # azims = torch.arange(0, 360, azim_step, dtype=torch.float32)
    #     azims = torch.arange(-180.0, 180.0, azim_step, dtype=torch.float32)
    #     for azim in azims:
    #         sphere_views.append({
    #             "elev": float(elev.item()),
    #             "azim": float(azim.item()),
    #         })

    seen, merged = set(), []
    # for v in principals + sphere_views:
    for v in unique_views:

        v["dist"] = float(dist)
        # Cartesian coordinates
        elev_rad = math.radians(v["elev"])
        azim_rad = math.radians(v["azim"])
        x = dist * math.cos(elev_rad) * math.sin(azim_rad)
        y = dist * math.sin(elev_rad)
        z = dist * math.cos(elev_rad) * math.cos(azim_rad)
        v["x"], v["y"], v["z"] = x, y, z

        # elev_float = v["elev"]
        # azim_float = v["azim"]
        # zenith_float= v["zenith"]
        # v["filename_stub"] = f"z{zenith_float:03.1f}_e{elev_float:+03.1f}_a{azim_float:03.1f}_x{x:+.2f}_y{y:+.2f}_z{z:+.2f}"
        # v["filename_stub"] = f"z{v["zenith"]:03d}_e{v["elev"]:+03d}_a{v["azim"]:03d}_x{x:+.2f}_y{y:+.2f}_z{z:+.2f}"
        merged.append(v)

    dists = torch.tensor([m["dist"] for m in merged], dtype=torch.float32, device=device)
    elevs = torch.tensor([m["elev"] for m in merged], dtype=torch.float32, device=device)
    azims = torch.tensor([m["azim"] for m in merged], dtype=torch.float32, device=device)

    R, T = look_at_view_transform(dist=dists, elev=elevs, azim=azims, device=device)

    # --- Roll handling ---
    # rolls: list of degrees or step size over [0, 360)
    if rolls is not None and len(rolls) > 0:
        roll_list = torch.tensor(rolls, dtype=torch.float32, device=device)
    elif roll_step_deg is None or float(roll_step_deg) <= 0:
        roll_list = torch.tensor([0.0], dtype=torch.float32, device=device)
    else:
        roll_list = torch.arange(-180.0, 180.0, roll_step_deg, dtype=torch.float32, device=device)

    N = R.shape[0]
    R_out, T_out, meta_out = [], [], []
    for rdeg in roll_list.tolist():
        theta = math.radians(rdeg)
        c, s = math.cos(theta), math.sin(theta)
        Rz = torch.zeros((N, 3, 3), dtype=R.dtype, device=device)
        Rz[:, 0, 0] = c; Rz[:, 0, 1] = -s; Rz[:, 1, 0] = s; Rz[:, 1, 1] = c; Rz[:, 2, 2] = 1.0
        Rr = torch.bmm(Rz, R)  # rotate camera frame about its Z axis (roll)
        R_out.append(Rr)
        T_out.append(T)
        for m in merged:
            m2 = dict(m)
            m2['roll'] = float(rdeg)
            roll_int = int(round(rdeg))
            # base_stub = m2.get('filename_stub', f"e{int(round(m2['elev'])):+03d}_a{int(round(m2['azim'])):03d}")
            # m2['filename_stub'] = base_stub + f"_r{roll_int:03d}"
            # m2["is_planar"] = m2["is_planar"] and (abs(m2["roll"]) in (0.0, 90.0, 180.0))
            # m2["is_planar_like"] = m2["is_planar_like"] and (abs(m2["roll"]) in (0.0, 90.0, 180.0))
            
            if (abs(m2["roll"]) in (0.0, 90.0, 180.0)) and (abs(m2["elev"]) in (0.0, 90.0, 180.0)) and (abs(m2["azim"]) in (0.0, 90.0, 180.0)):
                m2["is_planar"] = True
            else:
                m2["is_planar"] = False

            if ((abs(m2["roll"]) in (157.5, 22.5, 67.5, 112.5)) and (abs(m2["elev"]) in (0.0, 90.0, 180.0)) and (abs(m2["azim"]) in (0.0, 90.0, 180.0))) or \
            (abs(m2["roll"]) in (0.0, 90.0, 180.0)) and (abs(m2["elev"]) in (157.5, 22.5, 67.5, 112.5)) and (abs(m2["azim"]) in (0.0, 90.0, 180.0)) or \
            (abs(m2["roll"]) in (0.0, 90.0, 180.0)) and (abs(m2["elev"]) in (0.0, 90.0, 180.0)) and  (abs(m2["azim"]) in (157.5, 22.5, 67.5, 112.5)): \
                m2["is_planar_like"] = True
            else:
                m2["is_planar_like"] = False


            meta_out.append(m2)

    R = torch.cat(R_out, dim=0)
    T = torch.cat(T_out, dim=0)
    merged = meta_out
    return R, T, merged

# =============================
# Principal axis detection (camera-independent)
# =============================
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes


def principal_axes_from_mesh(meshes: Meshes, area_weighted: bool = True):
    V = meshes.verts_packed()
    if not area_weighted:
        mu = V.mean(0, keepdim=True)
        Vc = V - mu
        C = (Vc.t() @ Vc) / max(1, Vc.shape[0] - 1)
    else:
        F = meshes.faces_packed()
        tri = V[F]
        centroids = tri.mean(dim=1)
        n = torch.cross(tri[:,1] - tri[:,0], tri[:,2] - tri[:,0])
        areas = 0.5 * torch.linalg.norm(n, dim=1) + 1e-12
        w = areas / areas.sum()
        mu = (w[:,None] * centroids).sum(dim=0, keepdim=True)
        C = ((centroids - mu).t() * w) @ (centroids - mu)
    evals, evecs = torch.linalg.eigh(C)
    order = torch.argsort(evals, descending=True)
    axes = evecs[:, order]          # columns: major→minor
    if torch.det(axes) < 0:         # enforce right-handed
        axes[:, 2] *= -1
    return axes[:, 0] / (torch.norm(axes[:,0]) + 1e-12)  # major unit vector


def vector_to_elev_azim(v: torch.Tensor):
    """Convert a direction vector to (elev, azim) in degrees for look_at_view_transform."""
    v = v / (torch.norm(v) + 1e-12)
    elev = torch.asin(v[1]).item() * 180.0 / math.pi
    azim = torch.atan2(v[0], v[2]).item() * 180.0 / math.pi
    return elev, azim

# ---------- helpers ----------
def safe_normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / (torch.norm(v) + eps)


def camera_basis_from_center(
    cam_center: torch.Tensor,
    world_up: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:    
    """
    Returns right r, up u, forward f (all unit, right-handed).
    cam_center: world position of camera (aiming at origin).
    """
    device = cam_center.device
    dtype  = cam_center.dtype
    if world_up is None:
        world_up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)

    f = safe_normalize(-cam_center)                 # camera forward (from camera towards origin)
    r = torch.cross(world_up, f)
    if torch.norm(r) < 1e-8:                        # world_up ~ colinear with f, choose alternate up
        alt = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        if abs(torch.dot(alt, f)) > 0.99:
            alt = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        r = torch.cross(alt, f)
    r = safe_normalize(r)
    u = safe_normalize(torch.cross(f, r))
    return r, u, f

def rotate_about_axis(v: torch.Tensor, axis: torch.Tensor, angle_rad: float) -> torch.Tensor:
    """Rodrigues' rotation formula: rotate v around unit 'axis' by angle_rad."""
    axis = safe_normalize(axis)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return v*c + torch.cross(axis, v)*s + axis*torch.dot(axis, v)*(1.0 - c)


def major_axis_angles(
    u_major: torch.Tensor,
    cam_center: torch.Tensor,
    world_up: Optional[torch.Tensor] = None,
    roll_deg: Optional[float] = None,
    gravity_world: Optional[torch.Tensor] = None,
    upright_mode: str = "image",
    gravity_weight: float = 0.7
) -> Tuple[float, float, bool]:
    """
    to_eye_deg: angle between major axis and viewing direction (0..90)
    inplane_deg: signed deg in image plane vs *perceptual upright* (+CW), NaN if undefined
    valid: whether in-plane angle is defined
    """
    device, dtype = cam_center.device, cam_center.dtype
    if world_up is None:
        world_up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)

    # camera basis from look-at(0) (same as before)
    r_cam, u_cam, f = camera_basis_from_center(cam_center, world_up)

    # roll (visual tilt) if provided
    if roll_deg is not None:
        roll_rad = math.radians(float(roll_deg))
        r_cam = rotate_about_axis(r_cam, f, roll_rad)
        u_cam = rotate_about_axis(u_cam, f, roll_rad)

    # choose a perceptual-upright candidate
    if upright_mode == "image" or gravity_world is None:
        u_perc = u_cam
    else:
        # project (negative) gravity onto the image plane as an "up" cue
        g = safe_normalize(-gravity_world.to(device=device, dtype=dtype))   # gravity down -> use up
        u_grav_img = g - torch.dot(g, f) * f
        if torch.norm(u_grav_img) < 1e-8:
            u_grav_img = u_cam  # degenerate: camera looks along gravity
        u_grav_img = safe_normalize(u_grav_img)

        if upright_mode == "gravity":
            u_perc = u_grav_img
        elif upright_mode == "blend":
            # convex blend (then re-project to image plane to keep orthogonality)
            u_mix = gravity_weight * u_grav_img + (1 - gravity_weight) * u_cam
            u_perc = u_mix - torch.dot(u_mix, f) * f
            if torch.norm(u_perc) < 1e-8:
                u_perc = u_cam
            u_perc = safe_normalize(u_perc)
        else:
            u_perc = u_cam  # fallback

    # rebuild a right-handed image-plane basis using the chosen upright
    r = safe_normalize(torch.cross(f, u_perc))
    u = safe_normalize(torch.cross(r, f))

    r = r.cuda()
    u = u.cuda()
    # (1) angle major axis vs eyesight
    v = safe_normalize(u_major).cuda()
    f=f.cuda()
    dot = torch.clamp(torch.abs(torch.dot(v.cuda(), f.cuda())), 0.0, 1.0)
    to_eye_deg = math.degrees(math.acos(float(dot)))

    # (2) in-plane angle vs chosen "perceptual upright"
    v_proj = v - torch.dot(v, f) * f
    vpn = float(torch.norm(v_proj))
    if vpn < 1e-8:
        return to_eye_deg, float('nan'), False
    vx, vy = float(torch.dot(v_proj, r)), float(torch.dot(v_proj, u))
    inplane_deg = math.degrees(math.atan2(vx, vy))  # +CW from upright toward right
    return to_eye_deg, inplane_deg, True


def main(args):

    with open('/nfs/wattrel/data/md0/kung/ShapeNetCore.v2/taxonomy.json', 'r') as file:
        taxonomy_json = json.load(file)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    SHAPENET_PATH = "/nfs/wattrel/data/md0/kung/ShapeNetCore.v2"
    shapenet_dataset = ShapeNetCore(SHAPENET_PATH, version=2)


    split = args.split
    batch = args.batch

    # R, T, meta = build_view_set(dist=1.0, step_deg=22.5, device=device, roll_step_deg=22.5)
    R, T, meta = build_view_set(dist=1.0, step_deg=22.5, device=device, rolls=[0.0])

    # split_size = 128
    split_size = 57
    meta = meta[split_size*batch:split_size*(batch+1)]
    R = R[split_size*batch:split_size*(batch+1),:,:]
    T = T[split_size*batch:split_size*(batch+1),:]
    view_start_idx = split_size*batch
    # view_start_idx = 0

    # Example A: cameras for ALL views (principal + every 10° on equator)
    cameras = OpenGLPerspectiveCameras(R=R, T=T, device=device)


    raster_settings = RasterizationSettings(
        image_size=224,
        faces_per_pixel=10,        # keep small to save mem
        perspective_correct=True,
        cull_backfaces=True,
        bin_size=0,              # ↓ from 32 → fewer faces per bin
        max_faces_per_bin=20000,  # ↑ capacity per bin
    )

    lights = PointLights(
        location=torch.tensor([0.0, 1.0, -2.0], device=device)[None],device=device)
    # lights = PointLights(
    #     device=device,
    #     location=[[0.0, 0.0, 3.0]],
    #     ambient_color=((0.3, 0.3, 0.3),),
    #     diffuse_color=((0.7, 0.7, 0.7),),
    #     specular_color=((0.1, 0.1, 0.1),),
    # )
    # lights = PointLights(
    # device=device,
    # location=[[2.0, 2.0, 2.0]],
    # ambient_color=((0.6, 0.6, 0.6),),
    # diffuse_color=((0.4, 0.4, 0.4),),
    # specular_color=((0.0, 0.0, 0.0),),
    # )

    devices=device

    instance_id_to_idx = {}

    num_classes = 0
    class_list = []
    for cat in taxonomy_list:
        if cat not in instance_id_to_idx.keys():
            instance_id_to_idx[cat] = {}
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
            continue
        for model_idx, model_id in enumerate(model_list):
            view_idx = view_start_idx
            # view_idx = 0
            
            obj_path = os.path.join(SHAPENET_PATH, synsetId, model_id, "models", "model_normalized.obj")
            if not os.path.exists(obj_path):
                obj_path = os.path.join(SHAPENET_PATH, synsetId, model_id, "models", "model.obj")
            if not os.path.exists(obj_path):
                continue
            # try:
            images, depth = render_singlegpu_by_views(
                obj_path,
                shapenet_dataset,
                model_id=model_id,
                cameras=cameras,                      # batched cameras
                device=device,
                raster_settings=raster_settings,
                lights=lights,
                # blend_params=...,  # pass any extra kwargs your render() accepts
            )
            # except:
            #     continue

            

            model_exist+=1

            if split == "train":
                if model_exist > 50:
                    # num_classes+=1
                    # class_list.append(cat)
                    break
            elif split =="val":
                if model_exist <= 50:
                    continue
                elif model_exist >55:
                    break
            elif split == "test":
                if model_exist <= 55:
                    continue
                elif model_exist >105:
                    break
            elif split == "test-down":
                if model_exist <= 105:
                    continue
                elif model_exist >110:
                    break

            print(cat)
            print(str(model_exist)+'/300')

            root = 'modelnet_32_60_1_23'
            root_depth = 'depth_' + root
            os.makedirs(root, exist_ok=True)
            os.makedirs(os.path.join(root, cat), exist_ok=True)
            os.makedirs(os.path.join(root, cat, split), exist_ok=True)

            os.makedirs(root_depth, exist_ok=True)
            os.makedirs(os.path.join(root_depth, cat), exist_ok=True)
            os.makedirs(os.path.join(root_depth, cat, split), exist_ok=True)

            parallel_tol_deg = 25  # tweak as you like (e.g., 5–15)
            angles_out = []  

            # os.makedirs(os.path.join(root, cat, model_id), exist_ok=True)
            if model_id not in instance_id_to_idx[cat].keys():
                instance_id_to_idx[cat][model_id] = len(instance_id_to_idx[cat])+1
            for i in range(images.shape[0]):
                view_idx +=1
                # --- Dominant axis (longest) of the object (camera-independent) ---
                # obj_path = os.path.join(SHAPENET_PATH, cat, model_id, "models", "model_normalized.obj")
                # if not os.path.exists(obj_path):
                #     obj_path = os.path.join(SHAPENET_PATH, cat, model_id, "models", "model.obj")
                # try:
                meshes = load_objs_as_meshes([obj_path], device=str(device))
                u_major = principal_axes_from_mesh(meshes, area_weighted=True)   # (3,)
                # except Exception:
                #     # fallback to +Z if loading fails
                #     u_major = torch.tensor([0.0, 0.0, 1.0])



                if all(k in meta[i] for k in ('x','y','z')):
                    cam_center = torch.tensor([meta[i]['x'], meta[i]['y'], meta[i]['z']], dtype=torch.float32)
                else:
                    # fallback from (dist, elev, azim)
                    de = math.radians(float(meta[i]['elev']))
                    az = math.radians(float(meta[i]['azim']))
                    dd = float(meta[i].get('dist', 1.5))
                    cam_center = torch.tensor([dd*math.cos(de)*math.sin(az), dd*math.sin(de), dd*math.cos(de)*math.cos(az)],
                                              dtype=torch.float32)

                # to_eye_deg, inplane_deg, inplane_valid = major_axis_angles(u_major, cam_center, roll_deg=roll)
                to_eye_deg, inplane_deg, inplane_valid = major_axis_angles(
                                u_major, cam_center,
                                roll_deg=meta[i].get('roll', meta[i].get('tilt', None)),
                                gravity_world=torch.tensor([0.0, -1.0, 0.0], device=device),
                                upright_mode="gravity"   # or "blend" with gravity_weight=0.6~0.8
                            )

                is_dom_parallel = (to_eye_deg <= parallel_tol_deg)
                meta[i]['major_to_eye_deg'] = round(float(to_eye_deg), 3)
                meta[i]['major_proj_upright_deg'] = None if not inplane_valid else round(float(inplane_deg), 3)
                meta[i]['major_proj_upright_valid'] = bool(inplane_valid)
                meta[i]['is_dom_parallel'] = bool(is_dom_parallel)

                # view_dir = -cam_center / (torch.norm(cam_center) + 1e-12)
                # # parallel or anti-parallel both count (abs dot)
                # dot = torch.clamp(torch.abs(torch.dot(view_dir, u_major)), 0.0, 1.0)
                # angle_deg = math.degrees(math.acos(float(dot)))
                # is_dom_parallel = angle_deg <= parallel_tol_deg
                # meta[i]["is_dominant_axis_parallel"] = bool(is_dom_parallel)

                # images_i = (images[i].clamp(0, 1) * 255).byte().cpu().numpy()

                pil_image = transforms.ToPILImage()(images[i,:,:,:].permute(2,0,1)).convert('RGBA')
                file_name = str(model_id) + '_' + str(view_idx) + '_' 
                file_name = file_name + f"e{meta[i]['elev']:+03.1f}_a{meta[i]['azim']:03.1f}_r{meta[i]['roll']:03.1f}"
                file_name = file_name + f"_eye{meta[i]['major_to_eye_deg']}_upright{meta[i]['major_proj_upright_deg']}"

                # expanded or foreshortened
                if meta[i]["is_planar"]:
                    file_name = file_name + '_planar'
                # foreshortened
                if meta[i]["is_dom_parallel"]:
                    file_name = file_name + '_short'

                if meta[i]["is_planar_like"]:
                    file_name = file_name + '_like'

                

                file_name = file_name + '.png'
                final_name = file_name

                # Image.fromarray(images_i, mode="RGB").save(os.path.join(root,cat,split,final_name))
                pil_image.save(os.path.join(root,cat,split,final_name))
                # Image.fromarray(depth, mode="I;16").save(os.path.join(root_depth,cat,split,final_name))
                # Image.fromarray((depth * 255).byte().cpu().numpy()).save(os.path.join(root_depth,cat,split,final_name))
                d = depth[i]                      # (H, W)
                mask = d > 0

                d_norm = torch.zeros_like(d)
                if mask.any():
                    d_min = d[mask].min()
                    d_max = d[mask].max()
                    if d_max > d_min:
                        d_norm[mask] = (d[mask] - d_min) / (d_max - d_min)

                # d_norm[mask] = (d[mask] - d[mask].min()) / (d[mask].max() - d[mask].min())

                d_vis = (d_norm * 255).byte().cpu().numpy()
                Image.fromarray(d_vis, mode="L").save(os.path.join(root_depth,cat,split,final_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='view selection for multiview classification & detection')
    parser.add_argument('--split', type=str)
    parser.add_argument('--batch', type=int, default='0')
    args = parser.parse_args()

    main(args)

            