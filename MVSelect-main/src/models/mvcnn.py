import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
import matplotlib.pyplot as plt
from src.models.multiview_base import MultiviewBase
from src.models.mvselect import CamSelect
from src.models.architectures import (
    ARCHITECTURE_FEATURE_DIMS,
    SUPPORTED_ARCHITECTURES,
    TINYVIT_TIMM_MODEL,
)


# class MVCNN(MultiviewBase):
#     def __init__(self, dataset, arch='resnet18', aggregation='max', dataset_name=''):
#         super().__init__(dataset, aggregation)
#         if arch == 'resnet18':
#             resnet = models.resnet18(pretrained=True)
#             if dataset_name in ['rgb_depth', 'rgb_edge', 'depth_edge', 'rgb_depth_edge']:
#                 input_channels = 6
#                 if dataset_name == 'rgb_depth_edge':
#                     input_channels = 9
#                 old_conv = resnet.conv1

#                 new_conv = nn.Conv2d(
#                     in_channels=input_channels,
#                     out_channels=old_conv.out_channels,
#                     kernel_size=old_conv.kernel_size,
#                     stride=old_conv.stride,
#                     padding=old_conv.padding,
#                     bias=False
#                 )
#                 with torch.no_grad():
#                     # RGB
#                     new_conv.weight[:, 0:3] = old_conv.weight

#                     # Mean RGB for depth
#                     mean_rgb = old_conv.weight.mean(dim=1, keepdim=True)
#                     new_conv.weight[:, 3:6] = mean_rgb.repeat(1, 3, 1, 1) * 0.5

#                     if dataset_name == 'rgb_depth_edge':
#                         new_conv.weight[:, 6:9] = mean_rgb.repeat(1, 3, 1, 1) * 0.5

#                 resnet.conv1 = new_conv
#             self.base = nn.Sequential(*list(resnet.children())[:-2])
#             # self.base = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
#             self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#             self.classifier = nn.Linear(512, dataset.num_class)
#             base_dim = 512

#         else:
#             raise Exception('architecture currently support [vgg11, resnet18]')

        

#         # select camera based on initialization
#         self.select_module = CamSelect(dataset.num_cam, base_dim, 1, aggregation)
#         pass

#     def get_feat(self, imgs, M=None, down=1, visualize=False):
#         B, N, _, H, W = imgs.shape
#         imgs = F.interpolate(imgs.flatten(0, 1), scale_factor=1 / down)
#         imgs_feat = self.base(imgs)
#         imgs_feat = self.avgpool(imgs_feat)
#         _, C, H, W = imgs_feat.shape
#         return imgs_feat.unflatten(0, [B, N]), None

#     def get_output(self, overall_feat, visualize=False):
#         overall_result = self.classifier(torch.flatten(overall_feat, 1))
#         return overall_result

class MVCNN(MultiviewBase):
    def __init__(self, dataset, arch='resnet18', aggregation='max',
                 dataset_name='', active_single_view=False):
        super().__init__(dataset, aggregation)
        self.arch = arch
        self.active_single_view = bool(active_single_view)
        input_channels = 3
        if dataset_name in ['rgb_depth', 'rgb_edge', 'depth_edge']:
            input_channels = 6
        elif dataset_name == 'rgb_depth_edge':
            input_channels = 9

        if arch == 'resnet18':
            resnet = models.resnet18(pretrained=True)

            if input_channels != 3:
                old_conv = resnet.conv1
                new_conv = nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=False
                )

                with torch.no_grad():
                    new_conv.weight[:, 0:3] = old_conv.weight
                    mean_rgb = old_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight[:, 3:6] = mean_rgb.repeat(1, 3, 1, 1) * 0.5

                    if dataset_name == 'rgb_depth_edge':
                        new_conv.weight[:, 6:9] = mean_rgb.repeat(1, 3, 1, 1) * 0.5

                resnet.conv1 = new_conv

            self.base = nn.Sequential(*list(resnet.children())[:-2])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            base_dim = 512

        elif arch == 'vit':
            vit = models.vit_b_16(pretrained=True)

            # Remove classification head
            vit.heads = nn.Identity()

            self.base = vit
            self.avgpool = None  # not needed
            base_dim = vit.hidden_dim  # typically 768

        elif arch == 'tinyvit':
            try:
                import timm
            except ImportError as ex:
                raise ImportError(
                    "TinyViT requires timm. Install MVSelect-main/requirements.txt "
                    "or run `pip install timm==0.9.16`."
                ) from ex

            self.base = timm.create_model(
                TINYVIT_TIMM_MODEL,
                pretrained=True,
                num_classes=0,
                global_pool='avg',
                in_chans=input_channels,
            )
            self.avgpool = None
            base_dim = int(self.base.num_features)
            expected_dim = ARCHITECTURE_FEATURE_DIMS["tinyvit"]
            if base_dim != expected_dim:
                raise RuntimeError(
                    f"{TINYVIT_TIMM_MODEL} returned {base_dim} features; "
                    f"expected {expected_dim}. Check the installed timm version.")

        else:
            raise ValueError(
                f"architecture currently supports {list(SUPPORTED_ARCHITECTURES)}")

        self.feature_dim = base_dim
        self.classifier = nn.Linear(base_dim, dataset.num_class)

        # select camera based on initialization
        self.select_module = CamSelect(
            dataset.num_cam,
            base_dim,
            1,
            aggregation,
            selected_only_output=self.active_single_view,
        )

    def get_feat(self, imgs, M=None, down=1, visualize=False):
        B, N, _, H, W = imgs.shape
        imgs = F.interpolate(imgs.flatten(0, 1), scale_factor=1 / down)

        if self.arch == 'resnet18':
            # CNN branch (ResNet)
            imgs_feat = self.base(imgs)
            imgs_feat = self.avgpool(imgs_feat)
            return imgs_feat.unflatten(0, [B, N]), None

        if self.arch == 'vit':
            # Preserve the existing torchvision ViT behavior.
            with torch.no_grad():
                feats = self.base(imgs)  # (B*N, hidden_dim)

                feats = feats.unsqueeze(-1).unsqueeze(-1)  # (B*N, C, 1, 1)
                return feats.unflatten(0, [B, N]), None

        # timm TinyViT returns pooled pre-logit features because num_classes=0.
        # Gradients remain enabled so the ImageNet-pretrained backbone can be
        # fine-tuned together with the classifier and selector.
        feats = self.base(imgs)
        if feats.ndim == 4:
            feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        if feats.ndim != 2:
            raise RuntimeError(
                f"Unexpected TinyViT feature shape: {tuple(feats.shape)}")
        feats = feats.unsqueeze(-1).unsqueeze(-1)
        return feats.unflatten(0, [B, N]), None

    def get_output(self, overall_feat, visualize=False):
        overall_result = self.classifier(torch.flatten(overall_feat, 1))
        return overall_result


if __name__ == '__main__':
    from src.datasets import ModelNet40
    from torch.utils.data import DataLoader
    from thop import profile
    import itertools

    dataset = ModelNet40('/home/houyz/Data/modelnet/modelnet40_images_new_12x', 12)
    dataloader = DataLoader(dataset, 1, False, num_workers=0)
    imgs, tgt, keep_cams = next(iter(dataloader))
    model = MVCNN(dataset).cuda()
    init_prob = F.one_hot(torch.tensor([0, 1]), num_classes=dataset.num_cam)
    keep_cams[0, 3] = 0
    model.train()
    res = model(imgs.cuda(), None, 2, init_prob, 3, keep_cams)
    # macs, params = profile(model, inputs=(imgs[:, :2].cuda(),))
    # macs, params = profile(model.select_module, inputs=(torch.randn([1, 12, 512, 1, 1]).cuda(),
    #                                                     F.one_hot(torch.tensor([1]), num_classes=20).cuda()))
    # macs, params = profile(model, inputs=(torch.randn([1, 512, 1, 1]).cuda(),))
    pass
