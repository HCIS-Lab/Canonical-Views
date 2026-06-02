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
    def __init__(self, dataset, arch='resnet18', aggregation='max', dataset_name=''):
        super().__init__(dataset, aggregation)

        if arch == 'resnet18':
            resnet = models.resnet18(pretrained=True)

            if dataset_name in ['rgb_depth', 'rgb_edge', 'depth_edge', 'rgb_depth_edge']:
                input_channels = 6
                if dataset_name == 'rgb_depth_edge':
                    input_channels = 9

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

        else:
            raise Exception('architecture currently supports [resnet18, vit]')

        self.classifier = nn.Linear(base_dim, dataset.num_class)

        # select camera based on initialization
        self.select_module = CamSelect(dataset.num_cam, base_dim, 1, aggregation)

    def get_feat(self, imgs, M=None, down=1, visualize=False):
        B, N, _, H, W = imgs.shape
        imgs = F.interpolate(imgs.flatten(0, 1), scale_factor=1 / down)

        if isinstance(self.base, nn.Sequential):
            # CNN branch (ResNet)
            imgs_feat = self.base(imgs)
            imgs_feat = self.avgpool(imgs_feat)
            return imgs_feat.unflatten(0, [B, N]), None

        else:
            # ViT branch
            # print(imgs.shape)
            with torch.no_grad():
                feats = self.base(imgs)  # (B*N, hidden_dim)

                feats = feats.unsqueeze(-1).unsqueeze(-1)  # (B*N, C, 1, 1)
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
