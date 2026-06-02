import os
import numpy as np
import glob
import re
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from torchvision.datasets import VisionDataset
import json
import sys
import random
from collections import defaultdict



class Downstream_ModelNet40(VisionDataset):

    classnames = [
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



    def __init__(self, meta_dir, selected_view_type, start_epoch, end_epoch,\
        root, num_cam, split='train', per_cls_instances=0, dropout=0.0,
        select='agent'):
        super().__init__(root)
        self.num_cam, self.num_class = num_cam, len(self.classnames)
        self.split = split
        self.transform = T.Compose([T.Resize([224, 224]), T.ToTensor(),
                                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ])
        self.dropout = dropout

        self.img_fpaths = {cam: [] for cam in range(self.num_cam)}
        self.id = []
        self.targets = []


        self.class_idx_dict = {cls: [] for cls in self.classnames}

        self.pose_table = {}

        # choosing view types for the set
        self.view_type_list = ['Expanded', 'Expanded-like', 'Foreshortened', 'Foreshortened-like', 'Remainder']
        selected_view_type = [int(v) for v in selected_view_type]
        updated_view_type_list = []
        for selected in selected_view_type:
            updated_view_type_list.append(self.view_type_list[selected])
        self.view_type_list = updated_view_type_list

        pose_index_counting = 0

        if not 'random' in select:
            total_target_selction = {}
            for fn in os.listdir(meta_dir):
                if fn.endswith("selection.json"):
                    with open(os.path.join(meta_dir, fn), "r") as f:
                        whole_selection = json.load(f)

                    # extract target epochs of selection from a json
                    target_selection = []
                    for i in range(start_epoch, end_epoch+1):
                        target_selection.append(whole_selection[str(i)])  
                    for epoch_selection in target_selection:
                        for cls_idx in range(len(self.classnames)):
                            if not str(cls_idx) in total_target_selction:
                                total_target_selction[cls_idx] = {}
                            for view_type in self.view_type_list:
                                class_selection = epoch_selection[str(cls_idx)][view_type]
                                for fpath in class_selection:
                                    # instance name
                                    ins_name = fpath.split('_')[0]
                                    if not ins_name in total_target_selction[cls_idx].keys():
                                        total_target_selction[cls_idx][ins_name] = {}

                                    # view name
                                    if not fpath in total_target_selction[cls_idx][ins_name].keys():
                                        total_target_selction[cls_idx][ins_name][fpath] = 0
                                    total_target_selction[cls_idx][ins_name][fpath] += 1

        dataset = {}
        for cls_idx in range(len(self.classnames)):
            if not str(cls_idx) in dataset:
                dataset[str(cls_idx)] = {}
            if not 'random' in select:
                for ins in total_target_selction[cls_idx].keys():
                    top_k_keys = sorted(total_target_selction[cls_idx][ins], key=total_target_selction[cls_idx][ins].get, reverse=True)[:num_cam]
                    dataset[str(cls_idx)][ins] = top_k_keys
            else:
                all_test_imgs = sorted([f for f in os.listdir(f'{root}/{self.classnames[cls_idx]}/{split}/')])
                for fname in all_test_imgs:
                    ins = fname.split('_')[0]
                    dataset[str(cls_idx)][ins] = 1

        # to save the final dataset
        # if select == 'agent':
        #     with open(os.path.join(logdir, 'select_dataset.json'), "w") as f:
        #         json.dump(dataset, f)

        if select == 'random-fixed-all':
            shared_indices = random.sample(range(114), num_cam)
        for i, cls in enumerate(self.classnames):
            if select == 'selected_test':
                all_test_imgs = sorted([f for f in os.listdir(f'{root}/{cls}/{split}/')])
                
                split_test_imgs = defaultdict(list)
                for fname in all_test_imgs:
                    instance_id = fname.split('_')[0]
                    split_test_imgs[instance_id].append(fname)
                test_id_keys = list(split_test_imgs.keys())
            elif select == 'random-fixed-class':
                shared_indices = random.sample(range(114), num_cam)

            for j, id in enumerate(dataset[str(i)].keys()):
                if len(self.class_idx_dict[cls]) >= (per_cls_instances if per_cls_instances else np.inf) and \
                        id not in self.class_idx_dict[cls]:
                    break

                if select == 'agent':
                    imgs = dataset[str(i)][id]
                    if len(imgs) < num_cam:
                        num_short = num_cam - len(imgs)
                        padded_imgs = [f for f in os.listdir(f'{root}/{cls}/{split}/') if f.startswith(id)]
                        padded_imgs = random.sample(padded_imgs, num_short)
                        imgs = imgs + padded_imgs
                elif select == 'random':
                    imgs = [f for f in os.listdir(f'{root}/{cls}/{split}/') if f.startswith(id)]
                    imgs = random.sample(imgs, num_cam)
                elif select == 'random-fixed-all' or select == 'random-fixed-class':
                    imgs = [f for f in os.listdir(f'{root}/{cls}/{split}/') if f.startswith(id)]
                    imgs = [imgs[i] for i in shared_indices]
                # selected test set
                elif select == 'selected_test':
                    if j>4:
                        break
                    select_train = dataset[str(i)][id]
                    imgs = []

                    for img in select_train:
                        attr = img.split('.p')[0].split('_')
                        e, a, r = float(attr[2][1:]), float(attr[3][1:]), float(attr[4][1:])
                        test_imgs = split_test_imgs[test_id_keys[j]]
                        for test_img in test_imgs:
                            attr_t = test_img.split('.p')[0].split('_')
                            e_t, a_t, r_t = float(attr_t[2][1:]), float(attr_t[3][1:]), float(attr_t[4][1:])
                            if e==e_t and a==a_t and r==r_t:
                                imgs.append(test_img)

                for img_i, img in enumerate(imgs):
                    self.img_fpaths[img_i].append(f'{root}/{cls}/{split}/{img}')
                    if not id in self.class_idx_dict[cls]:
                        self.targets.append(self.classnames.index(cls))
                        self.class_idx_dict[cls].append(id)
                        self.id.append(str(cls)+'_'+str(id))

        for k,v in self.img_fpaths.items():
            if len(v)!=32*25:
                print(k)
                print(len(v))
                print('--------')
        print(len(self.targets))

        assert np.prod([len(i) == len(self.targets) for i in self.img_fpaths.values()]), \
            'plz ensure all models appear {num_cam} times!'
        print(f'{split}: {self.num_class} classes, {num_cam} views, {len(self.targets)} instances')

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx, visualize=False):
        imgs = []
        for cam in range(self.num_cam):
            img = Image.open(self.img_fpaths[cam][idx]).convert('RGB')
            imgs.append(self.transform(img))

        imgs = torch.stack(imgs)
        tgt = self.targets[idx]
        

        drop, keep_cams = np.random.rand() < self.dropout, torch.ones(self.num_cam, dtype=torch.bool)
        if drop:
            num_drop = np.random.randint(self.num_cam - 1)
            drop_cams = np.random.choice(self.num_cam, num_drop, replace=False)
            for cam in drop_cams:
                keep_cams[cam] = 0

        # dropout
        return imgs, tgt, keep_cams, self.id[idx]


if __name__ == '__main__':
    dataset = ModelNet40('/home/houyz/Data/modelnet/modelnet40_images_new_12x', 12)
    dataset.__getitem__(0)
    dataset.__getitem__(len(dataset) - 1, visualize=True)
