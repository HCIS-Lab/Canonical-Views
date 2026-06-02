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
def load_or_create_list(filepath='pose_table.json', non_roll=False):
    """
    Load a list from a pickle file if it exists; otherwise, create it and save it.
    """
    if not non_roll:
        if os.path.exists(filepath):
            # print(f"Loading existing list from {filepath}")
            with open(filepath, "rb") as f:
                values = json.load(f)
        else:
            # print(f"File not found, creating new list and saving to {filepath}")
            elevs = np.arange(-180.0, 180.0, 22.5).tolist() 
            azims = np.arange(-180.0, 180.0, 22.5).tolist() 
            rolls = np.arange(-180.0, 180.0, 22.5).tolist() 

            ea = []
            for e in elevs:
                for a in azims:
                    ea.append([e, a])
            values = {}
            i = 0
            for r in rolls:
                for ea_i in ea:
                    i+=1
                    k = str(e)+'_'+str(a)+'_'+str(r)
                    values[k]=i
                    # values.append(r,ea_i[0], ea_i[1])

            with open(filepath, "w") as f:
                json.dump(values, f)
        return values

    else:

        filepath = 'non_roll_pose_table.json'
        if os.path.exists(filepath):
            # print(f"Loading existing list from {filepath}")
            with open(filepath, "rb") as f:
                new_values = json.load(f)
        else:
            # print(f"File not found, creating new list and saving to {filepath}")
            elevs = np.arange(-180.0, 180.0, 22.5).tolist() 
            azims = np.arange(-180.0, 180.0, 22.5).tolist() 
            rolls = np.arange(-180.0, 180.0, 22.5).tolist() 

            ea = []
            for e in elevs:
                for a in azims:
                    ea.append([e, a])
            values = []
            for r in rolls:
                for ea_i in ea:
                    values.append([ea_i[0], ea_i[1], r])

            new_values = {}
            i = 0
            for v in values:
                if v[2] == 0.0:
                    i+=1
                    k = str(v[0]) + '_' + str(v[1]) + '_' + str(v[2])
                    # new_values.append(v)
                    new_values[k] = i
            # print(new_values)
            with open(filepath, "w") as f:
                json.dump(new_values, f)
        return new_values

def inplane_to_tilt90(inplane_deg):
    """
    Map signed in-plane angle (−180° to +180°) to [0°, 90°],
    representing the minimal tilt away from perceptual upright.

    Examples:
      0°   → 0°   (upright)
      90°  → 90°  (horizontal)
      180° → 0°   (upside down, same orientation)
      −45° → 45°
      −135°→ 45°
    """
    # Wrap into [0, 180)
    if inplane_deg is not None:
        inplane_deg = abs(inplane_deg) % 180.0
        # Fold 180→90 symmetry (vertical flip same as upright)
        if inplane_deg > 90.0:
            inplane_deg = 180.0 - inplane_deg
    return inplane_deg


class ModelNet40(VisionDataset):

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


    def __init__(self, root, dataset_name, num_cam, split='train', per_cls_instances=0, dropout=0.0, non_roll=False, non_like=False, edge=False):
        super().__init__(root)
        self.num_cam, self.num_class = num_cam, len(self.classnames)
        self.split = split
        self.transform = T.Compose([T.Resize([224, 224]), T.ToTensor(),
                                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ])
        self.dropout = dropout

        self.img_fpaths = {cam: [] for cam in range(self.num_cam)}
        print()
        self.targets = []
        # self.meta = {cam: [] for cam in range(self.num_cam)}
        self.e = {cam: [] for cam in range(self.num_cam)}
        self.a = {cam: [] for cam in range(self.num_cam)}
        self.r = {cam: [] for cam in range(self.num_cam)}
        self.id = {cam: [] for cam in range(self.num_cam)}

        self.long = {cam: [] for cam in range(self.num_cam)}
        self.short = {cam: [] for cam in range(self.num_cam)}
        self.planar_like = {cam: [] for cam in range(self.num_cam)}
        self.long_like = {cam: [] for cam in range(self.num_cam)}
        self.short_like = {cam: [] for cam in range(self.num_cam)}
        self.to_eye_deg = {cam: [] for cam in range(self.num_cam)}
        self.inplane_deg = {cam: [] for cam in range(self.num_cam)}
        # self.view_type = {cam: [] for cam in range(self.num_cam)}

        self.class_idx_dict = {cls: [] for cls in self.classnames}

        self.pose_table = {}
        pose_index_counting = 0

        for cls in self.classnames:
            # print(len(glob.glob(f'{root}/{cls}/{split}/*.png')))
            for fname in sorted(glob.glob(f'{root}/{cls}/{split}/*.png')):
                fname = os.path.basename(fname)
                # id, cam = map(int, re.findall(r'\d+', fname))
                attr = fname.split('.p')[0].split('_')
                id, cam = attr[0], int(attr[1])
                e, a, r = float(attr[2][1:]), float(attr[3][1:]), float(attr[4][1:])
                pose_string = str(e) + '_' + str(a) + '_' +str(r)
                
                to_eye_deg = float(attr[5][3:])
                # print(attr[-1][7:])
                if not attr[6][7:] == 'None':
                    inplane_deg = float(attr[6][7:]) 

                else:
                    inplane_deg = None
                if non_roll and r!=0.0:
                    continue

                is_long = 'planar' in fname and not 'short' in fname
                is_short = 'short' in fname and not 'like' in fname
                # is_planar_like = 'like' in fname and not 'short' in fname
                is_long_like = 'like' in fname and not 'short' in fname
                is_short_like = 'like' in fname and 'short' in fname

                # if is_long:
                #     view_type = 1
                # elif is_short:
                #     view_type = 2
                # elif is_long_like:
                #     view_type = 3
                # elif is_short_like:
                #     view_type = 4
                # else:
                #     view_type = 5

                if non_like:
                    if (is_long_like or is_short_like):
                        continue

                if pose_string not in self.pose_table:
                    self.pose_table[pose_string] = pose_index_counting
                    pose_index_counting += 1
                pose_index = self.pose_table[pose_string]

                if cam > self.num_cam and non_roll==False:
                    continue
                if len(self.class_idx_dict[cls]) >= (per_cls_instances if per_cls_instances else np.inf) and \
                        id not in self.class_idx_dict[cls]:

                    break
                self.img_fpaths[pose_index].append(f'{root}/{cls}/{split}/{fname}')
                self.e[pose_index].append(torch.tensor(e, dtype=torch.float32))
                self.a[pose_index].append(torch.tensor(a, dtype=torch.float32))
                self.r[pose_index].append(torch.tensor(r, dtype=torch.float32))
                self.id[pose_index].append(id)
                self.long[pose_index].append(torch.tensor(is_long))
                self.short[pose_index].append(torch.tensor(is_short))
                # self.planar_like[self.pose_table[pose_string]].append(torch.tensor(is_planar_like))
                self.long_like[pose_index].append(torch.tensor(is_long_like))
                self.short_like[pose_index].append(torch.tensor(is_short_like))
                self.to_eye_deg[pose_index].append(torch.tensor(to_eye_deg))
                # self.view_type[pose_index].append(torch.tensor(view_type))

                if inplane_deg != None:
                    self.inplane_deg[pose_index].append(torch.tensor(inplane_to_tilt90(inplane_deg)))
                else:
                    self.inplane_deg[pose_index].append(torch.tensor(1000.0))

                if id not in self.class_idx_dict[cls]:
                    self.targets.append(self.classnames.index(cls))
                    self.class_idx_dict[cls].append(id)
                    
                tabel_path = str(dataset_name) + '_pose_table.json'
                with open(tabel_path, "w") as f:
                    json.dump(self.pose_table, f)

        # for k,v in self.img_fpaths.items():
        #     if len(v)!=28*30:
        #         print(k)
        #         print(len(v))
        #         print('--------')
        # print(len(self.targets))
        assert np.prod([len(i) == len(self.targets) for i in self.img_fpaths.values()]), \
            'plz ensure all models appear {num_cam} times!'
        print(f'{split}: {self.num_class} classes, {num_cam} views, {len(self.targets)} instances')

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx, visualize=False):
        imgs = []
        a_list, e_list, r_list  = [], [], []
        long_list, short_list, long_like_list, short_like_list, to_eye_deg, inplane_deg, id_list = [], [], [], [], [], [], []
        img_fpaths = []
        for cam in range(self.num_cam):
            img = Image.open(self.img_fpaths[cam][idx]).convert('RGB')
            # if visualize:
            #     plt.imshow(img)
            #     plt.show()
            imgs.append(self.transform(img))
            # meta = self.meta[cam][idx]
            # metas.append(meta)
            a_list.append(self.a[cam][idx])
            e_list.append(self.e[cam][idx])
            r_list.append(self.r[cam][idx])

            long_list.append(self.long[cam][idx])
            short_list.append(self.short[cam][idx])
            long_like_list.append(self.long_like[cam][idx])
            short_like_list.append(self.short_like[cam][idx])
            to_eye_deg.append(self.to_eye_deg[cam][idx])
            inplane_deg.append(self.inplane_deg[cam][idx])
            id_list.append(self.id[cam][idx])
            img_fpaths.append(self.img_fpaths[cam][idx].split('/')[-1])
            # view_type_list.append(self.view_type[cam][idx])

        imgs = torch.stack(imgs)
        a_list = torch.stack(a_list)
        e_list = torch.stack(e_list)
        r_list = torch.stack(r_list)
        long_list = torch.stack(long_list)
        short_list = torch.stack(short_list)
        long_like_list = torch.stack(long_like_list)
        short_like_list = torch.stack(short_like_list)
        to_eye_deg = torch.stack(to_eye_deg)
        inplane_deg = torch.stack(inplane_deg)
        # view_type_list = torch.stack(view_type_list)

        # random_idx = np.random.randint(self.num_cam)
        # imgs = imgs[torch.cat([torch.arange(random_idx, self.num_cam), torch.arange(0, random_idx)])]
        tgt = self.targets[idx]
        
        # dropout
        drop, keep_cams = np.random.rand() < self.dropout, torch.ones(self.num_cam, dtype=torch.bool)
        if drop:
            num_drop = np.random.randint(self.num_cam - 1)
            drop_cams = np.random.choice(self.num_cam, num_drop, replace=False)
            for cam in drop_cams:
                keep_cams[cam] = 0
        # keep_cams = np.ones(self.num_cam, dtype=bool)
        # keep_cams[[0, 2, 5, 6, 8, 11]] = 0

        meta = {'e_list': e_list,
                'a_list': a_list,
                'r_list': r_list,
                'long_list': long_list,
                'short_list': short_list,
                'long_like_list': long_like_list,
                'short_like_list': short_like_list,
                'to_eye_deg': to_eye_deg,
                'inplane_deg': inplane_deg,
                'id_list': id_list,
                'img_fpaths': img_fpaths,
                # 'view_type_list': view_type_list
                }

        return imgs, tgt, keep_cams, meta


class RGB_Depth_Edge_Dataset(VisionDataset):

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


    def __init__(self, root, dataset_name, num_cam, split='train', per_cls_instances=0, dropout=0.0, non_roll=True):
        super().__init__(root)
        self.num_cam, self.num_class = num_cam, len(self.classnames)
        if 'rgb' in root:
            self.rgb = True
            root_rgb = root['rgb']
        else:
            self.rgb = False
            root_rgb = root['depth']
        if 'depth' in root:
            self.depth = True
            root_depth = root['depth']
        else:
            self.depth = False
        if 'edge' in root:
            self.edge = True
            root_edge = root['edge']
        else:
            self.edge = False

        self.split = split
        self.transform = T.Compose([T.Resize([224, 224]), T.ToTensor(),
                                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ])
        self.dropout = dropout

        self.img_fpaths = {cam: [] for cam in range(self.num_cam)}
        self.depth_fpaths = {cam: [] for cam in range(self.num_cam)}
        self.edge_fpaths = {cam: [] for cam in range(self.num_cam)}
        self.targets = []
        # self.meta = {cam: [] for cam in range(self.num_cam)}
        self.e = {cam: [] for cam in range(self.num_cam)}
        self.a = {cam: [] for cam in range(self.num_cam)}
        self.r = {cam: [] for cam in range(self.num_cam)}
        self.id = {cam: [] for cam in range(self.num_cam)}

        self.long = {cam: [] for cam in range(self.num_cam)}
        self.short = {cam: [] for cam in range(self.num_cam)}
        self.planar_like = {cam: [] for cam in range(self.num_cam)}
        self.long_like = {cam: [] for cam in range(self.num_cam)}
        self.short_like = {cam: [] for cam in range(self.num_cam)}
        self.to_eye_deg = {cam: [] for cam in range(self.num_cam)}
        self.inplane_deg = {cam: [] for cam in range(self.num_cam)}
        # self.view_type = {cam: [] for cam in range(self.num_cam)}

        self.class_idx_dict = {cls: [] for cls in self.classnames}

        self.pose_table = {}
        pose_index_counting = 0

        for cls in self.classnames:
            # print(len(glob.glob(f'{root}/{cls}/{split}/*.png')))
            for fname in sorted(glob.glob(f'{root_rgb}/{cls}/{split}/*.png')):
                fname = os.path.basename(fname)
                # id, cam = map(int, re.findall(r'\d+', fname))
                attr = fname.split('.p')[0].split('_')
                id, cam = attr[0], int(attr[1])
                e, a, r = float(attr[2][1:]), float(attr[3][1:]), float(attr[4][1:])
                pose_string = str(e) + '_' + str(a) + '_' +str(r)
                
                to_eye_deg = float(attr[5][3:])
                # print(attr[-1][7:])
                if not attr[6][7:] == 'None':
                    inplane_deg = float(attr[6][7:]) 

                else:
                    inplane_deg = None
                if non_roll and r!=0.0:
                    continue

                is_long = 'planar' in fname and not 'short' in fname and not 'like' in fname
                is_short = 'short' in fname and not 'like' in fname
                # is_planar_like = 'like' in fname and not 'short' in fname
                is_long_like = 'like' in fname and not 'short' in fname
                is_short_like = 'like' in fname and 'short' in fname

                # if is_long:
                #     view_type = 1
                # elif is_short:
                #     view_type = 2
                # elif is_long_like:
                #     view_type = 3
                # elif is_short_like:
                #     view_type = 4
                # else:
                #     view_type = 5

                if pose_string not in self.pose_table:
                    self.pose_table[pose_string] = pose_index_counting
                    pose_index_counting += 1
                pose_index = self.pose_table[pose_string]

                if cam > self.num_cam and non_roll==False:
                    continue
                if len(self.class_idx_dict[cls]) >= (per_cls_instances if per_cls_instances else np.inf) and \
                        id not in self.class_idx_dict[cls]:
                    break

                if self.rgb:
                    self.img_fpaths[pose_index].append(f'{root_rgb}/{cls}/{split}/{fname}')
                if self.depth:
                    self.depth_fpaths[pose_index].append(f'{root_depth}/{cls}/{split}/{fname}')
                if self.edge:
                    self.edge_fpaths[pose_index].append(f'{root_edge}/{cls}/{split}/{fname}')

                self.e[pose_index].append(torch.tensor(e, dtype=torch.float32))
                self.a[pose_index].append(torch.tensor(a, dtype=torch.float32))
                self.r[pose_index].append(torch.tensor(r, dtype=torch.float32))
                self.id[pose_index].append(id)
                self.long[pose_index].append(torch.tensor(is_long))
                self.short[pose_index].append(torch.tensor(is_short))
                # self.planar_like[self.pose_table[pose_string]].append(torch.tensor(is_planar_like))
                self.long_like[pose_index].append(torch.tensor(is_long_like))
                self.short_like[pose_index].append(torch.tensor(is_short_like))
                self.to_eye_deg[pose_index].append(torch.tensor(to_eye_deg))
                # self.view_type[pose_index].append(torch.tensor(view_type))
                if inplane_deg != None:
                    self.inplane_deg[pose_index].append(torch.tensor(inplane_to_tilt90(inplane_deg)))
                else:
                    self.inplane_deg[pose_index].append(torch.tensor(1000.0))

                if id not in self.class_idx_dict[cls]:
                    self.targets.append(self.classnames.index(cls))
                    self.class_idx_dict[cls].append(id)
                    


                tabel_path = str(dataset_name) + '_pose_table.json'
                with open(tabel_path, "w") as f:
                    json.dump(self.pose_table, f)

        # for k,v in self.img_fpaths.items():
        #     if len(v)!=28*30:
        #         print(k)
        #         print(len(v))
        #         print('--------')
        # print(len(self.targets))
        if self.rgb:
            assert np.prod([len(i) == len(self.targets) for i in self.img_fpaths.values()]), \
            'plz ensure all models appear {num_cam} times!'
        else:
            assert np.prod([len(i) == len(self.targets) for i in self.depth_fpaths.values()]), \
            'plz ensure all models appear {num_cam} times!'
        print(f'{split}: {self.num_class} classes, {num_cam} views, {len(self.targets)} instances')

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx, visualize=False):
        imgs, depths, edges = [], [], []
        a_list, e_list, r_list  = [], [], []
        long_list, short_list, long_like_list, short_like_list, to_eye_deg, inplane_deg, id_list = [], [], [], [], [], [], []
        img_fpaths = []
        for cam in range(self.num_cam):
            if self.rgb:
                img = Image.open(self.img_fpaths[cam][idx]).convert('RGB')
                imgs.append(self.transform(img))
            if self.depth:
                depth = Image.open(self.depth_fpaths[cam][idx]).convert('RGB')
                depths.append(self.transform(depth))
            if self.edge:
                edge = Image.open(self.edge_fpaths[cam][idx]).convert('RGB')
                edges.append(self.transform(edge))

            a_list.append(self.a[cam][idx])
            e_list.append(self.e[cam][idx])
            r_list.append(self.r[cam][idx])

            long_list.append(self.long[cam][idx])
            short_list.append(self.short[cam][idx])
            long_like_list.append(self.long_like[cam][idx])
            short_like_list.append(self.short_like[cam][idx])
            to_eye_deg.append(self.to_eye_deg[cam][idx])
            inplane_deg.append(self.inplane_deg[cam][idx])
            id_list.append(self.id[cam][idx])
            # view_type_list.append(self.view_type[cam][idx])
            if self.rgb:
                img_fpaths.append(self.img_fpaths[cam][idx].split('/')[-1])
            else:
                img_fpaths.append(self.depth_fpaths[cam][idx].split('/')[-1])
        if self.rgb:
            imgs = torch.stack(imgs)
        if self.depth:
            depths = torch.stack(depths)
        if self.edge:
            edges = torch.stack(edges)

        if self.rgb and self.depth and not self.edge:
            inputs = torch.cat((imgs, depths), dim=1)
        elif self.rgb and not self.depth and self.edge:
            inputs = torch.cat((imgs, edges), dim=1)
        elif not self.rgb and self.depth and self.edge:
            inputs = torch.cat((depths, edges), dim=1)
        elif self.rgb and self.depth and self.edge:
            inputs = torch.cat((imgs, depths, edges), dim=1)

        a_list = torch.stack(a_list)
        e_list = torch.stack(e_list)
        r_list = torch.stack(r_list)
        long_list = torch.stack(long_list)
        short_list = torch.stack(short_list)
        long_like_list = torch.stack(long_like_list)
        short_like_list = torch.stack(short_like_list)
        to_eye_deg = torch.stack(to_eye_deg)
        inplane_deg = torch.stack(inplane_deg)
        # view_type_list = torch.stack(view_type_list)

        # random_idx = np.random.randint(self.num_cam)
        # imgs = imgs[torch.cat([torch.arange(random_idx, self.num_cam), torch.arange(0, random_idx)])]
        tgt = self.targets[idx]
        
        # dropout
        drop, keep_cams = np.random.rand() < self.dropout, torch.ones(self.num_cam, dtype=torch.bool)
        if drop:
            num_drop = np.random.randint(self.num_cam - 1)
            drop_cams = np.random.choice(self.num_cam, num_drop, replace=False)
            for cam in drop_cams:
                keep_cams[cam] = 0
        # keep_cams = np.ones(self.num_cam, dtype=bool)
        # keep_cams[[0, 2, 5, 6, 8, 11]] = 0

        meta = {'e_list': e_list,
                'a_list': a_list,
                'r_list': r_list,
                'long_list': long_list,
                'short_list': short_list,
                'long_like_list': long_like_list,
                'short_like_list': short_like_list,
                'to_eye_deg': to_eye_deg,
                'inplane_deg': inplane_deg,
                'id_list': id_list,
                'img_fpaths': img_fpaths,
                # 'view_type_list': view_type_list
                }

        return inputs, tgt, keep_cams, meta


if __name__ == '__main__':
    dataset = ModelNet40('/home/houyz/Data/modelnet/modelnet40_images_new_12x', 12)
    dataset.__getitem__(0)
    dataset.__getitem__(len(dataset) - 1, visualize=True)
