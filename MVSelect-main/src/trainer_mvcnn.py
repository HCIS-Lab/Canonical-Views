import itertools
import random
import time
import copy
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from src.utils.meters import AverageMeter
from src.trainer import BaseTrainer, find_instance_lvl_strategy, find_dataset_lvl_strategy
from src.utils.image_utils import add_heatmap_to_image, img_color_denormalize
from src.models.mvselect import aggregate_feat, get_eps_thres
import math


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
def aggregate_feat(feat, selection, aggregation='mean'):
    if selection is None:
        overall_feat = feat.mean(dim=1) if aggregation == 'mean' else feat.max(dim=1)[0]
    else:
        selection = selection.bool().to(feat.device)
        overall_feat = feat * selection[:, :, None, None, None]
        if aggregation == 'mean':
            overall_feat = overall_feat.sum(dim=1) / (selection.sum(dim=1).view(-1, 1, 1, 1) + 1e-8)
        elif aggregation == 'max':
            overall_feat = overall_feat.max(dim=1)[0]
        else:
            raise Exception
    return overall_feat

class ClassifierTrainer(BaseTrainer):
    def __init__(self, model, logdir, args, ):
        super(ClassifierTrainer, self).__init__(model, logdir, args, )
        self._selector_limit_warned = False

    def selector_keep_cams(self, keep_cams, meta):
        """Apply optional view-family restriction only to MVSelect actions.

        The initial view is not restricted here; this mask is used as the
        candidate pool for additional selector actions. Baselines such as
        random_select_test and restricted_view_test intentionally do not call
        this helper.
        """
        limit = getattr(self.args, "selector_view_limit", "all")
        if limit == "all":
            return keep_cams

        device = keep_cams.device if torch.is_tensor(keep_cams) else "cuda"
        keep = keep_cams.to(device).bool()
        is_expanded = meta['long_list'].to(device).bool()
        is_expanded_like = meta['long_like_list'].to(device).bool()
        is_foreshortened = meta['short_list'].to(device).bool()
        is_foreshortened_like = meta['short_like_list'].to(device).bool()

        if limit == "expanded_family":
            allowed = is_expanded | is_expanded_like
        elif limit == "foreshortened_family":
            allowed = is_foreshortened | is_foreshortened_like
        elif limit == "foreshortened_family_remainder":
            allowed = ~(is_expanded | is_expanded_like)
        elif limit == "remainder":
            allowed = ~(is_expanded | is_expanded_like |
                        is_foreshortened | is_foreshortened_like)
        else:
            raise ValueError(f"Unknown selector_view_limit: {limit}")

        limited = keep & allowed
        empty = ~limited.any(dim=1)
        if empty.any():
            # If dropcam removed every view from the requested family, re-enable
            # that family so the selector still obeys the family restriction.
            # Only fall back to the original keep_cams when the dataset truly
            # has no view from the requested family for an instance.
            allowed_empty = empty & ~allowed.any(dim=1)
            limited[empty & ~allowed_empty] = allowed[empty & ~allowed_empty]
            limited[allowed_empty] = keep[allowed_empty]
            if not self._selector_limit_warned:
                print(f"WARNING: selector_view_limit={limit} had no valid "
                      f"candidate after keep_cams filtering for at least one "
                      f"instance; re-enabled that family when present, and "
                      f"fell back to original keep_cams only when absent.")
                self._selector_limit_warned = True
        return limited

    def task_loss_reward(self, init_feature, init_prob, overall_feat, tgt, step):
        if step ==0:
            init_feature = aggregate_feat(init_feature, init_prob, aggregation='max')
            init_output = self.model.get_output(init_feature)
            init_loss = F.cross_entropy(init_output, tgt, reduction='none')
            self.last_loss = init_loss
        output = self.model.get_output(overall_feat)
        task_loss = F.cross_entropy(output, tgt, reduction='none')
        # reward = torch.zeros_like(task_loss) if step < self.args.steps - 1 else (output.argmax(1) == tgt).float()
        # print(self.last_loss.shape)
        # print(task_loss.shape)
        # if step == 0:
        #     reward = torch.zeros_like(task_loss)
        # else:
        reward = (self.last_loss - task_loss).detach()
        self.last_loss = task_loss.detach()
        return task_loss, reward

    def train(self, epoch, dataloader, optimizer, scheduler=None, log_interval=200, all_cameras=False, freeze_epoch=100, freeze_backbone=False):
        self.model.train()
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if 'base' in name:
                    param.requires_grad = False
                
        if self.args.steps:
            if epoch > freeze_epoch:
                print('--------freeze base and select')
                for name, param in self.model.named_parameters():
                    if 'select' in name:
                        param.requires_grad = False
            else:
                for name, param in self.model.named_parameters():
                    if 'base' in name or 'select' in name:
                        param.requires_grad = True

        if self.args.base_lr_ratio == 0:
            self.model.base.eval()
        losses, correct, miss = 0, 0, 1e-8
        longest, short, longest_like, short_like, num_selections = 0, 0, 0, 0, 0

        t0 = time.time()
        action_sum = torch.zeros([dataloader.dataset.num_cam]).cuda()
        return_avg = None
        for batch_idx, (imgs, tgt, keep_cams, meta) in enumerate(dataloader):
            B, N = imgs.shape[:2]
            imgs, tgt = imgs.cuda(), tgt.cuda()

            # Random-view subsampling: pick K random views per instance,
            # re-sampled every batch. Saves compute when N is large (e.g.
            # N=114 → K=5). Stage 2 ignores this flag.
            #
            # NOTE: this OVERRIDES dropcam for the K sampled positions. The
            # dataloader still emits a dropcam'd keep_cams (some entries may be
            # False), but for the K random views we forcibly mark them all
            # active so the aggregator uses all K. Otherwise sampled positions
            # could land on dropcam'd indices and collapse to zero in the mean.
            # If you want dropcam too, set --dropcam 0.
            train_K = getattr(self.args, "train_num_views", None)
            if (not self.args.steps) and train_K is not None and self.model.training and train_K < N:
                # idx: [B, K] independent random indices per batch item.
                idx = torch.argsort(torch.rand(B, N, device=imgs.device), dim=1)[:, :train_K]
                idx_imgs = idx.view(B, train_K, 1, 1, 1).expand(-1, -1, *imgs.shape[2:])
                imgs = imgs.gather(1, idx_imgs)
                keep_cams = torch.ones(B, train_K, dtype=torch.bool, device=imgs.device)
                N = train_K

            feat, _ = self.model.get_feat(imgs, None, self.args.down)
            if self.args.steps:
                eps_thres = get_eps_thres(epoch - 1 + batch_idx / len(dataloader), self.args.epochs)
                selector_keep_cams = self.selector_keep_cams(keep_cams, meta)
                loss, (action_sum, return_avg, value_loss) = \
                    self.expand_episode(feat, selector_keep_cams, tgt, eps_thres, (action_sum, return_avg), all_cameras=all_cameras)
            else:
                overall_feat = aggregate_feat(feat, keep_cams, self.model.aggregation)
                output = self.model.get_output(overall_feat)
                loss = F.cross_entropy(output, tgt)

                pred = torch.argmax(output, 1)
                correct += (pred == tgt).sum().item()
                miss += B - (pred == tgt).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses += loss.item()

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
                elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts) or \
                        isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
                    scheduler.step(epoch - 1 + batch_idx / len(dataloader))
            # logging
            if (batch_idx + 1) % log_interval == 0 or batch_idx + 1 == len(dataloader):
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print(f'Train epoch: {epoch}, batch:{(batch_idx + 1)}, '
                      f'loss: {losses / (batch_idx + 1):.3f}, time: {t_epoch:.1f}')
                if self.args.steps:
                    print(f'value loss: {value_loss:.3f}, eps: {eps_thres:.3f}, return: {return_avg[-1]:.2f}')
                    # print(f'value loss: {value_loss:.3f}, policy loss: {policy_loss:.3f}, '
                    #       f'return: {return_avg[-1]:.2f}, entropy: {entropies.mean():.3f}')
                    # print(' '.join('cam {} {:.2f} |'.format(cam, freq) for cam, freq in
                    #                zip(range(N), F.normalize(action_sum, p=1, dim=0).cpu())))
                pass
        return losses / len(dataloader), None if self.args.steps else correct / (correct + miss) * 100.0

    def test(self, dataloader, init_cam=None, return_miss_ins=False, feature_file_name=False, epoch=0):
        t0 = time.time()
        self.model.eval()

        # planar = 0
        out_meta = {'longest': 0, 'short': 0, 'longest_like': 0, 'short_like': 0, \
                        'to_eye_deg_bar': [0]*18, 'inplane_deg_bar': [0]*19}

        view_type_map = {1:'Expanded', 2: 'Foreshortened',
                        3:'Expanded-like', 4: 'Foreshortened-like',
                        5: 'Remainder'}
        select_json = {}
        miss_ins = {classname: [] for classname in classnames}
        num_selections = 0
        K = len(init_cam) if init_cam is not None else 1
        selection = torch.zeros([K]).cuda()
        losses, correct, miss = torch.zeros([K]), torch.zeros([K]), torch.zeros([K]) + 1e-8
        action_sum = torch.zeros([K, dataloader.dataset.num_cam]).cuda()
        
        num_classes = 32  # set this properly
        per_class_correct = torch.zeros(K, num_classes)
        per_class_total = torch.zeros(K, num_classes)

        feature_save = []
        view_type_save = []
        view_index_save = []
        view_class_save = []
        selected_feature_save = []
        selected_class_save = []
        selected_instance_save = []
        selected_init_cam_save = []
        selected_mask_save = []

        for batch_idx, (imgs, tgt, keep_cams, meta) in enumerate(dataloader):
            B, N = imgs.shape[:2]
            imgs, tgt = imgs.cuda(), tgt.cuda()
            outputs, actions = [], []
            feat = None

            # print('--------------------')
            # print(elf.args.steps)
            # print(init_cam)
            # print('--------------------')
            with torch.no_grad():

                if self.args.steps == 0 or init_cam is None:
                    if feature_file_name and (epoch%10==0 or epoch < 20):
                        feat, _ = self.model.get_feat(imgs, None, self.args.down)
                        overall_feat = aggregate_feat(feat, keep_cams, self.model.aggregation)
                        output = self.model.get_output(overall_feat)
                        action = None
                    else:
                        output, _, (_, _, action, _) = self.model(imgs, None, self.args.down)
                    # print('---------')
                    # print(output)
                    # print('---------')
                    # action [B,N]
                    outputs.append(output)
                    actions.append(action)
                else:

                    feat, _ = self.model.get_feat(imgs, None, self.args.down)
                    selector_keep_cams = self.selector_keep_cams(keep_cams, meta)
                    # K, B, N
                    for k in range(K):
                        overall_feat, (_, _, action, _) = \
                            self.model.do_steps(feat, init_cam[k].repeat([B, 1]), self.args.steps, selector_keep_cams)
                        output = self.model.get_output(overall_feat)
                        outputs.append(output)
                        actions.append(action)
                        if feature_file_name and (epoch%10==0 or epoch < 20):
                            if getattr(self.args, 'active_single_view', False):
                                selected_mask = torch.zeros(
                                    [B, N], dtype=torch.bool,
                                    device=feat.device)
                            else:
                                selected_mask = init_cam[k].repeat(
                                    [B, 1]).to(feat.device).bool()
                            for step_action in action:
                                selected_mask = selected_mask | step_action.bool()
                            initial_cam_idx = int(init_cam[k].argmax().item())
                            for b in range(B):
                                selected_feature_save.append(overall_feat[b].detach().cpu().numpy())
                                selected_class_save.append(np.array([tgt[b].item()]))
                                selected_instance_save.append(np.array([meta['id_list'][0][b]]))
                                selected_init_cam_save.append(np.array([initial_cam_idx]))
                                selected_mask_save.append(selected_mask[b].detach().cpu().numpy())
                        # print(K) 20
                        # print(N) 20
                        # print(B) 8
                        # # print(len(action)) 2 (steps)
                        # # print(action[0].shape) [8,20]
                        # print(action[1].shape) [8,20]
                        num_selections += len(action)*B
                        for s in range(len(action)):
                            for b in range(B):
                                idx = action[s][b].to(torch.int64).argmax(dim=-1).item() 
                                classname = tgt[b].item()
                                instance_id = meta['id_list'][idx][b]

                                selection += action[s][b]
                                if not classname in select_json:
                                    select_json[classname] = {'Expanded': [],
                                                                'Foreshortened': [],
                                                                'Expanded-like': [],
                                                                'Foreshortened-like': [],
                                                                'Remainder': []}
                                # if not instance_id in select_json[classname]:
                                #     select_json[classname][instance_id] = []
                                # select_json[classname][instance_id].append(
                                #     {
                                #     'e': meta['e_list'][b][idx].item(),
                                #     'a': meta['a_list'][b][idx].item(),
                                #     'r': meta['r_list'][b][idx].item(),
                                #     }
                                #     )

                                is_longest = meta['long_list'][b][idx].item()
                                is_short = meta['short_list'][b][idx].item()
                                is_longest_like = meta['long_like_list'][b][idx].item()
                                is_short_like = meta['short_like_list'][b][idx].item()

                                if is_short:
                                    out_meta['short']+=1
                                    select_json[classname]['Foreshortened'].append(meta['img_fpaths'][idx][b])
                                elif is_longest:
                                    out_meta['longest']+=1
                                    select_json[classname]['Expanded'].append(meta['img_fpaths'][idx][b])
                                elif is_short_like:
                                    out_meta['short_like']+=1
                                    select_json[classname]['Foreshortened-like'].append(meta['img_fpaths'][idx][b])
                                elif is_longest_like:
                                    out_meta['longest_like']+=1
                                    select_json[classname]['Expanded-like'].append(meta['img_fpaths'][idx][b])
                                else:
                                    select_json[classname]['Remainder'].append(meta['img_fpaths'][idx][b])


                                # [85,90]
                                if int(meta['to_eye_deg'][b][idx]//10) == 9:
                                    out_meta['to_eye_deg_bar'][8] +=1
                                else:
                                    # e.g., [0,5)
                                    out_meta['to_eye_deg_bar'][int(meta['to_eye_deg'][b][idx]//10)] +=1

                                # invalid plane
                                if meta['inplane_deg'][b][idx]/10 > 9.0:
                                    out_meta['inplane_deg_bar'][9] +=1
                                # [85,90]
                                elif meta['inplane_deg'][b][idx]/10 == 9.0:
                                    out_meta['inplane_deg_bar'][8] +=1
                                # e.g., [80,85)
                                else:
                                    out_meta['inplane_deg_bar'][int(meta['inplane_deg'][b][idx]//10)] +=1

            if feature_file_name and (epoch%10==0 or epoch < 20) and feat is not None:
                for view_idx in range(N):
                    for b in range(B):
                        feature_save.append(feat[b, view_idx].cpu().numpy())
                        view_index_save.append(np.array([view_idx]))
                        view_class_save.append(np.array([tgt[b].item()]))
                        is_longest = meta['long_list'][b][view_idx].item()
                        is_short = meta['short_list'][b][view_idx].item()
                        is_longest_like = meta['long_like_list'][b][view_idx].item()
                        is_short_like = meta['short_like_list'][b][view_idx].item()
                        # ['Expanded', 'Expanded-like', 'Foreshortened', 'Foreshortened-like', 'Remainder']
                        if is_longest:
                            view_type_save.append(np.array([0]))
                        elif is_longest_like:
                            view_type_save.append(np.array([1]))
                        elif is_short:
                            view_type_save.append(np.array([2]))
                        elif is_short_like:
                            view_type_save.append(np.array([3]))
                        else:
                            view_type_save.append(np.array([4]))

            for k in range(K):
                loss = F.cross_entropy(outputs[k], tgt)
                if init_cam is not None:
                    # record actions
                    action_sum[k] += torch.cat(actions[k]).sum(dim=0)
                losses[k] += loss.item()

                pred = torch.argmax(outputs[k], 1)
                correct_mask = (pred == tgt)

                correct[k] += (pred == tgt).sum().item()
                miss[k] += B - (pred == tgt).sum().item()
                for c in range(32):
                    class_mask = (tgt == c)
                    per_class_total[k, c] += class_mask.sum().item()
                    per_class_correct[k, c] += (correct_mask & class_mask).sum().item()
                # for b in range(B):
                #     if pred[b]!=tgt[b]:
                #         # print(B)
                #         # print(b)
                #         # print(len(meta))
                #         # print('-----')
                #         cls_name = meta['img_fpaths'][b].split('/')[-3]
                #         miss_ins[cls_name].append(meta['id_list'][b])

        if feature_file_name and (epoch%10==0 or epoch < 20):
            feature_save = np.concatenate(feature_save, axis=0)
            view_index_save = np.concatenate(view_index_save, axis=0)
            view_type_save = np.concatenate(view_type_save, axis=0)
            view_class_save = np.concatenate(view_class_save, axis=0)
            save_kwargs = dict(
                features=feature_save,
                view_index=view_index_save,
                view_type=view_type_save,
                view_class=view_class_save,
                recognition_input=np.asarray(
                    'selected_view_only'
                    if getattr(self.args, 'active_single_view', False)
                    else 'initial_plus_selected_views'
                ),
            )
            if selected_feature_save:
                save_kwargs.update(
                    selected_features=np.concatenate(selected_feature_save, axis=0),
                    selected_class=np.concatenate(selected_class_save, axis=0),
                    selected_instance=np.concatenate(selected_instance_save, axis=0),
                    selected_init_cam=np.concatenate(selected_init_cam_save, axis=0),
                    selected_mask=np.stack(selected_mask_save, axis=0),
                )
            np.savez(feature_file_name, **save_kwargs)

        for k in range(K):
            if init_cam is not None:
                # print(f'init camera {init_cam[k].nonzero()[0].item()}: MVSelect')
                idx = action_sum[k].nonzero().cpu()[:, 0]
                # print(' '.join('cam {} {:.2f} |'.format(cam, freq) for cam, freq in
                #                zip(idx, F.normalize(action_sum[k], p=1, dim=0).cpu()[idx])))

            # print(f'Test, loss: {losses[k] / len(dataloader):.3f}, prec: {correct[k] / (correct[k] + miss[k]):.2%}' +
            #       ('' if init_cam is not None else f', time: {time.time() - t0:.1f}s'))

        
        prec = correct / (correct + miss)
        print('*************************************')
        if init_cam is not None:
            print(f'MVSelect average prec {prec.mean()*100:.1f}±{prec.std()*100:.1f}%, time: {time.time() - t0:.1f}')
        else:
            print(f'All views average prec {prec.mean()*100:.1f}±{prec.std()*100:.1f}%, time: {time.time() - t0:.1f}')
        print('*************************************')

        total_correct = per_class_correct.sum(dim=0)
        total_samples = per_class_total.sum(dim=0)

        per_class_acc = total_correct / total_samples.clamp(min=1)
        per_class_acc *= 100.0
        per_class_acc = [tensor.item() for tensor in per_class_acc]

        if return_miss_ins:
            return losses.mean() / len(dataloader), [correct.sum() / (correct + miss).sum() * 100.0, ], per_class_acc, (out_meta, selection, num_selections, select_json), miss_ins
        else:
            return losses.mean() / len(dataloader), [correct.sum() / (correct + miss).sum() * 100.0, ], per_class_acc, (out_meta, selection, num_selections, select_json)


    def random_select_test(self, dataloader, epoch=0, num_selections=5):
        t0 = time.time()
        self.model.eval()

        K = 10
        selection = torch.zeros([K]).cuda()
        losses, correct, miss = torch.zeros([K]), torch.zeros([K]), torch.zeros([K]) + 1e-8
        action_sum = torch.zeros([K, dataloader.dataset.num_cam]).cuda()
        
        num_classes = 32  # set this properly
        per_class_correct = torch.zeros(K, num_classes)
        per_class_total = torch.zeros(K, num_classes)


        for batch_idx, (imgs, tgt, keep_cams, meta) in enumerate(dataloader):
            B, N = imgs.shape[:2]
            imgs, tgt = imgs.cuda(), tgt.cuda()
            outputs, actions = [], []

            with torch.no_grad():
                for k in range(K):

                    idx = torch.randperm(N)[:num_selections]  # (5,)
                    imgs_sampled = imgs[:, idx]
                    output, _, (_, _, action, _) = self.model(imgs_sampled, None, self.args.down)
                    outputs.append(output)
                    actions.append(action)

            for k in range(K):
                # loss = F.cross_entropy(outputs[k], tgt)
                    # record actions
                # action_sum[k] += torch.cat(actions[k]).sum(dim=0)
                # losses[k] += loss.item()

                pred = torch.argmax(outputs[k], 1)
                correct_mask = (pred == tgt)

                correct[k] += (pred == tgt).sum().item()
                miss[k] += B - (pred == tgt).sum().item()
                for c in range(32):
                    class_mask = (tgt == c)
                    per_class_total[k, c] += class_mask.sum().item()
                    per_class_correct[k, c] += (correct_mask & class_mask).sum().item()


        # for k in range(K):
        #     # if init_cam is not None:
        #         # print(f'init camera {init_cam[k].nonzero()[0].item()}: MVSelect')
        #     idx = action_sum[k].nonzero().cpu()[:, 0]
                # print(' '.join('cam {} {:.2f} |'.format(cam, freq) for cam, freq in
                #                zip(idx, F.normalize(action_sum[k], p=1, dim=0).cpu()[idx])))

            # print(f'Test, loss: {losses[k] / len(dataloader):.3f}, prec: {correct[k] / (correct[k] + miss[k]):.2%}' +
            #       ('' if init_cam is not None else f', time: {time.time() - t0:.1f}s'))

        # if init_cam is not None:
        prec = correct / (correct + miss)
        print('*************************************')
        print(f'Random Selection average prec {prec.mean()*100:.1f}±{prec.std()*100:.1f}%, time: {time.time() - t0:.1f}')
        print('*************************************')

        total_correct = per_class_correct.sum(dim=0)
        total_samples = per_class_total.sum(dim=0)

        per_class_acc = total_correct / total_samples.clamp(min=1)
        per_class_acc *= 100.0
        per_class_acc = [tensor.item() for tensor in per_class_acc]

        return 0, [correct.sum() / (correct + miss).sum() * 100.0, ], per_class_acc, (_, _, _, _)

    def restricted_view_test(self, dataloader, epoch=0, num_views=1, view_type="longest", n_runs=10):
        t0 = time.time()
        self.model.eval()

        K = n_runs

        selection = torch.zeros([K]).cuda()
        losses, correct, miss = torch.zeros([K]), torch.zeros([K]), torch.zeros([K]) + 1e-8
        # action_sum = torch.zeros([K, dataloader.dataset.num_cam]).cuda()

        num_classes = 32
        per_class_correct = torch.zeros(K, num_classes)
        per_class_total = torch.zeros(K, num_classes)

        for batch_idx, (imgs, tgt, keep_cams, meta) in enumerate(dataloader):

            B, N = imgs.shape[:2]
            imgs, tgt = imgs.cuda(), tgt.cuda()

            outputs, actions = [], []

            with torch.no_grad():

                for k in range(K):

                    sampled_imgs = []

                    for b in range(B):

                        valid_idx = []

                        for idx in range(N):

                            is_longest = meta['long_list'][b][idx].item()
                            is_short = meta['short_list'][b][idx].item()
                            is_longest_like = meta['long_like_list'][b][idx].item()
                            is_short_like = meta['short_like_list'][b][idx].item()

                            if view_type == "longest":
                                cond = is_longest

                            elif view_type == "short":
                                cond = is_short

                            elif view_type == "long_like":
                                cond = is_longest_like

                            elif view_type == "short_like":
                                cond = is_short_like

                            elif view_type == "remainder":
                                cond = not (is_longest or is_short or is_longest_like or is_short_like)

                            else:
                                raise ValueError("Unknown view type")

                            if cond:
                                valid_idx.append(idx)

                        # if no candidate exists fall back to random
                        if len(valid_idx) == 0:
                            if num_views == 1:
                                chosen = torch.randint(0, N, (num_views,)).item()
                            else:
                                chosen = torch.randint(0, N, (num_views,))
                        else:
                            if num_views == 1:
                                chosen = valid_idx[torch.randint(0, len(valid_idx), (num_views,)).item()]
                            else:
                                valid_idx = torch.tensor(valid_idx, device=imgs.device)

                                chosen = valid_idx[torch.randint(0, len(valid_idx), (num_views,))]

                        sampled_imgs.append(imgs[b, chosen])

                    imgs_sampled = torch.stack(sampled_imgs)  
                    # print(imgs_sampled.shape)
                    if num_views == 1:
                        imgs_sampled = imgs_sampled.unsqueeze(1)  # B x 1 x C x H x W

                    output, _, (_, _, action, _) = self.model(imgs_sampled, None, self.args.down)

                    outputs.append(output)
                    actions.append(action)

            for k in range(K):

                # action_sum[k] += torch.cat(actions[k]).sum(dim=0)

                pred = torch.argmax(outputs[k], 1)
                correct_mask = (pred == tgt)

                correct[k] += correct_mask.sum().item()
                miss[k] += B - correct_mask.sum().item()

                for c in range(num_classes):

                    class_mask = (tgt == c)

                    per_class_total[k, c] += class_mask.sum().item()
                    per_class_correct[k, c] += (correct_mask & class_mask).sum().item()

        prec = correct / (correct + miss)

        print('*************************************')
        print(f'{view_type} {int(num_views)}-view average prec {prec.mean()*100:.1f}±{prec.std()*100:.1f}%, time: {time.time() - t0:.1f}')
        print('*************************************')

        total_correct = per_class_correct.sum(dim=0)
        total_samples = per_class_total.sum(dim=0)

        per_class_acc = total_correct / total_samples.clamp(min=1)
        per_class_acc *= 100.0
        per_class_acc = [tensor.item() for tensor in per_class_acc]

        # Per-run details (4th tuple position): per-K accuracy (% as numpy)
        # and per-K, per-class accuracy (% as numpy of shape [K, num_classes]).
        prec_per_run = (prec * 100.0).cpu().numpy()
        per_class_per_run = (per_class_correct
                             / per_class_total.clamp(min=1) * 100.0).cpu().numpy()

        return 0, [correct.sum() / (correct + miss).sum() * 100.0], per_class_acc, \
               (prec_per_run, per_class_per_run, None, None)
    def test_cam_combination(self, dataloader, step=0):
        self.model.eval()
        t0 = time.time()
        candidates = np.eye(dataloader.dataset.num_cam)
        combinations = np.array(list(itertools.combinations(candidates, step + 1))).sum(1)
        K, N = combinations.shape
        loss_s, pred_s, gt_s = [], [], []
        for batch_idx, (imgs, tgt, keep_cams, meta) in enumerate(dataloader):
            B, N = imgs.shape[:2]
            gt_s.append(tgt)
            tgt = tgt.unsqueeze(0).repeat([K, 1])
            # K, B, N
            with torch.no_grad():
                output, _ = self.model.forward_combination(imgs.cuda(), None, self.args.down, combinations, keep_cams)
            loss = F.cross_entropy(output.flatten(0, 1), tgt.flatten(0, 1).cuda(), reduction="none")
            pred = torch.argmax(output, -1)
            loss_s.append(loss.unflatten(0, [K, B]).cpu())
            pred_s.append(pred.cpu())
        loss_s, pred_s, gt_s = torch.cat(loss_s, 1), torch.cat(pred_s, 1), torch.cat(gt_s)
        # K, num_frames
        tp_s = (pred_s == gt_s[None, :]).float()
        # instance level selection
        instance_lvl_strategy = find_instance_lvl_strategy(tp_s, combinations)
        instance_lvl_oracle = np.take_along_axis(tp_s, instance_lvl_strategy, axis=0).mean(1).numpy()[:, None]
        # dataset level selection
        keep_cam_idx = combinations[:, keep_cams[0].bool().numpy()].sum(1).astype(bool)
        dataset_lvl_prec = tp_s.mean(1).numpy()[:, None]
        dataset_lvl_strategy = find_dataset_lvl_strategy(dataset_lvl_prec, combinations)
        dataset_lvl_best_prec = dataset_lvl_prec[dataset_lvl_strategy]
        oracle_info = f'{step} steps, averave acc {dataset_lvl_prec[keep_cam_idx].mean()*100:.1f}±{dataset_lvl_prec[keep_cam_idx].std()*100:.1f}%, ' \
                      f'dataset lvl best {dataset_lvl_best_prec.mean()*100:.1f}±{dataset_lvl_best_prec.std()*100:.1f}%, ' \
                      f'instance lvl oracle {instance_lvl_oracle.mean()*100:.1f}±{instance_lvl_oracle.std()*100:.1f}%, time: {time.time() - t0:.1f}s'
        print(oracle_info)
        return loss_s.mean(1).numpy(), dataset_lvl_prec * 100.0, instance_lvl_oracle * 100.0, oracle_info
