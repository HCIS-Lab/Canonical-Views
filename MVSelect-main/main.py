import os

os.environ['OMP_NUM_THREADS'] = '1'
import time
import itertools
import argparse
import sys
import shutil
from distutils.dir_util import copy_tree
import datetime
import tqdm
import random
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from src.datasets import *
from src.models.mvdet import MVDet
from src.models.mvcnn import MVCNN
from src.utils.logger import Logger
from src.utils.draw_curve import draw_curve, plot
from src.utils.str2bool import str2bool
from src.trainer import PerspectiveTrainer, find_dataset_lvl_strategy
from src.trainer_mvcnn import ClassifierTrainer
from torch.utils.data import DataLoader, default_collate
import torch.nn as nn
import json
from pathlib import Path
from filelock import FileLock

def main(args):
    # check if in debug mode
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        print('Hmm, Big Debugger is watching me')
        is_debug = True
        torch.autograd.set_detect_anomaly(True)
    else:
        print('No sys.gettrace')
        is_debug = False

    # seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # deterministic
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
    else:
        torch.backends.cudnn.benchmark = True

    # dataset
    if  args.dataset == 'rgb':
        fpath = os.path.expanduser('/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_23')
        

    elif args.dataset == 'edge':
        fpath = os.path.expanduser('/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_edge_1_23_10.0')

    elif args.dataset == 'depth':
        fpath = os.path.expanduser('/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/depth_modelnet_32_60_1_23')
    else:
        fpath = {}
        if 'rgb' in args.dataset:
            fpath['rgb'] = os.path.expanduser('/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_23')
        if 'depth' in args.dataset:
            fpath['depth'] =os.path.expanduser('/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/depth_modelnet_32_60_1_23')
        if 'edge' in args.dataset:
            fpath['edge'] = os.path.expanduser('/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_edge_1_23_10.0')
     

    if args.non_roll:
        if args.non_like:
            num_cam = 192
        else:
            # num_cam = 256
            num_cam = 114
    else:
        num_cam = 4096

    args.task = 'mvcnn'
    result_type = ['prec']
    args.lr = 5e-5 if args.lr is None else args.lr
    args.select_lr = 1e-4 if args.select_lr is None else args.select_lr
    # args.batch_size = 8 if args.batch_size is None else args.batch_size
    args.batch_size = 6 if args.batch_size is None else args.batch_size
    if not args.dataset in ['rgb_depth', 'rgb_edge', 'depth_edge', 'rgb_depth_edge']:
        args.num_workers = 8
        train_set = ModelNet40(fpath, args.dataset, num_cam, split='train', dropout=args.dropcam, per_cls_instances=args.num_train_instances, non_roll=args.non_roll, non_like=args.non_like)
        val_set = ModelNet40(fpath, args.dataset, num_cam, split='train', per_cls_instances=5, non_roll=args.non_roll, non_like=args.non_like)
        test_set = ModelNet40(fpath, args.dataset, num_cam, split='test', per_cls_instances=5, non_roll=args.non_roll, non_like=args.non_like)
    else:
        args.num_workers = 12
        train_set = RGB_Depth_Edge_Dataset(fpath, args.dataset, num_cam, split='train', dropout=args.dropcam, per_cls_instances=args.num_train_instances, non_roll=args.non_roll)
        val_set = RGB_Depth_Edge_Dataset(fpath, args.dataset, num_cam, split='train', per_cls_instances=5, non_roll=args.non_roll)
        test_set = RGB_Depth_Edge_Dataset(fpath, args.dataset, num_cam, split='test', per_cls_instances=5, non_roll=args.non_roll)
    
    if args.dataset == 'rgb_depth_edge':
        args.num_workers = 24
    # if args.steps:
    #     args.lr /= 5
        # args.epochs *= 2

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, worker_init_fn=seed_worker)

    N = train_set.num_cam

    # logging
    select_settings = f'steps{args.steps}_'
    selector_view_settings = (
        f'selview_{args.selector_view_limit}_'
        if args.steps and args.selector_view_limit != 'all'
        else ''
    )
    lr_settings = f'base{args.base_lr_ratio}other{args.other_lr_ratio}' + \
                  f'select_wd{args.select_wd}' + \
                  f'select{args.select_lr}' if args.steps else ''
    
    logdir = 'logs/'
    if args.name != '':
        logdir = logdir + args.name + '_'

    time = datetime.datetime.today()
    exp_name = time
    logdir = logdir + f'{args.dataset}/' 
    if args.freeze_epoch != 100:
        logdir = logdir + 'freeze_' + str(args.freeze_epoch) + '_'
    logdir = logdir + args.arch +\
             f'{select_settings if args.steps else ""}' \
             f'{selector_view_settings}' \
             f'train_ins{args.num_train_instances}_lr{args.lr}{lr_settings}_e{args.epochs}_' \
             f'{time:%Y-%m-%d_%H-%M-%S}' if not args.eval \
        else f'logs/{args.dataset}/EVAL_{args.resume}'
    os.makedirs(logdir, exist_ok=True)

    sys.stdout = Logger(os.path.join(logdir, str(exp_name)+ '_log.txt'), )
    print(logdir)
    print('Settings:')
    print(vars(args))

    if args.steps:
        meta_log = 'meta_logs/'
        if args.name != '':
            meta_log = meta_log + args.name + '_'
        meta_log = meta_log + f'{args.dataset}/' 
        if args.freeze_epoch != 100:
            meta_log = meta_log  + 'freeze_' + str(args.freeze_epoch) + '_'
        meta_log = meta_log + args.arch +f'{select_settings if args.steps else ""}' \
                 f'{selector_view_settings}' \
                 f'train_ins{args.num_train_instances}_lr{args.lr}{lr_settings}_e{args.epochs}' 
        os.makedirs(meta_log, exist_ok=True)


    # model
    if args.task == 'mvcnn':
        model = MVCNN(train_set, args.arch, args.aggregation, args.dataset).cuda()
        # model= nn.DataParallel(model)

    # load checkpoint
    if args.steps and not args.skip_stage1:
        perf_suffix = f'_freeze{args.freeze_epoch}' if args.freeze_epoch != 100 else ''
        performance_path = f'logs/{args.dataset}/{args.arch}{perf_suffix}_performance.txt'
        with open(performance_path, 'r') as fp:
            result_str = fp.read()
        print(result_str)
        load_dir = result_str.split('\n')[1].replace('# ', '')
        pretrained_dict = torch.load(f'{load_dir}/model.pth')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'select' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    elif args.steps and args.skip_stage1:
        print('--skip_stage1 set: bypassing checkpoint load. '
              'Backbone remains ImageNet-pretrained (MVCNN default); '
              'classifier and selector start from their default init.')

    
    if args.resume:
        print(f'loading checkpoint: logs/{args.dataset}/{args.resume}')
        pretrained_dict = torch.load(f'logs/{args.dataset}/{args.resume}/model.pth')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    param_dicts = [{"params": [p for n, p in model.named_parameters()
                               if 'base' not in n and 'select' not in n and p.requires_grad],
                    "lr": args.lr * args.other_lr_ratio, },
                   {"params": [p for n, p in model.named_parameters() if 'base' in n and p.requires_grad],
                    "lr": args.lr * args.base_lr_ratio, },
                   {"params": [p for n, p in model.named_parameters() if 'select' in n and p.requires_grad],
                    "lr": args.select_lr, 
                    "weight_decay": args.select_wd,}, ]
    optimizer = optim.Adam(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    def warmup_lr_scheduler(epoch, warmup_epochs=0.1 * args.epochs):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return (np.cos((epoch - warmup_epochs) / (args.epochs - warmup_epochs) * np.pi) + 1) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr_scheduler)

    if args.task == 'mvcnn':
        trainer = ClassifierTrainer(model, logdir, args)
    else:
        trainer = PerspectiveTrainer(model, logdir, args)

    # draw curve
    x_epoch, train_loss_s, train_prec_s, test_loss_s, test_prec_s, per_class_acc_list = [], [], [], [], [], []
    longest_ratio_hist, short_ratio_hist, longest_like_ratio_hist, short_like_ratio_hist = [], [], [], []
    acc_hist, longest_count_hist, short_count_hist, longest_like_count_hist, short_like_count_hist = [], [], [], [], []
    eye_deg_bar_hist, inplane_deg_bar_hist = [], []

    all_views_test_prec_s, all_views_per_class_acc_list = [], []
    random_test_prec_s, random_per_class_acc_list = [], []

    expanded_test_prec_s, expanded_per_class_acc_list = [], []
    expanded_like_test_prec_s, expanded_like_per_class_acc_list = [], []
    foreshortened_test_prec_s, foreshortened_per_class_acc_list = [], []
    foreshortened_like_test_prec_s, foreshortened_like_per_class_acc_list = [], []
    remainder_test_prec_s, remainder_per_class_acc_list = [], []

    expanded_test_prec_s_3, expanded_per_class_acc_list_3 = [], []
    expanded_like_test_prec_s_3, expanded_like_per_class_acc_list_3 = [], []
    foreshortened_test_prec_s_3, foreshortened_per_class_acc_list_3 = [], []
    foreshortened_like_test_prec_s_3, foreshortened_like_per_class_acc_list_3 = [], []
    remainder_test_prec_s_3, remainder_per_class_acc_list_3 = [], []

    expanded_test_prec_s_5, expanded_per_class_acc_list_5 = [], []
    expanded_like_test_prec_s_5, expanded_like_per_class_acc_list_5 = [], []
    foreshortened_test_prec_s_5, foreshortened_per_class_acc_list_5 = [], []
    foreshortened_like_test_prec_s_5, foreshortened_like_per_class_acc_list_5 = [], []
    remainder_test_prec_s_5, remainder_per_class_acc_list_5 = [], []

    sel_counts_hist = []
    sel_counts_total = torch.zeros(num_cam).cuda()
    total_selected_hist = []
    # total_sel_ep = int(args.steps)*int(args.batch_size)*num_cam
    # learn
    if not args.eval:
        # trainer.test(test_loader)
        total_select_json = {}
        for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
            print('Training...')
            train_loss, train_prec = trainer.train(epoch, train_loader, optimizer, scheduler, all_cameras=args.all_cameras, freeze_epoch=args.freeze_epoch, freeze_backbone=args.freeze_backbone)
            # if epoch % max(args.epochs // 10, 1) == 0:
            print('Testing...')
            # test_loss, test_prec, (longest, short, longest_like, short_like, selection, total_sel_ep) = trainer.test(test_loader, torch.eye(N) if args.steps else None)
            if args.save_feature:
                os.makedirs(os.path.join(meta_log, str(exp_name)), exist_ok=True)
                feature_file_name = os.path.join(meta_log, str(exp_name), 'feature_' +str(epoch)+'.npz')
            else:
                feature_file_name = False
            test_loss, test_prec, per_class_acc, (meta, selection, total_sel_ep, select_json) = trainer.test(test_loader, torch.eye(N) if args.steps else None, feature_file_name=feature_file_name, epoch=epoch)
            
            if args.steps:
                _, all_views_test_prec, all_views_per_class_acc, (_, _, _, _) = trainer.test(test_loader, None, epoch=epoch)
                _, random_test_prec, random_per_class_acc, (_, _, _, _) = trainer.random_select_test(test_loader, epoch=epoch, num_selections=args.steps)
                _, expanded_test_prec, expanded_per_class_acc, (_, _, _, _) = trainer.restricted_view_test(test_loader, view_type="longest")
                _, expanded_like_test_prec, expanded_like_per_class_acc, (_, _, _, _) = trainer.restricted_view_test(test_loader, view_type="long_like")
                _, foreshortened_test_prec, foreshortened_per_class_acc, (_, _, _, _) = trainer.restricted_view_test(test_loader, view_type="short")
                _, foreshortened_like_test_prec, foreshortened_like_per_class_acc, (_, _, _, _) = trainer.restricted_view_test(test_loader, view_type="short_like")
                _, remainder_test_prec, remainder_per_class_acc, (_, _, _, _) = trainer.restricted_view_test(test_loader, view_type="remainder")

                _, expanded_test_prec_3, expanded_per_class_acc_3, (_, _, _, _) = trainer.restricted_view_test(test_loader, num_views=3, view_type="longest")
                _, expanded_like_test_prec_3, expanded_like_per_class_acc_3, (_, _, _, _) = trainer.restricted_view_test(test_loader, num_views=3, view_type="long_like")
                _, foreshortened_test_prec_3, foreshortened_per_class_acc_3, (_, _, _, _) = trainer.restricted_view_test(test_loader, num_views=3, view_type="short")
                _, foreshortened_like_test_prec_3, foreshortened_like_per_class_acc_3, (_, _, _, _) = trainer.restricted_view_test(test_loader, num_views=3, view_type="short_like")
                _, remainder_test_prec_3, remainder_per_class_acc_3, (_, _, _, _) = trainer.restricted_view_test(test_loader, num_views=3, view_type="remainder")

                _, expanded_test_prec_5, expanded_per_class_acc_5, (_, _, _, _) = trainer.restricted_view_test(test_loader, num_views=5, view_type="longest")
                _, expanded_like_test_prec_5, expanded_like_per_class_acc_5, (_, _, _, _) = trainer.restricted_view_test(test_loader, num_views=5, view_type="long_like")
                _, foreshortened_test_prec_5, foreshortened_per_class_acc_5, (_, _, _, _) = trainer.restricted_view_test(test_loader, num_views=5, view_type="short")
                _, foreshortened_like_test_prec_5, foreshortened_like_per_class_acc_5, (_, _, _, _) = trainer.restricted_view_test(test_loader, num_views=5, view_type="short_like")
                _, remainder_test_prec_5, remainder_per_class_acc_5, (_, _, _, _) = trainer.restricted_view_test(test_loader, num_views=5, view_type="remainder")


            sel_counts_total += selection
            sel_counts_hist.append(selection.cpu())
            acc_hist.append(test_prec)

            total_select_json[epoch] = select_json

            longest_count_hist.append(meta['longest'])
            short_count_hist.append(meta['short'])
            longest_like_count_hist.append(meta['longest_like'])
            short_like_count_hist.append(meta['short_like'])

            total_selected_hist.append(total_sel_ep)

            longest_ratio = (meta['longest'] / max(1, total_sel_ep))
            short_ratio = (meta['short'] / max(1, total_sel_ep))
            longest_like_ratio = (meta['longest_like'] / max(1, total_sel_ep))
            short_like_ratio = (meta['short_like'] / max(1, total_sel_ep))

            longest_ratio_hist.append(float(longest_ratio))
            short_ratio_hist.append(float(short_ratio))
            longest_like_ratio_hist.append(float(longest_like_ratio))
            short_like_ratio_hist.append(float(short_like_ratio))

            eye_deg_bar_hist.append([x / max(1, total_sel_ep) for x in meta['to_eye_deg_bar']])
            inplane_deg_bar_hist.append([x / max(1, total_sel_ep) for x in meta['inplane_deg_bar']])           

            # draw & save
            x_epoch.append(epoch)
            train_loss_s.append(train_loss)
            train_prec_s.append(train_prec)
            test_loss_s.append(test_loss)
            test_prec_s.append(test_prec[0])
            per_class_acc_list.append(per_class_acc)

            if args.steps:
                all_views_test_prec_s.append(all_views_test_prec[0])
                all_views_per_class_acc_list.append(all_views_per_class_acc)


                random_test_prec_s.append(random_test_prec[0])
                random_per_class_acc_list.append(random_per_class_acc)

                expanded_test_prec_s.append(expanded_test_prec[0])
                expanded_per_class_acc_list.append(expanded_per_class_acc)

                expanded_like_test_prec_s.append(expanded_like_test_prec[0])
                expanded_like_per_class_acc_list.append(expanded_like_per_class_acc)

                foreshortened_test_prec_s.append(foreshortened_test_prec[0])
                foreshortened_per_class_acc_list.append(foreshortened_per_class_acc)

                foreshortened_like_test_prec_s.append(foreshortened_like_test_prec[0])
                foreshortened_like_per_class_acc_list.append(foreshortened_like_per_class_acc)

                remainder_test_prec_s.append(remainder_test_prec[0])
                remainder_per_class_acc_list.append(remainder_per_class_acc)

            # 
                expanded_test_prec_s_3.append(expanded_test_prec_3[0])
                expanded_like_test_prec_s_3.append(expanded_like_test_prec_3[0])
                foreshortened_test_prec_s_3.append(foreshortened_test_prec_3[0])
                foreshortened_like_test_prec_s_3.append(foreshortened_like_test_prec_3[0])
                remainder_test_prec_s_3.append(remainder_test_prec_3[0])

                expanded_test_prec_s_5.append(expanded_test_prec_5[0])
                expanded_like_test_prec_s_5.append(expanded_like_test_prec_5[0])
                foreshortened_test_prec_s_5.append(foreshortened_test_prec_5[0])
                foreshortened_like_test_prec_s_5.append(foreshortened_like_test_prec_5[0])
                remainder_test_prec_s_5.append(remainder_test_prec_5[0])



            draw_curve(os.path.join(logdir, 'learning_curve.jpg'), x_epoch, train_loss_s, test_loss_s,
                       train_prec_s, test_prec_s)
            # Save weights every epoch (overwrites). After training completes
            # `logdir/model.pth` holds the last-epoch weights for downstream
            # eval (temporal_selection_test, zero_shot_view_type_test, etc.).
            torch.save(model.state_dict(), os.path.join(logdir, 'model.pth'))
            # Optional per-epoch snapshot for time-resolved downstream analysis
            # (e.g., temporal_selection_test.py --per_epoch_checkpoint).
            if args.save_every_epoch > 0 and (epoch % args.save_every_epoch == 0):
                torch.save(model.state_dict(),
                           os.path.join(logdir, f'model_e{epoch}.pth'))
        if args.steps:
            with open(os.path.join(meta_log, str(exp_name)+'_selection.json'), 'w') as f:
                json.dump(total_select_json, f, indent=4)

        remainder_ratio_hist = plot(logdir, test_prec_s, sel_counts_total.cpu(), sel_counts_hist, total_selected_hist, \
        longest_ratio_hist, short_ratio_hist, longest_like_ratio_hist, short_like_ratio_hist, \
        longest_count_hist, short_count_hist, longest_like_count_hist, short_like_count_hist, \
        eye_deg_bar_hist, inplane_deg_bar_hist, \
        args.epochs, args.steps, non_roll=args.non_roll, non_like=args.non_like)

        if args.steps:
            best_epoch = np.argmax(test_prec_s)
            if (args.epochs-1) - best_epoch > 1:
                overfitting_onset_detected = True
            else:
                overfitting_onset_detected = False
            def slope(y):
                y = np.asarray(y)
                x = np.arange(len(y))
                x_mean = x.mean()
                y_mean = y.mean()
                return np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
            Degradation_detected = slope(test_prec_s[-1:]) < 0
            
            if not overfitting_onset_detected and not Degradation_detected:
                overfit = True
            else:
                overfit = False
            meta_log_file_path = Path(os.path.join(meta_log, str(exp_name)+'_meta'+'.json'))
            test_prec_s = [tensor.item() for tensor in test_prec_s]
            all_views_test_prec_s = [tensor.item() for tensor in all_views_test_prec_s]
            random_test_prec_s = [tensor.item() for tensor in random_test_prec_s]
            expanded_test_prec_s = [tensor.item() for tensor in expanded_test_prec_s]
            expanded_like_test_prec_s = [tensor.item() for tensor in expanded_like_test_prec_s]
            foreshortened_test_prec_s = [tensor.item() for tensor in foreshortened_test_prec_s]
            foreshortened_like_test_prec_s = [tensor.item() for tensor in foreshortened_like_test_prec_s]
            remainder_test_prec_s = [tensor.item() for tensor in remainder_test_prec_s]

            expanded_test_prec_s_3 = [tensor.item() for tensor in expanded_test_prec_s_3]
            expanded_like_test_prec_s_3 = [tensor.item() for tensor in expanded_like_test_prec_s_3]
            foreshortened_test_prec_s_3 = [tensor.item() for tensor in foreshortened_test_prec_s_3]
            foreshortened_like_test_prec_s_3 = [tensor.item() for tensor in foreshortened_like_test_prec_s_3]
            remainder_test_prec_s_3 = [tensor.item() for tensor in remainder_test_prec_s_3]

            expanded_test_prec_s_5 = [tensor.item() for tensor in expanded_test_prec_s_5]
            expanded_like_test_prec_s_5 = [tensor.item() for tensor in expanded_like_test_prec_s_5]
            foreshortened_test_prec_s_5 = [tensor.item() for tensor in foreshortened_test_prec_s_5]
            foreshortened_like_test_prec_s_5 = [tensor.item() for tensor in foreshortened_like_test_prec_s_5]
            remainder_test_prec_s_5 = [tensor.item() for tensor in remainder_test_prec_s_5]

            # random_test_prec_s = [tensor.item() for tensor in random_test_prec_s]

            meta_file = {'overfit': overfit,
                        'selector_view_limit': args.selector_view_limit,
                        'accuracy': test_prec_s,
                        'per_class_acc': per_class_acc_list,
                        'expanded': longest_ratio_hist,
                        'Expanded-like': longest_like_ratio_hist,
                        'Foreshortened': short_ratio_hist,
                        'Foreshortened-like': short_like_ratio_hist,
                        'Remainder': remainder_ratio_hist,
                        'all_views_accuracy': all_views_test_prec_s,
                        'all_views_per_class_acc': all_views_per_class_acc_list,

                        'random_accuracy': random_test_prec_s,
                        'random_per_class_acc': random_per_class_acc_list,

                        'expanded_accuracy': expanded_test_prec_s,
                        'expanded_per_class_acc': expanded_per_class_acc_list,

                        'expanded_like_accuracy': expanded_like_test_prec_s,
                        'expanded_like_per_class_acc': expanded_like_per_class_acc_list,

                        'foreshortened_accuracy': foreshortened_test_prec_s,
                        'foreshortened_per_class_acc': foreshortened_per_class_acc_list,

                        'foreshortened_like_accuracy': foreshortened_like_test_prec_s,
                        'foreshortened_like_per_class_acc': foreshortened_like_per_class_acc_list,

                        'remainder_accuracy': remainder_test_prec_s,
                        'remainder_per_class_acc': remainder_per_class_acc_list,

                        # 
                        'expanded_accuracy_3': expanded_test_prec_s_3,
                        'expanded_like_accuracy_3': expanded_like_test_prec_s_3,
                        'foreshortened_accuracy_3': foreshortened_test_prec_s_3,
                        'foreshortened_like_accuracy_3': foreshortened_like_test_prec_s_3,
                        'remainder_accuracy_3': remainder_test_prec_s_3,

                        #
                        'expanded_accuracy_5': expanded_test_prec_s_5,
                        'expanded_like_accuracy_5': expanded_like_test_prec_s_5,
                        'foreshortened_accuracy_5': foreshortened_test_prec_s_5,
                        'foreshortened_like_accuracy_5': foreshortened_like_test_prec_s_5,
                        'remainder_accuracy_5': remainder_test_prec_s_5,
                        }

            with open(meta_log_file_path, "w") as f:
                json.dump(meta_file, f)


    def log_best2cam_strategy(result_type=('prec',), max_steps=2):
        candidates = np.eye(N)
        combinations = np.array(list(itertools.combinations(candidates, 2))).sum(1)
        combination_indices = np.array(list(itertools.combinations(list(range(N)), 2)))
        info_str = {}
        # diagonal: step == 0
        val_loss_diag, val_prec_diag, _, _ = trainer.test_cam_combination(val_loader, 0)
        test_loss_diag, test_prec_diag, _, info_str[0] = trainer.test_cam_combination(test_loader, 0)
        # non-diagonal: step == 1
        val_loss_s, val_prec_s, val_oracle_s, _ = trainer.test_cam_combination(val_loader, 1)
        test_loss_s, test_prec_s, test_oracle_s, info_str[1] = trainer.test_cam_combination(test_loader, 1)
        
        # for i in range(2, max_steps + 1):
        #     _, _, _, info_str[i] = trainer.test_cam_combination(test_loader, i)
        info_str = '\n'.join(info_str.values())

        def combine2mat(diag_terms, non_diag_terms):
            combined_mat = np.zeros([len(diag_terms), len(diag_terms)] + list(diag_terms.shape[1:]))
            combined_mat[np.eye(len(diag_terms), dtype=bool)] = diag_terms
            non_diag_indices = list(itertools.combinations(list(range(len(diag_terms))), 2))
            for i in range(len(non_diag_indices)):
                idx = non_diag_indices[i]
                combined_mat[idx[0], idx[1]] = combined_mat[idx[1], idx[0]] = non_diag_terms[i]
            return combined_mat

        def find_cam(init_cam, combination_id):
            cam_tuple = list(combination_indices[combination_id])
            cam_tuple.remove(init_cam)
            return cam_tuple[0]

        val_loss_strategy = find_dataset_lvl_strategy(-val_loss_s, combinations)
        val_metric_strategy = find_dataset_lvl_strategy(val_prec_s[:, 0], combinations)
        test_metric_strategy = find_dataset_lvl_strategy(test_prec_s[:, 0], combinations)

        _, prec, per_class_acc, _ = trainer.test(test_loader)
        np.savetxt(f'{logdir}/losses_val_test.txt', np.concatenate([combine2mat(val_loss_diag, val_loss_s),
                                                                    combine2mat(test_loss_diag, test_loss_s)]), '%.2f')
        for i in range(len(result_type)):
            fname = f'{result_type[i]}_{prec[i]:.1f}_' \
                    f'Lstrategy{test_prec_s[val_loss_strategy].mean(0)[i]:.1f}_' \
                    f'Rstrategy{test_prec_s[val_metric_strategy].mean(0)[i]:.1f}_' \
                    f'theory{test_prec_s[test_metric_strategy].mean(0)[i]:.1f}_' \
                    f'avg{test_prec_s.mean(0)[i]:.1f}.txt'
            np.savetxt(f'{logdir}/{fname}',
                       np.concatenate([combine2mat(val_prec_diag, val_prec_s)[:, :, i],
                                       combine2mat(test_prec_diag, test_prec_s)[:, :, i]]), '%.1f',
                       header=f'loading checkpoint...\n'
                              f'{logdir}\n'
                              f'val / test',
                       footer=(f'\n{info_str}\n\n' if i == 0 else '') + f'\tdataset level: loss strategy\n' +
                              ' '.join(f'cam {find_cam(cam, val_loss_strategy[cam])} |' for cam in range(N)) + '\n' +
                              ' '.join(f'{test_prec_s[val_loss_strategy][cam, i]:.1f}% |'
                                       for cam in range(N)) + '\n' +
                              f'\tdataset level: result strategy\n' +
                              ' '.join(f'cam {find_cam(cam, val_metric_strategy[cam])} |' for cam in range(N)) + '\n' +
                              ' '.join(f'{test_prec_s[val_metric_strategy][cam, i]:.1f}% |'
                                       for cam in range(N)) + '\n' +
                              f'\tdataset level: theory\n' +
                              ' '.join(f'cam {find_cam(cam, test_metric_strategy[cam])} |' for cam in range(N)) + '\n' +
                              ' '.join(f'{test_prec_s[test_metric_strategy][cam, i]:.1f}% |'
                                       for cam in range(N)) + '\n' +
                              f'\tinstance level: oracle\n' +
                              ' '.join(f'----- |' for cam in range(N)) + '\n' +
                              ' '.join(f'{test_oracle_s[cam, i]:.1f}% |'
                                       for cam in range(N)) + '\n' +
                              f'2 best cam: loss_strategy {test_prec_s[val_loss_strategy].mean(0)[i]:.1f}, '
                              f'result_strategy {test_prec_s[val_metric_strategy].mean(0)[i]:.1f}, '
                              f'theory {test_prec_s[test_metric_strategy].mean(0)[i]:.1f}, '
                              f'oracle {test_oracle_s.mean(0)[i]:.1f}, average {test_prec_s.mean(0)[i]:.1f}\n'
                              f'all cam: {prec[i]:.1f}')
            with open(f'{logdir}/{fname}', 'r') as fp:
                if i == 0:
                    print(fp.read())
            if not args.eval and i == 0:
                perf_suffix = f'_freeze{args.freeze_epoch}' if args.freeze_epoch != 100 else ''
                shutil.copyfile(f'{logdir}/{fname}',
                                f'logs/{args.dataset}/{args.arch}{perf_suffix}_performance.txt')

    print('Test loaded model...')
    print(logdir)
    if args.steps == 0:
        log_best2cam_strategy(result_type)
    else:
        if args.eval:
            _, _, per_class_acc, (out_meta, selection, total_sel_ep, _) = trainer.test(test_loader, torch.eye(N))
        else:
            _, _, per_class_acc, (out_meta, selection, total_sel_ep, _) = trainer.test(test_loader)


if __name__ == '__main__':
    # common settings
    parser = argparse.ArgumentParser(description='view selection for multiview classification & detection')
    
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--eval', action='store_true', help='evaluation only')
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--aggregation', type=str, default='max', choices=['mean', 'max'])
    parser.add_argument('-d', '--dataset', type=str, default='rgb',
                        choices=['rgb','depth', 'edge', 'rgb_depth', 'rgb_edge', 'depth_edge', 'rgb_depth_edge'])
    parser.add_argument('-j', '--num_workers', type=int, default=2)
    parser.add_argument('-b', '--batch_size', type=int, default=None, help='input batch size for training')
    parser.add_argument('--dropcam', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=None, help='learning rate for task network')
    parser.add_argument('--select_lr', type=float, default=None, help='learning rate for MVselect')
    parser.add_argument('--select_wd', type=float, default=1e-4, help='learning rate for MVselect')
    parser.add_argument('--base_lr_ratio', type=float, default=1.0)
    parser.add_argument('--other_lr_ratio', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--deterministic', type=str2bool, default=False)
    
    #vis
    parser.add_argument('--save_feature', action='store_true', help='evaluation only')

    #dataset
    parser.add_argument('--non_roll', action='store_true', help='evaluation only')
    parser.add_argument('--non_like', action='store_true', help='evaluation only')

    # MVSelect settings
    parser.add_argument('--steps', type=int, default=0,
                        help='number of camera views to choose. if 0, then no selection')
    parser.add_argument('--selector_view_limit', type=str, default='all',
                        choices=['all', 'expanded_family',
                                 'foreshortened_family',
                                 'foreshortened_family_remainder',
                                 'remainder'],
                        help='Stage-2 only (--steps >0): restrict MVSelect action '
                             'candidates to a view family. The initial view is not '
                             'restricted, and random/restricted/all-view test baselines '
                             'are unaffected. Choices: all, expanded_family '
                             '(expanded + expanded-like), foreshortened_family '
                             '(foreshortened + foreshortened-like), '
                             'foreshortened_family_remainder, remainder.')
    parser.add_argument('--train_num_views', type=int, default=None,
                        help='All-view training mode (--steps 0): if set to K, each training batch '
                             'is restricted to K random views per instance, re-sampled '
                             'every batch. None = use all available views (default).')
    parser.add_argument('--skip_stage1', action='store_true',
                        help='Selector training (--steps >0): skip loading an existing '
                             'checkpoint. The backbone remains ImageNet-pretrained '
                             '(MVCNN default); classifier and selector heads start from '
                             'their default init and train jointly with the selector.')
    parser.add_argument('--save_every_epoch', type=int, default=1,
                        help='If >0, additionally save model_e<E>.pth every N epochs '
                             '(in addition to model.pth which overwrites every epoch). '
                             '0 = off. 1 = every epoch (heaviest on disk). 5/10 = sampled.'
                             ' Needed if you want to run temporal_selection_test.py with '
                             '--per_epoch_checkpoint (model-at-t analysis).')
    parser.add_argument('--num_train_instances', type=int, default=30)
    parser.add_argument('--freeze_epoch', type=int, default=100)
    parser.add_argument('--freeze_backbone', action='store_true')


    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor (default: 0.99)')
    parser.add_argument('--down', type=int, default=1, help='down sample the image to 1/N size')
    parser.add_argument('--all_cameras', action='store_true', help='evaluation only')

    # parser.add_argument('--beta_entropy', type=float, default=0.01)
    # multiview detection specific settings
    parser.add_argument('--eval_init_cam', type=str2bool, default=False,
                        help='only consider pedestrians covered by the initial camera')
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--augmentation', type=str2bool, default=True)
    parser.add_argument('--id_ratio', type=float, default=0)
    parser.add_argument('--cls_thres', type=float, default=0.6)
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per view loss')
    parser.add_argument('--use_mse', type=str2bool, default=False)
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--outfeat_dim', type=int, default=0)
    parser.add_argument('--world_reduce', type=int, default=4)
    parser.add_argument('--world_kernel_size', type=int, default=10)
    parser.add_argument('--img_reduce', type=int, default=12)
    parser.add_argument('--img_kernel_size', type=int, default=10)

    args = parser.parse_args()

    main(args)
