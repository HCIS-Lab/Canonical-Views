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

    fpath = os.path.expanduser('/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_23')

    num_cam = args.num_cam
    args.task = 'mvcnn'
    result_type = ['prec']
    args.lr = 5e-5 if args.lr is None else args.lr
    args.select_lr = 1e-4 if args.select_lr is None else args.select_lr
    args.batch_size = 6 if args.batch_size is None else args.batch_size


    # logging
    lr_settings = f'base{args.base_lr_ratio}other{args.other_lr_ratio}'
    logdir = 'downstream_logs/'
    if args.name != '':
        logdir = logdir + args.name + '_'

    if 'random' in args.select:
        args.selected_rep = args.select
        args.selected_view_type = '01234'
    logdir = (
        logdir
        + f"{args.selected_rep}/{args.selected_view_type}/"
        + f"{args.select}/numviews{args.num_cam}/"
        + f"train_ins{args.num_train_instances}"
        + f"_lr{args.lr}"
        + f"{lr_settings}"
)

    os.makedirs(logdir, exist_ok=True)

    selection_dir = (
        "meta_logs/"
        + f"{args.selected_rep}/"
        + f"steps{args.selected_steps}_"
        + f"train_ins{25}_lr{1e-5}base{1.0}other{1.0}"
        + "select_wd0.0001select0.0001_e100"
    )
    time = datetime.datetime.today()
    exp_name = time

    train_set = Downstream_ModelNet40(
        selection_dir, args.selected_view_type, args.start, args.end, \
        '/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_23', \
        num_cam, split='test', per_cls_instances=args.num_train_instances, select=args.select)
    
    
    # if args.select == 'selected_test':
    test_set = Downstream_ModelNet40(
        selection_dir, args.selected_view_type, args.start, args.end, \
        '/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_23', \
        num_cam, split='test-down', per_cls_instances=5, select='random')

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, worker_init_fn=seed_worker)

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, worker_init_fn=seed_worker)

    N = train_set.num_cam


    print(logdir)
    print('Settings:')
    print(vars(args))

    # model
    model = MVCNN(train_set, args.arch, args.aggregation).cuda()



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

    # draw curve
    x_epoch, train_loss_s, train_prec_s, test_loss_s, test_prec_s = [], [], [], [], []
    acc_hist = []
    best_prec = 0.0
    # learn
    if not args.eval:
        # trainer.test(test_loader)
        total_select_json = {}
        for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
            print('Training...')
            train_loss, train_prec = trainer.train(epoch, train_loader, optimizer, scheduler, all_cameras=args.all_cameras)
            # if epoch % max(args.epochs // 10, 1) == 0:
            print('Testing...')

            test_loss, test_prec, (_, _, _, _), miss_ins = trainer.test(test_loader, torch.eye(N) if args.steps else None, return_miss_ins=True)
            acc_hist.append(test_prec)

            # if test_prec[0] > best_prec:
            #     best_prec = test_prec[0]
            #     with open(os.path.join(logdir, exp_name +'_miss_ins.json'), "w") as f:
            #         json.dump(miss_ins, f)

            x_epoch.append(epoch)
            train_loss_s.append(train_loss)
            train_prec_s.append(train_prec)
            test_loss_s.append(test_loss)
            test_prec_s.append(test_prec[0])
            draw_curve(os.path.join(logdir, str(exp_name) +'_learning_curve.jpg'), x_epoch, train_loss_s, test_loss_s,
                       train_prec_s, test_prec_s)

        print('Best test result:')
        print(max(test_prec_s))

        test_prec_s = [tensor.item() for tensor in test_prec_s]
        meta_file = {'accuracy_epochs': test_prec_s,
                    'best_accuracy': max(test_prec_s),
                    }

        meta_log_file_path = Path(os.path.join(logdir, str(exp_name)+'_meta'+'.json'))
        with open(meta_log_file_path, "w") as f:
            json.dump(meta_file, f)

# CUDA_VISIBLE_DEVICES=2 python train_downstream.py --selected_view_type R -selected_rep rgb --start 80 --end 100 --num_cam 5 --non_roll --num_train_instances 25

if __name__ == '__main__':
    # common settings
    parser = argparse.ArgumentParser(description='view selection for multiview classification & detection')
    
    parser.add_argument('--selected_view_type', type=str, default='R')
    parser.add_argument('--selected_steps', type=int, default=5)
    parser.add_argument('--selected_rep', type=str, default='rgb',
                        choices=['rgb','depth', 'edge', 'rgb_depth', 'rgb_edge', 'depth_edge', 'rgb_depth_edge'])

    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=20)
    parser.add_argument('--num_cam', type=int, default=2)

    parser.add_argument('--eval', action='store_true', help='evaluation only')
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--select', type=str, default='agent')
    parser.add_argument('--aggregation', type=str, default='max', choices=['mean', 'max'])
    parser.add_argument('-d', '--dataset', type=str, default='modelnet_32_60_latest',
                        choices=['modelnet_32_60_latest', 'modelnet_32_110_fine_qaulity', 'modelnet_36_110_fine_qaulity'])
    parser.add_argument('-j', '--num_workers', type=int, default=4)
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
    
    #dataset
    parser.add_argument('--non_roll', action='store_true', help='evaluation only')
    parser.add_argument('--non_like', action='store_true', help='evaluation only')

    # MVSelect settings
    parser.add_argument('--steps', type=int, default=0,
                        help='number of camera views to choose. if 0, then no selection')
    parser.add_argument('--num_train_instances', type=int, default=50)

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
