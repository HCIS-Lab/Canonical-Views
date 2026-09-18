#!/usr/bin/env python3
"""Evaluate saved ResNet-18 recognizers on identical five-image test sets.

This is a new evaluation protocol, not recovery of an old figure. No training.
All models/checkpoints receive the same objects, images and perturbations.
Use --preflight to check paths and generate the test manifest without inference.
"""
import argparse, csv, hashlib, json, random, statistics
from pathlib import Path

def samples_for_instance(paths, seed, class_idx, object_id, repeat, count):
    rng=random.Random(f'{seed}|{class_idx}|{object_id}|{repeat}|views')
    return rng.sample(sorted(paths),count)

def perturbation_parameters(seed, class_idx, object_id, repeat, slot, rotation, jitter, hue):
    rng=random.Random(f'{seed}|{class_idx}|{object_id}|{repeat}|{slot}|perturbation')
    return dict(angle=rng.uniform(-rotation,rotation),brightness=rng.uniform(1-jitter,1+jitter),
                contrast=rng.uniform(1-jitter,1+jitter),saturation=rng.uniform(1-jitter,1+jitter),
                hue=rng.uniform(-hue,hue))

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry',type=Path,default=Path(__file__).with_name('paper_core_checkpoints.json'))
    p.add_argument('--instances',type=Path,default=Path(__file__).with_name('paper_test_instances.csv'))
    p.add_argument('--data-root',type=Path,default=Path('/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_23'))
    p.add_argument('--epochs',type=int,nargs='+',default=[10,30,100])
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--num-views',type=int,default=5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--rotation-deg',type=float,default=10)
    p.add_argument('--jitter-strength',type=float,default=.4)
    p.add_argument('--hue-strength',type=float,default=.1)
    p.add_argument('--gpu-id',type=int,default=0)
    p.add_argument('--batch-size',type=int,default=8,help='Objects per batch; each has five images.')
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--smoke-test',action='store_true',help='Only first model, two object/draw trials; not paper results.')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.repeats<1 or args.num_views<1 or args.batch_size<1:p.error('Counts must be positive.')
    if not (0<=args.jitter_strength<=1 and 0<=args.hue_strength<=.5 and args.rotation_deg>=0):p.error('Invalid perturbation magnitudes.')
    if args.output.exists():p.error('Use a new output directory; existing results will not be overwritten.')
    models=json.loads(args.registry.read_text())['models'];jobs=[]
    for m in models:
        for epoch in args.epochs:
            path=Path(m['checkpoints'].get(str(epoch),''))
            if not path.is_file():p.error(f'Missing checkpoint: {m["run"]}, epoch {epoch}: {path}')
            jobs.append(dict(condition=m['condition'],run=m['run'],epoch=epoch,checkpoint=str(path)))
    instances=list(csv.DictReader(args.instances.open()))
    if len(instances)!=160 or len({r['object_id'] for r in instances})!=160:p.error('Expected 160 distinct held-out test objects.')
    trials=[]
    for item in instances:
        cls=int(item['class_idx']);oid=item['object_id'];folder=args.data_root/item['category']/'test'
        paths=[x for x in folder.glob(oid+'_*.png') if float(x.stem.split('_')[4][1:])==0]
        if len(paths)!=114:p.error(f'Expected 114 non-roll views for {folder}/{oid}; found {len(paths)}')
        for repeat in range(args.repeats):
            chosen=samples_for_instance(paths,args.seed,cls,oid,repeat,args.num_views)
            params=[perturbation_parameters(args.seed,cls,oid,repeat,i,args.rotation_deg,args.jitter_strength,args.hue_strength) for i in range(args.num_views)]
            trials.append(dict(class_idx=cls,object_id=oid,repeat=repeat,paths=[str(x) for x in chosen],perturbations=params))
    if args.smoke_test:
        jobs=jobs[:1];trials=trials[:2]
    # Check actual CUDA allocation before reserving an output directory.
    if not args.preflight:
        import torch
        if not torch.cuda.is_available():
            p.error('CUDA is unavailable. Run diagnose_paper_gpu.py in this same environment.')
        if not 0 <= args.gpu_id < torch.cuda.device_count():
            p.error('GPU index is outside the visible device list. Use logical --gpu-id 0 within an allocation.')
        device=torch.device(f'cuda:{args.gpu_id}')
        try:
            torch.cuda.set_device(device)
            torch.ones(1,device=device).sum().item()
            torch.cuda.synchronize(device)
        except Exception as ex:
            p.error(f'CUDA tensor allocation failed before inference: {ex}')
    args.output.mkdir(parents=True)
    plan={'protocol':'common_random_five_saved_classifier_checkpoints','new_training':False,
          'completed':False,'smoke_test':args.smoke_test,'new_evaluation':not args.preflight,'settings':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
          'architecture':'ResNet-18; spatial average pooling, maximum pooling across views',
          'test_images_shared_across_all_models_and_epochs':True,'jobs':jobs,'trials':trials,
          'script_sha256':digest(Path(__file__)),'registry_sha256':digest(args.registry),'instances_sha256':digest(args.instances)}
    (args.output/'evaluation_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    if args.preflight:
        print(f'Preflight passed: {len(jobs)} model/checkpoint jobs; {len(trials)} shared object/draw trials. No model evaluation performed. Plan: {args.output}/evaluation_plan.json')
        return
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models as tvmodels, transforms
    from torchvision.transforms import functional as TF
    from PIL import Image
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    class Recognizer(nn.Module):
        def __init__(self):
            super().__init__()
            self.base=nn.Sequential(*list(tvmodels.resnet18(weights=None).children())[:-2])
            self.avgpool=nn.AdaptiveAvgPool2d((1,1));self.classifier=nn.Linear(512,32)
        def forward(self,x):
            b,n=x.shape[:2];f=self.avgpool(self.base(x.flatten(0,1))).flatten(1).view(b,n,-1)
            return self.classifier(f.max(dim=1).values)
    model=Recognizer().to(device).eval()
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                  transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    conditions=['clean','rotation','colour_jitter','rotation_and_colour_jitter']
    def images(trial,condition):
        imgs=[]
        for path,param in zip(trial['paths'],trial['perturbations']):
            with Image.open(path) as src:im=src.convert('RGB')
            if 'rotation' in condition:im=TF.rotate(im,param['angle'],fill=255)
            if 'colour_jitter' in condition:
                im=TF.adjust_brightness(im,param['brightness']);im=TF.adjust_contrast(im,param['contrast'])
                im=TF.adjust_saturation(im,param['saturation']);im=TF.adjust_hue(im,param['hue'])
            imgs.append(transform(im))
        return torch.stack(imgs)
    fields=['condition','run','epoch','checkpoint_sha256','test_condition','repeat','class_idx','object_id','prediction','correct','category_loss','prediction_margin']
    summaries=[]
    with (args.output/'trial_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for job in jobs:
            path=Path(job['checkpoint']);state=torch.load(path,map_location='cpu',weights_only=True)
            if 'state_dict' in state:state=state['state_dict']
            state={k.removeprefix('module.'):v for k,v in state.items()}
            recognition={k:v for k,v in state.items() if not k.startswith('select_module.')}
            model.load_state_dict(recognition,strict=True)
            sha=digest(path);job['checkpoint_sha256']=sha
            for condition in conditions:
                correct=[];losses=[]
                with torch.inference_mode():
                    for offset in range(0,len(trials),args.batch_size):
                        batch=trials[offset:offset+args.batch_size]
                        x=torch.stack([images(t,condition) for t in batch]).to(device)
                        y=torch.tensor([t['class_idx'] for t in batch],device=device)
                        logits=model(x);loss=F.cross_entropy(logits,y,reduction='none').cpu().tolist()
                        pred=logits.argmax(1).cpu().tolist();top=logits.topk(2,dim=1).values;margin=(top[:,0]-top[:,1]).cpu().tolist()
                        for t,pr,ll,mm in zip(batch,pred,loss,margin):
                            hit=int(pr==t['class_idx']);correct.append(hit);losses.append(ll)
                            writer.writerow(dict(condition=job['condition'],run=job['run'],epoch=job['epoch'],checkpoint_sha256=sha,
                                                 test_condition=condition,repeat=t['repeat'],class_idx=t['class_idx'],object_id=t['object_id'],
                                                 prediction=pr,correct=hit,category_loss=ll,prediction_margin=mm))
                summaries.append(dict(condition=job['condition'],run=job['run'],epoch=job['epoch'],test_condition=condition,
                                      n_objects=len({t["object_id"] for t in trials}),n_sampling_repeats=args.repeats,n_trials=len(correct),
                                      accuracy_pct=100*statistics.mean(correct),mean_category_loss=statistics.mean(losses)))
            f.flush();print(f'Completed {job["condition"]} / {job["run"]} / epoch {job["epoch"]}',flush=True)
    for r in summaries:
        clean=next(x for x in summaries if all(x[k]==r[k] for k in ['run','epoch']) and x['test_condition']=='clean')
        r['accuracy_drop_percentage_points']=clean['accuracy_pct']-r['accuracy_pct']
    with (args.output/'run_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    plan['completed']=True;plan['completed_scope']='smoke_test_only' if args.smoke_test else 'full_protocol';plan['torch_version']=torch.__version__
    (args.output/'evaluation_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(f'Completed saved-model evaluation: {args.output}. Compute uncertainty across trained runs, not across sampling repeats or views.')

if __name__=='__main__':main()
