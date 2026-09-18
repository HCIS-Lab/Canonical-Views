#!/usr/bin/env python3
"""Run a documented CLIP probe when the original numerical output is missing.

This generates new inference results. It does not claim to reproduce missing
historical values or train CLIP. Original output folders are left unchanged.
"""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-id',type=int,default=0)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=Path('/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_23'))
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args();repo=Path(__file__).resolve().parent;script=repo/'clip_zero_shot_view_type.py'
    if not script.is_file():p.error('Place this wrapper in the actual MVSelect-main directory.')
    out=args.output.resolve()
    if out.exists():p.error('Use a new output directory; existing results will not be overwritten.')
    if not args.data_root.is_dir() and not args.plan_only:p.error(f'Missing image dataset: {args.data_root}')
    command=[sys.executable,str(script),'--data_root',str(args.data_root),'--split','test',
             '--per_cls_instances','25','--n_runs','5','--clip_model','openai/clip-vit-base-patch32',
             '--random_seed','42','--batch_size','32','--gpu_id',str(args.gpu_id),'--output_dir',str(out)]
    out.mkdir(parents=True)
    record={'protocol':'new_documented_CLIP_evaluation','historical_result_recovered':False,
            'new_training':False,'new_inference_requested':not args.plan_only,
            'argv':command,'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),
            'prompt_ensemble':True,'test_objects_per_category':25,'sampling_repeats':5,
            'scope_note':'Matches the 25-per-category external-probe scope; differs from the five-per-category core recognizer test set.'}
    (out/'probe_config.json').write_text(json.dumps(record,indent=2)+'\n')
    if args.plan_only:print(f'Plan only; no inference: {out}/probe_config.json');return
    print(f'Running CLIP inference. Progress is saved in {out}/probe.log',flush=True)
    with (out/'probe.log').open('w') as log:
        result=subprocess.run(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
    record['exit_code']=result.returncode
    record['results_csv_exists']=(out/'results.csv').exists()
    (out/'probe_config.json').write_text(json.dumps(record,indent=2)+'\n')
    if result.returncode:raise SystemExit(f'Probe failed; return {out}/probe.log and probe_config.json for diagnosis.')
    if not record['results_csv_exists']:raise SystemExit('Probe returned without results.csv; inspect probe.log.')
    print(f'Return {out}/results.csv, probe_config.json and probe.log')

if __name__=='__main__':main()
