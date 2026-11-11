"""Exercise actual v1 eval/play loading with two physical steps, not policy evaluation."""
import argparse
from pathlib import Path
import torch
import supermimic.eval as evaluation
from supermimic.play import _load_policy_overlay
from supermimic.maxcoord.inference import load_policy


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(8)
    saved=torch.load(args.checkpoint,map_location='cuda',weights_only=False)
    assert saved['format']=='maxcoord_v1'
    cfg=evaluation._flatten_saved(saved)
    seen=[]
    def short_rollout(sim,motion,n,device,step_fn,**kw):
        for _ in range(2):
            out=step_fn(sim,motion,torch.ones(n,dtype=torch.bool,device=device))
            assert torch.isfinite(out['a']).all()
        seen.append(n)
        zeros=torch.zeros(n,device=device)
        return zeros,zeros.bool(),zeros.bool(),zeros.long(),zeros,zeros
    evaluation._run_rollout=short_rollout
    evaluation._print_results=lambda *a,**kw:None
    evaluation.evaluate(args.checkpoint,'cuda',progress=False)
    assert seen==[40],seen
    overlay=_load_policy_overlay(args.checkpoint,evaluation._task_from_cfg(cfg),cfg,'cuda')
    overlay.reset(0)
    overlay.step();overlay.step()
    print('legacy v1 eval dispatch (40 clips, 2 steps) and play (2 steps): PASS')


if __name__=='__main__':main()
