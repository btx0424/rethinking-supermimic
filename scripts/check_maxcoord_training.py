"""Six graph iterations per architecture: collect count, reward terms and logging."""
import argparse
import csv
import json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from supermimic.maxcoord.trainer import Trainer, MaxCoordConfig
from supermimic.maxcoord.logging import grouped_metrics
from supermimic.motion import CONTROL_DT
from unittest.mock import patch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wm', choices=['explicit','latent'], required=True)
    parser.add_argument('--actor', choices=['explicit','latent'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    torch.set_num_threads(8)
    cfg=MaxCoordConfig(wm_kind=args.wm, actor_kind=args.actor, num_envs=16, buffer_steps=96,
        wm_hidden=(32,32), actor_hidden=(32,32), latent_dim=16, wm_batches=1, wm_batch=16,
        pi_batch=8, wm_window=8, horizon=32, critic_epochs=1, critic_minibatches=2,
        pi_warmup=0, cuda_graphs=True, iterations=6, collect_steps=4,
        swanlab_mode='disabled', log_every=3, ckpt_every=6)
    t=Trainer(cfg, OmegaConf.create({'name':'lafan1'}),args.output)
    calls=[]; count=0; raw_steps=[]; raw_buffers=[]
    original_step=t.sim.step
    original_collect=t.collect
    def step(*a,**kw):
        nonlocal count
        count+=1
        out=original_step(*a,**kw)
        raw_buffers.append({k:out[k].detach().clone() for k in ("s","s1","a","xi")})
        raw_steps.append({"total":out["r"].mean().detach().clone(),
                          **{k:v.mean().detach().clone() for k,v in out["terms"].items()}})
        return out
    def collect(steps):
        before=count
        counts={k:v.count.clone() for k,v in t.vecnorms.items()}
        result=original_collect(steps)
        for k,norm in t.vecnorms.items():
            expected=counts[k]
            for _ in range(steps):expected.mul_(norm.decay).add_(cfg.num_envs)
            torch.testing.assert_close(norm.count,expected,rtol=0,atol=0)
        calls.append(count-before)
        for key in t.mdp.term_names:
            expected=torch.stack([r[key] for r in raw_steps[before:count]]).mean()/CONTROL_DT
            torch.testing.assert_close(result["sim_term_"+key],expected)
        torch.testing.assert_close(result["sim_reward"],torch.stack([r["total"] for r in raw_steps[before:count]]).mean()/CONTROL_DT)
        return result
    t.sim.step=step;t.collect=collect
    original_add=t.buffer.add
    def add(batch):
        i=t.buffer.ptr
        original_add(batch)
        for key,value in raw_buffers[-1].items():
            torch.testing.assert_close(getattr(t.buffer,key)[i],value,rtol=0,atol=0)
    t.buffer.add=add
    def frozen_update(fn):
        def checked(*a,**kw):
            before={k:v.clone() for k,v in t.vecnorms.state_dict().items()}
            result=fn(*a,**kw)
            for k,v in t.vecnorms.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
            return result
        return checked
    t.fit_world_model=frozen_update(t.fit_world_model)
    t.train_actor_critic=frozen_update(t.train_actor_critic)
    uploaded = []
    with patch("supermimic.maxcoord.trainer.ExperimentLogger.log", side_effect=lambda metrics,step: uploaded.append((step,dict(metrics)))):
        t.fit()
    assert [step for step,_ in uploaded] == list(range(1, 7))
    assert all("Performance/train_time" in metrics and "Performance/collect_time" in metrics for _,metrics in uploaded)
    timings=list(csv.DictReader((args.output/'performance.csv').open()))
    assert len(timings)==6 and all(float(r['collect_time'])>0 and float(r['train_time'])>0 for r in timings)
    assert calls == [32]+[4]*6,calls
    assert t.accelerated.wm_update.graph is not None
    assert t.accelerated.policy_update.graph is not None
    rows=list(csv.DictReader((args.output/'train_log.csv').open()))
    for row in rows:
        terms=t.mdp.term_names
        for prefix,total in [('sim_term_','sim_reward'),('imag_term_','imag_raw_reward')]:
            values=[float(row[prefix+name]) for name in terms]
            assert abs(sum(values)-float(row[total])) < 1e-5,(total,values,row[total])
        groups=grouped_metrics(row)
        assert all('/' in name for name in groups)
        assert {name.split('/')[0] for name in groups} == {'SimReward','WMReward','WMLearning','ActorCriticLearning','Performance'}
    saved=torch.load(args.output/'last.pt',weights_only=False)
    assert saved['format']=='maxcoord_v4'
    result={'passed':True,'collect_calls':calls,'reward_terms':list(t.mdp.term_names),'reward_units':'before_dt','vecnorm_collect_only':True,'raw_buffer':True,'sections':sorted({name.split('/')[0] for name in groups})}
    (args.output/'checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
