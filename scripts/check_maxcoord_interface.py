"""Verify WM refactor numerics, state protocol and critic context without a simulator."""
import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from check_maxcoord import fixture
from supermimic.maxcoord.interface import WorldModel, WMBatch
from supermimic.maxcoord.models import build_models
from supermimic.maxcoord.trainer import MaxCoordConfig, Trainer
from supermimic.supermimic.network import ScalarCritic


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);args=p.parse_args()
    spec=importlib.util.spec_from_file_location('reference_v2',args.reference/'__init__.py',submodule_search_locations=[str(args.reference)])
    reference=importlib.util.module_from_spec(spec);sys.modules[spec.name]=reference;spec.loader.exec_module(reference)
    from reference_v2.models import build_models as old_build
    from reference_v2.loss import rollout_loss as old_loss
    torch.set_num_threads(2)
    full,_=fixture();xi=torch.randn(3,33);actions=torch.randn(3,8,29)
    for kind in ['explicit','latent']:
        cfg=MaxCoordConfig(wm_kind=kind,wm_hidden=(32,32),actor_hidden=(32,32),latent_dim=16,input_normalization="legacy")
        wm,actor=build_models(cfg,14,29);old,_=old_build(cfg,14,29);old.load_state_dict(wm.state_dict())
        assert isinstance(wm,WorldModel)
        wm.train(False);assert not wm.training;wm.train()
        batch=WMBatch(full,actions,full[:,None].expand(-1,8,-1),xi)
        result=wm.compute_loss(batch);loss,metrics=old_loss(old,full,actions,batch.next_full,xi_n=xi)
        torch.testing.assert_close(result.loss,loss,rtol=0,atol=0)
        for k in metrics:torch.testing.assert_close(result.metrics[k],metrics[k],rtol=0,atol=0)
        result.loss.backward();loss.backward()
        for (_,a),(_,b) in zip(wm.named_parameters(),old.named_parameters(),strict=True):
            if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
        state=wm.initialize(full,xi);rollout=wm.rollout(state,actions)
        expected=old.initialize(full,xi)
        for action in actions.unbind(1):expected=old.step(expected,action)
        for a,b in zip(rollout.final,expected,strict=True):
            if a is not None:torch.testing.assert_close(a,b,rtol=0,atol=0)
        targets=wm.prepare_targets(batch.next_full)
        assert not targets.f.requires_grad and (targets.z is None or not targets.z.requires_grad)
        # Critic gets context, actor does not; verify sensitivity to xi with identical f/goal.
        trainer=Trainer.__new__(Trainer)
        trainer.cfg=cfg;trainer.actor=actor;trainer.task=SimpleNamespace(num_joints=29);trainer.vecnorms=None
        g=torch.randn(3,380);x=Trainer.critic_inputs(trainer,state,g)
        changed=state._replace(xi_n=xi+1);y=Trainer.critic_inputs(trainer,changed,g)
        assert x.shape[-1]==594+33
        torch.testing.assert_close(x[...,:-33],y[...,:-33]);torch.testing.assert_close(x[...,-33:],xi)
        critic=ScalarCritic(x.shape[-1]);assert not torch.allclose(critic(x),critic(y))
        print(kind, 'old/new loss, metrics, gradients and rollout EXACT; critic context PASS')
    # Disabling WM xi must not drop the critic's context.
    cfg.wm_xi=False;wm,_=build_models(cfg,14,29)
    st=wm(wm.initialize(full,xi),actions[:,0]);torch.testing.assert_close(st.xi_n,xi)
    print('Context retained independently of WM conditioning: PASS')


if __name__=='__main__':main()
