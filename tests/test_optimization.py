import pytest
import torch

from supermimic.algo.supermimic.capture import optimize_capturable
from supermimic.algo.supermimic.config import TrainConfig
from supermimic.nn.optim import build_optimizer


def test_norm_then_value_clip():
    model = torch.nn.Linear(2, 1, bias=False)
    opt = torch.optim.SGD(model.parameters(), lr=1)
    before = model.weight.detach().clone()
    norm = optimize_capturable(model(torch.tensor([[3.0, 4.0]])).sum(), model, opt, 1.0, 0.1)
    torch.testing.assert_close(norm, torch.tensor(5.0))
    torch.testing.assert_close(before - model.weight, torch.full_like(before, 0.1))


def test_optimizer_defaults_and_adam_split():
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    model[0].weight._non_muon = True
    legacy = build_optimizer(model, 2e-4, "muon")
    aligned = build_optimizer(model, 2e-4, "muon_adam", adam_betas=(0.9, 0.95))
    assert isinstance(legacy.optimizers[-1], torch.optim.AdamW)
    assert legacy.optimizers[-1].param_groups[0]["weight_decay"] == 0.01
    assert isinstance(aligned.optimizers[-1], torch.optim.Adam)
    assert aligned.optimizers[-1].param_groups[0]["weight_decay"] == 0
    assert aligned.optimizers[-1].param_groups[0]["betas"] == (0.9, 0.95)
    assert {id(p) for g in aligned.param_groups for p in g["params"]} == {
        id(p) for p in model.parameters()
    }
    assert TrainConfig().critic_optim == "muon_adam"
    assert TrainConfig().actor_grad_clip_value == 0.1


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_bad_clip(value):
    with pytest.raises(ValueError):
        TrainConfig(actor_grad_clip_value=value).validate()
