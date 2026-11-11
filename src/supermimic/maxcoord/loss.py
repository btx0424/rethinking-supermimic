"""Eight actions, eight integrated successors; state and latent supervision."""
import torch

from .interface import WMBatch

NOMINAL_ERRORS = {
    "root_pos": 0.01, "root_ori": 0.02, "body_pos": 0.05,
    "body_rot": 0.1, "body_vel": 1.0, "body_avel": 3.14,
}


def state_loss(rep, pred_f, pred_aux, true_f, true_aux):
    pred = rep.loss_blocks(pred_f, pred_aux)
    target = rep.loss_blocks(true_f, true_aux)
    mse = {key: (pred[key] - target[key]).square().mean() for key in pred}
    # Match Hybrid: weighted mean over channels, not equal mean of group means.
    widths = {key: pred[key].shape[-1] for key in pred}
    total = sum(mse[key] * widths[key] / NOMINAL_ERRORS[key]**2 for key in pred)
    return total / sum(widths.values()), mse


def rollout_loss(wm, initial_full, actions, next_full, consistency_weight=1.0, step=None, xi_n=None):
    """Legacy function wrapper; all architecture-specific supervision lives in WM."""
    result = wm.compute_loss(WMBatch(initial_full, actions, next_full, xi_n),
                             consistency_weight, step=step)
    return result.loss, {k: v.detach() for k, v in result.metrics.items()}


LATENT_METRICS = ("wm_latent_entropy", "wm_latent_max_probability", "wm_latent_cosine")


def latent_diagnostics(wm, predicted, target):
    cosine = torch.nn.functional.cosine_similarity(predicted, target, dim=-1).mean()
    if wm.simnorm.dim:
        probs = predicted.reshape(*predicted.shape[:-1], -1, wm.simnorm.dim)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(-1).mean()
        maximum = probs.max(-1).values.mean()
    else:
        entropy = maximum = cosine.new_zeros(())
    return torch.stack((entropy, maximum, cosine))
