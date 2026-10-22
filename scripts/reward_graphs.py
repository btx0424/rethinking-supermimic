"""Plot Gaussian (paper) vs Cauchy reward kernels against the raw error.

The paper kernel is ``exp(-error² / σ²)``. The Cauchy option is the same
argument with a heavy tail: ``1 / (1 + error² / σ²)``. Both kernels in
``mdp.py`` take the already-squared feature, so this script squares the
x-axis before calling them and differentiates w.r.t. the raw error.

Usage:
  PYTHONPATH=src python scripts/reward_graphs.py
  PYTHONPATH=src python scripts/reward_graphs.py --out logs/reward_kernels.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from supermimic.mdp import (
    REWARD_TERM_SCALE,
    REWARD_TERM_WEIGHT,
    cauchy_kernel,
    gaussian_kernel,
    gaussian_linear_kernel,
)


def _kernel_and_grad(fn, error: torch.Tensor, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """``error`` is the raw error. Kernels consume ``error²``."""
    x = error.detach().clone().requires_grad_(True)
    k = fn(x.square(), scale)
    k.sum().backward()
    return k.detach().cpu().numpy(), x.grad.detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("logs/reward_kernels.png"),
        help="Output figure path",
    )
    parser.add_argument("--xmax", type=float, default=2.0, help="Max raw error on the x-axis")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    x = torch.linspace(0.0, args.xmax, 512)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)

    kernels = (
        (r"gaussian $\exp(-x^{2}/\sigma^{2})$", gaussian_kernel, "#1f77b4", "-"),
        (r"cauchy $1/(1+x^{2}/\sigma^{2})$", cauchy_kernel, "#d62728", "-"),
        ("gaussian+linear", gaussian_linear_kernel, "#2ca02c", "-"),
    )
    ax = axes[0, 0]
    for name, fn, color, ls in kernels:
        k, _ = _kernel_and_grad(fn, x, 0.3)
        ax.plot(x.numpy(), k, color=color, ls=ls, lw=2, label=name)
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
    ax.set_title(r"Kernel value ($\sigma = 0.3$)")
    ax.set_xlabel("error x")
    ax.set_ylabel("k(x)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    for name, fn, color, _ls in kernels:
        _, g = _kernel_and_grad(fn, x, 0.3)
        ax.plot(x.numpy(), g, color=color, lw=2, label=name.split()[0])
    ax.set_title(r"$\partial k/\partial x$ ($\sigma = 0.3$)")
    ax.set_xlabel("error x")
    ax.set_ylabel(r"$\partial k/\partial x$")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    for term, color in (("body_pos", "#2ca02c"), ("body_ori", "#9467bd"), ("root_pos", "#8c564b")):
        sigma = REWARD_TERM_SCALE[term]
        w = REWARD_TERM_WEIGHT["gaussian"][term]
        kg, _ = _kernel_and_grad(gaussian_kernel, x, sigma)
        kc, _ = _kernel_and_grad(cauchy_kernel, x, sigma)
        kl, _ = _kernel_and_grad(gaussian_linear_kernel, x, sigma)
        ax.plot(x.numpy(), w * kg, color=color, lw=1.6, label=rf"gauss {term}")
        ax.plot(x.numpy(), w * kc, color=color, lw=1.4, ls="--", label=f"cauchy {term}")
        ax.plot(x.numpy(), w * kl, color=color, lw=1.6, ls=":", label=f"g+lin {term}")
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
    ax.set_title("Weighted tracking (solid / dashed / dotted)")
    ax.set_xlabel("error x")
    ax.set_ylabel("w · k(x)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=3, frameon=False)

    ax = axes[1, 1]
    # Paper penalties, shared by every reward_style. Feature is already x².
    ax.plot(
        x.numpy(),
        -REWARD_TERM_WEIGHT["gaussian"]["action_rate"] * x.numpy() ** 2,
        color="#1f77b4",
        lw=2,
        label=r"action $-0.1\,x^{2}$",
    )
    ax.plot(
        x.numpy(),
        -REWARD_TERM_WEIGHT["gaussian"]["joint_limit"] * x.numpy() ** 2,
        color="#ff7f0e",
        lw=2,
        ls="--",
        label=r"joint limit $-2\,x^{2}$",
    )
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_title("Action-rate / joint-limit (all styles)")
    ax.set_xlabel("error x")
    ax.set_ylabel("term (before ×dt)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle(r"Reward kernels vs raw error $x$ (Gaussian uses $x^{2}$)", fontsize=13)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
