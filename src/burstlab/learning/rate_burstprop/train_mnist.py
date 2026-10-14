"""Quick MNIST check for the rate-model Burstprop exercise: Burstprop vs. backprop on the same network.

AI DISCLOSURE: this file was written by Claude Opus.

Both networks start from identical weights and use the pure update rule (W += Delta W: no momentum, no weight
decay). Each epoch prints test accuracy for both, plus the angle between Burstprop's hidden-layer update and
backprop's, measured on the Burstprop network's current weights.

Run from the repo root (after implementing burstprop.py):

    uv run python src/burstlab/learning/rate_burstprop/train_mnist.py
    uv run python src/burstlab/learning/rate_burstprop/train_mnist.py --hidden 500 500 500 --epochs 10

MNIST downloads to data/ at the repo root unless --data-root points at an existing copy (a directory containing
MNIST/raw/).
"""

import argparse
import math
from itertools import pairwise
from pathlib import Path

import burstprop as bp
import torch
import torch.nn.functional as F
import torchvision

REPO_ROOT = Path(__file__).resolve().parents[4]


def load_mnist(root, train):
    ds = torchvision.datasets.MNIST(root, train=train, download=True)
    x = (
        ds.data.reshape(len(ds), -1).float() / 255
    )  # pixel intensities in [0, 1], used directly as input rates
    return x, ds.targets


def init_net(sizes, seed):
    g = torch.Generator().manual_seed(seed)
    weights = [
        torch.randn(n_out, n_in, generator=g) * math.sqrt(2 / (n_in + n_out))
        for n_in, n_out in pairwise(sizes)
    ]
    biases = [torch.zeros(n_out) for n_out in sizes[1:]]
    return weights, biases


def burstprop_step(x, target, weights, biases, lrs, p_baseline):
    rates = bp.forward(x, weights, biases)
    deltas = bp.burst_deltas(
        rates, target, weights[1:], p_baseline
    )  # symmetric feedback: Y_l = W_{l+1}
    updates = bp.weight_updates(
        deltas, rates, lr=1.0
    )  # unit learning rate; per-layer rates applied below
    for l, (lr, (dW, db)) in enumerate(zip(lrs, updates)):
        weights[l] += lr * dW
        biases[l] += lr * db


def backprop_grads(x, target, weights, biases):
    ws = [w.clone().requires_grad_() for w in weights]
    bs = [b.clone().requires_grad_() for b in biases]
    e = x
    for w, b in zip(ws, bs):
        e = torch.sigmoid(F.linear(e, w, b))
    (
        0.5 * ((e - target) ** 2).sum()
    ).backward()  # summed over the batch, matching the Burstprop rule
    return [w.grad for w in ws], [b.grad for b in bs]


def accuracy(x, labels, weights, biases):
    return (
        (bp.forward(x, weights, biases)[-1].argmax(1) == labels).float().mean().item()
    )


def hidden_angles(x, target, weights, biases, p_baseline):
    """Degrees between Burstprop's update direction and backprop's, per hidden layer."""
    rates = bp.forward(x, weights, biases)
    deltas = bp.burst_deltas(rates, target, weights[1:], p_baseline)
    updates = bp.weight_updates(deltas, rates, lr=1.0)
    grads, _ = backprop_grads(x, target, weights, biases)
    cos = [
        F.cosine_similarity(dW.flatten(), -g.flatten(), dim=0)
        for (dW, _), g in zip(updates[:-1], grads[:-1])
    ]
    return [math.degrees(math.acos(c.clamp(-1, 1).item())) for c in cos]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--hidden", type=int, nargs="+", default=[100], help="hidden layer sizes"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--lr-burstprop",
        type=float,
        default=0.5,
        help="output layer; hidden layers get 4x per layer down",
    )
    parser.add_argument("--lr-backprop", type=float, default=0.1)
    parser.add_argument("--p-baseline", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    x_train, y_train = load_mnist(args.data_root, train=True)
    x_test, y_test = load_mnist(args.data_root, train=False)
    t_train = F.one_hot(y_train, 10).float()

    sizes = [784, *args.hidden, 10]
    w_burst, b_burst = init_net(sizes, args.seed)
    w_back, b_back = [w.clone() for w in w_burst], [b.clone() for b in b_burst]

    # Burstprop's hidden-layer updates shrink by ~1/4 per layer below the output (see test_burstprop.py), so give
    # each layer down 4x the learning rate. This is the per-layer learning rate trick in Payeur's code, made explicit.
    n = len(sizes) - 1
    lrs_burst = [args.lr_burstprop * 4 ** (n - 1 - l) for l in range(n)]

    probe = slice(0, 1000)  # fixed training batch for measuring alignment
    print(f"network {sizes}  burstprop lrs {lrs_burst}  backprop lr {args.lr_backprop}")
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(len(x_train))
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i : i + args.batch_size]
            burstprop_step(
                x_train[idx], t_train[idx], w_burst, b_burst, lrs_burst, args.p_baseline
            )
            grads_w, grads_b = backprop_grads(
                x_train[idx], t_train[idx], w_back, b_back
            )
            for l in range(n):
                w_back[l] -= args.lr_backprop * grads_w[l]
                b_back[l] -= args.lr_backprop * grads_b[l]
        angles = hidden_angles(
            x_train[probe], t_train[probe], w_burst, b_burst, args.p_baseline
        )
        print(
            f"epoch {epoch:2d}  test acc: burstprop {accuracy(x_test, y_test, w_burst, b_burst):.4f}"
            f"  backprop {accuracy(x_test, y_test, w_back, b_back):.4f}"
            f"  hidden-layer angle to backprop (deg): {', '.join(f'{a:.1f}' for a in angles)}"
        )


if __name__ == "__main__":
    main()
