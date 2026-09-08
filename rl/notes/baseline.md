# Baseline

What the three trainers of CAX reach on the same task, and what the RL notebook already is.
Numbers are from the executed notebooks in `docs/build/notebooks/` (built 2026-09-04 on one H100).

## The task

Grow the gecko emoji at 40 px in a 72x72 grid (pad 16) from the official seed (alpha and hidden channels at one), 16 channels, 3 kernels (identity and Sobel), one hidden layer of 128, cell dropout 0.5, 8,768 trainable parameters.
The measurement is the mean squared error of the RGBA channels against the target, averaged over the grid.
A blank canvas scores 0.0305.

## The three trainers

| Notebook | Method | Train steps | Wall clock | Final MSE |
| --- | --- | --- | --- | --- |
| 40 | backpropagation through 128 steps, loss at a random step of the second half, pool of 1,024 | 8,192 | 398 s | 1.1e-4 (batch loss) |
| 62 | Open-ES, population 512, std 0.008, loss on the second half of the rollout, no pool | 5,000 gens | 108 min | 3.4e-3 |
| 63 | actor-critic, horizon 64, discount 0.999, lambda 0.95, pool of 1,024 with 16 resets per batch of 32 | 16,384 | 1,703 s | 2.4e-4 (pool MSE) |

The two MSEs of 40 and 63 are not the same measurement: 40 reports the training batch at a random step in [64, 128), 63 reports the whole pool, whose states are at every developmental age.
An equal-footing evaluation (fresh seeds, 256 steps, MSE at every step) is the first thing to build.

## What 63 already is

The notebook is not model-free reinforcement learning.
It is a short-horizon actor-critic with analytic gradients, the design of SHAC (Xu et al., 2022) and of the Dreamer actor: the rollout is truncated at 64 of 128 steps, the return past the horizon is a learned critic, the critic regresses to lambda-returns from a Polyak target copy, and the actor maximizes the bootstrapped return by differentiating through the cellular automaton and through the critic.
The critic is a global strided CNN with spatial mean pooling, bounded negative through a softplus, and its bootstrap is ramped in over 200 steps.
Stability came from three things, each validated by a probe in `docs/build/probes/63_probe_*.ipynb`: discount 0.999 (sharper discounts exploded mid-training), zero_nans and a global clip at 0.5 after per-tensor gradient normalization, and 16 pool resets per batch of 32.

What it has not shown yet is the claim it is built on: that the cost of training is independent of the developmental length.
At horizon 64 of 128 it is two times worse and four times slower than backpropagation, and no shorter horizon has been tried.
