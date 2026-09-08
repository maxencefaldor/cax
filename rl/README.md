# rl

Reinforcement learning for the training of neural cellular automata, as a research programme on top of CAX.

## The map

```text
rl/
├── README.md              # this file
└── notes/
    ├── baseline.md        # what the three existing trainers reach on the gecko, and what 63 already is
    ├── literature.md      # the survey: RL for NCA, analytic policy gradients, the NCA frontier
    ├── recipes.md         # implementation facts from the full text of the papers the experiments copy
    └── experiments.md     # the proposed experiments, ranked, each with its question and its verdict rule
```

The reference implementations are the example notebooks: [40](../examples/40_growing_nca.ipynb) trains a growing NCA by backpropagation through the developmental rollout, [62](../examples/62_growing_nca_es.ipynb) by evolution strategies, and [63](../examples/63_growing_nca_rl.ipynb) by a short-horizon actor-critic whose gradient flows analytically through the rollout and a learned value function.
