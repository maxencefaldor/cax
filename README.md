# CAX: Cellular Automata Accelerated in JAX

<div align="center">
	<img src="docs/cax.png" alt="logo" width="512"></img>
</div>

[![Pyversions](https://img.shields.io/pypi/pyversions/cax.svg?style=flat)](https://pypi.python.org/pypi/cax)
[![PyPI version](https://badge.fury.io/py/cax.svg)](https://badge.fury.io/py/cax)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Paper](http://img.shields.io/badge/paper-arxiv.2410.02651-B31B1B.svg)](https://arxiv.org/abs/2410.02651)
[![X](https://img.shields.io/badge/X-%23000000.svg?style=for-the-badge&logo=X&logoColor=white&style=flat)](https://x.com/maxencefaldor/status/1842211478796918945)

CAX is a high-performance and flexible open-source library designed to **accelerate artificial life research**. 🧬

## Overview 🔎

Are you interested in emergence, self-organization, or open-endedness? Whether you're a researcher or just curious about the fascinating world of artificial life, CAX is your digital lab! 🔬

Designed for speed and flexibility, CAX allows you to easily experiment with self-organizing behaviors and emergent phenomena. 🧑‍🔬

**Get started here** [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/00_getting_started.ipynb)

## Why CAX? 💡

CAX supports discrete and continuous systems, including neural cellular automata, across any number of dimensions. Beyond traditional cellular automata, it also handles particle systems and more, all unified under a single, intuitive API.

### Rich 🎨

CAX provides a comprehensive collection of 15+ ready-to-use systems. From simulating one-dimensional [elementary cellular automata](examples/10_elementary_ca.ipynb) to training three-dimensional [self-autoencoding neural cellular automata](examples/45_self_autoencoding_mnist.ipynb), or even creating beautiful [Lenia](examples/20_lenia.ipynb) simulations, CAX provides a versatile platform for exploring the rich world of self-organizing systems.

### Flexible 🧩

CAX makes it easy to extend existing systems or build custom ones from scratch for endless experimentation and discovery. Design your own experiments to probe the boundaries of artificial open-ended evolution and emergent complexity.

### Fast 🚀

CAX is built on top of the JAX/Flax ecosystem for speed and scalability. The library benefits from vectorization and parallelization on various hardware accelerators such as CPU, GPU, and TPU. This allows you to scale your experiments from small prototypes to massive simulations with minimal code changes.

### Tested & Documented 📚

The library is thoroughly tested and documented with numerous examples to get you started! Our comprehensive guides walk you through everything from basic cellular automata to advanced neural implementations.

## Implemented Systems 🦎

| Cellular Automata | Reference | Example |
| --- | --- | --- |
| Elementary Cellular Automata | [Wolfram (2002)](https://www.wolframscience.com/nks/) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/10_elementary_ca.ipynb) |
| Conway's Game of Life | [Gardner (1970)](https://web.stanford.edu/class/sts145/Library/life.pdf) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/11_life.ipynb) |
| Lenia | [Chan (2020)](https://arxiv.org/abs/2005.03742) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/20_lenia.ipynb) |
| Flow Lenia | [Plantec et al. (2022)](https://arxiv.org/abs/2212.07906) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/21_flow_lenia.ipynb) |
| Particle Lenia | [Mordvintsev et al. (2022)](https://google-research.github.io/self-organising-systems/particle-lenia/) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/22_particle_lenia.ipynb) |
| Particle Life | [Mohr (2018)](https://particle-life.com/) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/30_particle_life.ipynb) |
| Boids | [Reynolds (1987)](https://www.red3d.com/cwr/boids/) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/31_boids.ipynb) |
| Growing Neural Cellular Automata | [Mordvintsev et al. (2020)](https://distill.pub/2020/growing-ca/) |[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/40_growing_nca.ipynb) |
| Growing Conditional Neural Cellular Automata | [Sudhakaran et al. (2022)](http://arxiv.org/abs/2205.06806) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/41_growing_conditional_nca.ipynb) |
| Growing Unsupervised Neural Cellular Automata | [Palm et al. (2021)](https://arxiv.org/abs/2201.12360) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/42_growing_unsupervised_nca.ipynb) |
| Diffusing Neural Cellular Automata | [Faldor et al. (2024)](https://arxiv.org/abs/2410.02651) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/43_diffusing_nca.ipynb) |
| Self-classifying MNIST Digits | [Randazzo et al. (2020)](https://distill.pub/2020/selforg/mnist/) |[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/44_self_classifying_mnist.ipynb) |
| Self-autoencoding MNIST Digits | [Faldor et al. (2024)](https://arxiv.org/abs/2410.02651) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/45_self_autoencoding_mnist.ipynb) |
| 1D-ARC Neural Cellular Automata | [Faldor et al. (2024)](https://arxiv.org/abs/2410.02651) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/46_1d_arc_nca.ipynb) |
| Attention-based Neural Cellular Automata | [Tesfaldet et al. (2022)](https://arxiv.org/abs/2211.01233) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/47_attention_nca.ipynb) |

## Getting Started 🚦

Here, you can see the basic CAX API usage:

```python
import jax
from flax import nnx

from cax.core.ca import CA
from cax.core.perceive import ConvPerceive
from cax.core.update import NCAUpdate

seed = 0

channel_size = 16
num_kernels = 3
hidden_layer_sizes = (128,)
cell_dropout_rate = 0.5

key = jax.random.key(seed)
rngs = nnx.Rngs(seed)

perceive = ConvPerceive(
	channel_size=channel_size,
	perception_size=num_kernels * channel_size,
	rngs=rngs,
	feature_group_count=channel_size,
)
update = NCAUpdate(
	channel_size=channel_size,
	perception_size=num_kernels * channel_size,
	hidden_layer_sizes=hidden_layer_sizes,
	rngs=rngs,
	cell_dropout_rate=cell_dropout_rate,
	zeros_init=True,
)
ca = CA(perceive, update)

state = jax.random.normal(key, (64, 64, channel_size))
state, metrics = ca(state, num_steps=128)
```

For a more detailed overview, get started with this notebook [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/00_getting_started.ipynb)

## Installation ⚙️

You will need Python 3.10 or later, and a working JAX installation installed in a virtual environment.

Then, install CAX from PyPi with `uv`:
```
uv pip install cax
```

or with `pip`:
```
pip install cax
```

## Citing CAX 📝

If you use CAX in your research, please cite the following paper:

```bibtex
@inproceedings{cax,
	title       = {{CAX}: Cellular Automata Accelerated in {JAX}},
	author      = {Maxence Faldor and Antoine Cully},
	booktitle   = {The Thirteenth International Conference on Learning Representations},
	year        = {2025},
	url         = {https://openreview.net/forum?id=o2Igqm95SJ},
	keywords    = {artificial life, emergence, self-organization, open-endedness, cellular automata, neural cellular automata},
}
```

## Contributing 👷

Contributions are welcome! If you find a bug or are missing your favorite self-organizing system, please open an issue or submit a pull request following our [contribution guidelines](CONTRIBUTING.md) 🤗.
