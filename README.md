# CAX: Cellular Automata Accelerated in JAX

<div align="center">
	<img src="https://raw.githubusercontent.com/maxencefaldor/cax/main/docs/assets/cax.png" alt="logo" width="448"></img>
</div>

<div align="center">
	<a href="https://pypi.python.org/pypi/cax"><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/cax.svg?style=flat"></img></a>
	<a href="https://pypi.python.org/pypi/cax"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/cax.svg?style=flat"></img></a>
	<a href="https://arxiv.org/abs/2410.02651"><img alt="Paper" src="http://img.shields.io/badge/paper-arxiv.2410.02651-B31B1B.svg"></img></a>
	<a href="https://x.com/maxencefaldor/status/1842211478796918945"><img alt="X URL" src="https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2Fmaxencefaldor%2Fstatus%2F1842211478796918945"></img></a>
</div>

CAX is a high-performance and flexible open-source library designed to **accelerate artificial life research** — cellular automata, particle systems, and other self-organizing complex systems, all in JAX. 🧬

## Overview 🔎

Are you interested in emergence, self-organization, or open-endedness? Whether you're a researcher or just curious about the fascinating world of artificial life, CAX is your digital lab! 🔬

Designed for speed and flexibility, CAX allows you to easily experiment with self-organizing behaviors and emergent phenomena. 🧑‍🔬

**Get started here** [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/00_getting_started.ipynb)

## Why CAX? 💡

CAX supports discrete and continuous systems, including neural cellular automata, across any number of dimensions. Beyond traditional cellular automata, it also handles particle systems and more, all unified under a single, intuitive API.

### Rich 🎨

CAX provides a comprehensive collection of 25+ ready-to-use systems. From simulating one-dimensional [elementary cellular automata](examples/10_elementary.ipynb) to training three-dimensional [self-autoencoding neural cellular automata](examples/45_self_autoencoding_mnist.ipynb), or even creating beautiful [Lenia](examples/20_lenia.ipynb) simulations, CAX provides a versatile platform for exploring the rich world of self-organizing systems.

### Flexible 🧩

CAX makes it easy to extend existing systems or build custom ones from scratch for endless experimentation and discovery. Design your own experiments to probe the boundaries of artificial open-ended evolution and emergent complexity.

### Fast 🚀

CAX is built on top of the JAX/Flax ecosystem for speed and scalability. The library benefits from vectorization and parallelization on various hardware accelerators such as CPU, GPU, and TPU. This allows you to scale your experiments from small prototypes to massive simulations with minimal code changes.

### Tested & Documented 📚

The library is thoroughly tested and [documented](https://maxencefaldor.github.io/cax/) with numerous examples to get you started! Our comprehensive guides walk you through everything from basic cellular automata to advanced neural implementations.

## Examples 📓

| Example | Reference | Colab |
| --- | --- | --- |
| [Elementary Cellular Automata](examples/10_elementary.ipynb) | [Wolfram (2002)](https://www.wolframscience.com/nks/) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/10_elementary.ipynb) |
| [Conway's Game of Life](examples/11_life.ipynb) | [Gardner (1970)](https://web.stanford.edu/class/sts145/Library/life.pdf) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/11_life.ipynb) |
| [Langton's Ant](examples/12_langton_ant.ipynb) | [Langton (1986)](https://doi.org/10.1016/0167-2789(86)90237-X) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/12_langton_ant.ipynb) |
| [Abelian Sandpile](examples/13_sandpile.ipynb) | [Bak et al. (1987)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.59.381) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/13_sandpile.ipynb) |
| [Lenia](examples/20_lenia.ipynb) | [Chan (2020)](https://arxiv.org/abs/2005.03742) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/20_lenia.ipynb) |
| [Flow Lenia](examples/21_flow_lenia.ipynb) | [Plantec et al. (2022)](https://arxiv.org/abs/2212.07906) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/21_flow_lenia.ipynb) |
| [Particle Lenia](examples/22_particle_lenia.ipynb) | [Mordvintsev et al. (2022)](https://google-research.github.io/self-organising-systems/particle-lenia/) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/22_particle_lenia.ipynb) |
| [Reaction-Diffusion](examples/23_reaction_diffusion.ipynb) | [Gray & Scott (1984)](https://doi.org/10.1016/0009-2509(84)87017-7) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/23_reaction_diffusion.ipynb) |
| [Particle Life](examples/30_particle_life.ipynb) | [Mohr (2018)](https://particle-life.com/) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/30_particle_life.ipynb) |
| [Boids](examples/31_boids.ipynb) | [Reynolds (1987)](https://www.red3d.com/cwr/boids/) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/31_boids.ipynb) |
| [Growing Neural Cellular Automata](examples/40_growing_nca.ipynb) | [Mordvintsev et al. (2020)](https://distill.pub/2020/growing-ca/) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/40_growing_nca.ipynb) |
| [Growing Conditional Neural Cellular Automata](examples/41_growing_conditional_nca.ipynb) | [Sudhakaran et al. (2022)](http://arxiv.org/abs/2205.06806) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/41_growing_conditional_nca.ipynb) |
| [Growing Unsupervised Neural Cellular Automata](examples/42_growing_unsupervised_nca.ipynb) | [Palm et al. (2021)](https://arxiv.org/abs/2201.12360) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/42_growing_unsupervised_nca.ipynb) |
| [Diffusing Neural Cellular Automata](examples/43_diffusing_nca.ipynb) | [Faldor et al. (2024)](https://arxiv.org/abs/2410.02651) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/43_diffusing_nca.ipynb) |
| [Self-classifying MNIST Digits](examples/44_self_classifying_mnist.ipynb) | [Randazzo et al. (2020)](https://distill.pub/2020/selforg/mnist/) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/44_self_classifying_mnist.ipynb) |
| [Self-autoencoding MNIST Digits](examples/45_self_autoencoding_mnist.ipynb) | [Faldor et al. (2024)](https://arxiv.org/abs/2410.02651) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/45_self_autoencoding_mnist.ipynb) |
| [Texture Neural Cellular Automata](examples/46_texture_nca.ipynb) | [Niklasson et al. (2021)](https://distill.pub/selforg/2021/textures/) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/46_texture_nca.ipynb) |
| [1D-ARC Neural Cellular Automata](examples/47_1d_arc_nca.ipynb) | [Faldor et al. (2024)](https://arxiv.org/abs/2410.02651) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/47_1d_arc_nca.ipynb) |
| [Attention-based Neural Cellular Automata](examples/48_attention_nca.ipynb) | [Tesfaldet et al. (2022)](https://arxiv.org/abs/2211.01233) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/48_attention_nca.ipynb) |
| [Isotropic Neural Cellular Automata](examples/49_isotropic_nca.ipynb) | [Mordvintsev et al. (2022)](https://arxiv.org/abs/2205.01681) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/49_isotropic_nca.ipynb) |
| [Differentiable Logic Cellular Automata](examples/50_difflogic_ca.ipynb) | [Miotti et al. (2025)](https://google-research.github.io/self-organising-systems/difflogic-ca/) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/50_difflogic_ca.ipynb) |
| [Variational Autoencoder](examples/60_vae.ipynb) | [Kingma & Welling (2013)](https://arxiv.org/abs/1312.6114) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/60_vae.ipynb) |
| [Recurrent Residual Convolutional Neural Network](examples/61_rrcnn.ipynb) |  | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/61_rrcnn.ipynb) |
| [Growing Neural Cellular Automata with Evolution Strategies](examples/62_growing_nca_es.ipynb) |  | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/62_growing_nca_es.ipynb) |
| [Growing Neural Cellular Automata with Reinforcement Learning](examples/63_growing_nca_rl.ipynb) |  | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/63_growing_nca_rl.ipynb) |
| [Leniabreeder](examples/64_leniabreeder.ipynb) | [Faldor & Cully (2024)](https://arxiv.org/abs/2406.04235) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/64_leniabreeder.ipynb) |
| [Gradient Descent in Lenia](examples/65_lenia_grad.ipynb) |  | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/65_lenia_grad.ipynb) |
| [Lenia Gradients in Depth](examples/66_lenia_grad_in_depth.ipynb) |  | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/66_lenia_grad_in_depth.ipynb) |

## Getting Started 🚦

Here, you can see the basic CAX API usage with Conway's Game of Life:

```python
import jax
import jax.numpy as jnp

from cax.cs.life import Life

seed = 0

num_steps = 128
spatial_dims = (32, 32)
channel_size = 1
rule_golly = "B3/S23"  # Conway's Game of Life

key = jax.random.key(seed)

birth, survival = Life.birth_survival_from_string(rule_golly)
cs = Life(birth=birth, survival=survival)

state_init = jax.random.bernoulli(key, p=0.5, shape=(*spatial_dims, channel_size)).astype(
	jnp.float32
)
state_final, states = cs(state_init, num_steps=num_steps, return_states=True)
```

For a more detailed overview, get started with this notebook [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/00_getting_started.ipynb)

## Installation ⚙️

You will need Python 3.12 or later, and a working JAX installation installed in a virtual environment.

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
	title = {{CAX}: {Cellular} {Automata} {Accelerated} in {JAX}},
	volume = {2025},
	url = {https://proceedings.iclr.cc/paper_files/paper/2025/file/19206a6ed5ed0aaeed440448dfc5cf7e-Paper-Conference.pdf},
	booktitle = {International {Conference} on {Representation} {Learning}},
	author = {Faldor, Maxence and Cully, Antoine},
	editor = {Yue, Y. and Garg, A. and Peng, N. and Sha, F. and Yu, R.},
	year = {2025},
	pages = {8947--8960},
	keywords = {artificial life, emergence, self-organization, open-endedness, cellular automata, neural cellular automata},
}
```

## Contributing 👷

Contributions are welcome! If you find a bug or are missing your favorite self-organizing system, please open an issue or submit a pull request following our [contribution guidelines](https://maxencefaldor.github.io/cax/contributing/) 🤗.
