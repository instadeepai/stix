# STIX: Stochastic Interpolants, A Unifying Framework

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python 3.11](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/downloads/release/python-3110/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Tests and Linters 🧪](https://github.com/instadeepai/stix/actions/workflows/tests_and_linters.yaml/badge.svg?branch=main)](https://github.com/instadeepai/stix/actions/workflows/tests_and_linters.yaml)
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/maximeseince/a96fa297589ba72c83ee54ef6ce4a206/raw/stix-coverage.json)

## 👀 Overview

This repository provides a concise implementation and mathematical guide to **Stochastic Interpolants**, based on the framework introduced in *Stochastic Interpolants: A Unifying Framework for Flows and Diffusions* by Michael S. Albergo, Nicholas M. Boffi, and Eric Vanden-Eijnden.

The goal of this project is to demonstrate how a single, minimal setup can encompass and unify a wide class of generative models, including but not limited to **Flow Matching**, **EDM-style Diffusion**, **Bayesian Flow Networks (BFNs)** and **Discrete Flow Matching**.

We also see this library as a stepping stone for future research: every component is highly modular and can be extended or swapped out with ease, especially when it comes to guidance, coupling, or multi-modality.

Written in JAX, `stix` provides:

- 🌐 Training and sampling of multimodal models
- 🎯 Built-in standard generative frameworks: Flow Matching, Diffusion, and BFN
- 🧭 Guidance for conditional generation
- 🧩 Highly modular components : interpolants, couplings, and guidance are all independently swappable for research liberty
- 📐 Full mathematical infrastructure for stochastic interpolants: interpolant schedules, couplings, and time samplers

Have a look at the [Installation](#-installation) section for details on how to install `stix`. If you want to get started with the library or get a feel of what is possible, you can dive into the [introduction](https://instadeepai.github.io/stix/introduction.html) and the tutorials [notebooks](#-tutorials).

### Why `stix`?

**A framework for nearly every interpolation scheme:** Flow matching, diffusion (e.g. VE and VP), Bayesian Flow Networks, and masked and uniform discrete diffusion all under a single framework. Adding your own is incredibly straightforward.

**Discrete and continuous modalities in one framework, by abstracting over the `Generator`:** Build multimodal models with both discrete and continuous data easily. Truly discrete diffusion runs as a continuous-time Markov chain and continuous data as an ODE or SDE, yet both are sampled simultaneously from the same network call.

**Sampling decoupled from training:** On one-sided paths, target, noise, velocity and score convert in closed form, so a velocity-trained model samples as either an ODE or an SDE with no second head and no retraining.

**Different options for handling discrete data:** We offer mask and uniform diffusion, as well as methods to learn continuous embeddings of discrete data to be used with continuous interpolants.

Have a look at the [Installation](#-installation) section for details on how to install `stix`. If you want to get started with the library or get a feel of what is possible, you can dive into the [introduction](https://instadeepai.github.io/stix/introduction.html) and the tutorials [notebooks](#-tutorials).

We also provide extensive [documentation](https://instadeepai.github.io/stix/api_reference/index.html) of the different classes and components.

## 📦 Installation

Install as a dependency (from PyPI)


```bash
pip install stix-ml
```

Or with `uv`:

```bash
uv add stix-ml
```

### Install for development

```bash
git clone https://github.com/instadeepai/stix.git && cd stix
uv sync --group dev
```

Install the hooks once after cloning:

```bash
uv run pre-commit install
```

They will now run automatically on every commit. To run all hooks against every file manually:

```bash
uv run pre-commit run --all-files
```

## 📓 Tutorials

The [`tutorials/notebooks`](tutorials/notebooks) directory walks through the library end to end. Each notebook is self-contained and provides a thoroughly documented walkthrough.

### Getting started

1. [**Training and sampling**](tutorials/notebooks/1.training_and_sampling.ipynb) : build the full pipeline to train and sample a generative model.

2. [**Multimodal data loading with `grain`**](tutorials/notebooks/2.grain_multimodal_dataloading.ipynb) : feed real and synthetic data into `stix` as `Batch` objects, with checkpointing.

3. [**Building generative models**](tutorials/notebooks/3.generative_model.ipynb) : write your own `GenerativeModel` with custom losses and velocity/score conversions.

### Going further

4. [**Conditioning and guidance**](tutorials/notebooks/4.conditioning_and_guidance.ipynb) : conditional sampling via context and intrinsic guidance recipes, and how to write your own.
5. [**Coupling**](tutorials/notebooks/5.coupling.ipynb) : pair source and target distributions using methods like product-of-marginals, mini-batch OT, rectified flow.
6. [**Discrete models**](tutorials/notebooks/6.discrete_models.ipynb) : train and sample discrete and mixed continuous-discrete models.

## 🙏 Acknowledgments

We would like to thank Krisztina Sinkovics (InstaDeep), Bora Guloglu (InstaDeep), Louis Robinson (InstaDeep) and Shaun Kandathil (InstaDeep) for beta-testing and giving feedback on the iterations of this work.

## 📚 Citing our work

Please cite this repository when using `stix` in your work.

The BibTeX formatted citation:

```bibtex
@software{stix2026,
  author       = {Simons, Jack and Seince, Maxime and Leach, Adam and
                  Brunken, Christoph and Tilly, Jules and Heyraud, Valentin},
  title        = {{stix}: Stochastic Interpolants, A Unifying Framework},
  year         = {2026},
  version      = {0.1.0},
  organization = {InstaDeep},
  license      = {Apache-2.0},
  url          = {https://github.com/instadeepai/stix},
}
```
