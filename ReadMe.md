# OL-FM: Operator-Latent Flow Matching for Function-Valued Dynamics

This repository contains the reference implementation that accompanies the paper on Operator-Latent Flow Matching (OL-FM) for learning dynamics of function-valued states. The project studies dynamical systems where each state is a function rather than a finite-dimensional vector, and introduces a latent model that evolves in a Hilbert space while respecting underlying function-space geometry.

OL-FM combines neural operators with continuous-time latent dynamics and flow matching. The model represents each input field as an element of a Hilbert space and learns an operator-valued vector field that drives latent trajectories between source and target functions. Training is performed using a flow matching objective, which aligns the model’s induced probability flow with simple reference paths in latent space. This viewpoint allows OL-FM to benefit from both operator learning for partial differential equations and continuous-time latent models, while providing a more structured latent space than standard vector embeddings.

The experiments in this repository evaluate OL-FM on fluid and transport benchmarks. For two-dimensional incompressible Navier–Stokes in vorticity form, OL-FM is compared against established baselines such as Fourier Neural Operators and latent vector models, with a focus on accuracy, parameter efficiency, and stability of long-horizon rollouts. For one-dimensional Burgers equations, the cross-resolution setting probes how OL-FM handles changes in spatial discretisation, highlighting its ability to work in a function-space representation rather than being tied to a fixed grid.

## Repository contents

Benchmark.py contains the main implementation of OL-FM for the Navier–Stokes benchmark together with baseline architectures, data generation on periodic grids, training loops, and evaluation routines used in the paper. This script is the core reference for the operator-latent flow matching model in the fluid dynamics setting.

Benchmark.ipynb is an interactive notebook that mirrors the benchmark experiments, allowing step-by-step inspection of model components, training behaviour, and diagnostic plots such as error curves, field reconstructions, and spectral summaries.

Benchmark.pdf provides a static, notebook-style report for the benchmark experiments, including figures and tables used to summarise the behaviour of OL-FM and the comparison models.

Cross Resolution.py implements the Burgers cross-resolution experiments. It focuses on learning dynamics at one spatial resolution and assessing behaviour when evaluated at a different resolution, illustrating how OL-FM operates in a discretisation-agnostic function space.

Cross Resolution.ipynb is the corresponding notebook for the cross-resolution study, collecting the main plots and numerical results that demonstrate how the model extrapolates across grids in the Burgers setting.

Cross Resolution.pdf is a report version of the cross-resolution experiments, suitable for quick inspection of the figures and numerical summaries without running the code.

Requirements.txt lists the main Python dependencies needed to reproduce the experiments and analysis associated with the OL-FM paper.

## Intended use

This repository is intended as a research reference for readers interested in operator learning, flow matching, and function-space latent models. It provides a concrete implementation of OL-FM in canonical PDE benchmarks, along with the exact scripts and notebooks that support the empirical results reported in the paper. Researchers can use it to better understand the modelling choices, experimental setup, and diagnostic checks used to evaluate operator-based latent dynamics.
