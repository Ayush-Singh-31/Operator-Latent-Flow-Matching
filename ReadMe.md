# OL-FM: Operator-Latent Flow Matching for Function-Valued Dynamics

This repository packages the reproducibility bundle for the OL-FM paper, which studies how to learn dynamics whose states are full functions rather than finite-dimensional vectors. OL-FM blends neural operators with continuous-time latent flows and a flow-matching loss, giving a geometry-aware latent representation that evolves in Hilbert space. The code here drives the Navier–Stokes and Burgers benchmarks, the cross-resolution stress test, and the figures reported in the paper.

## Highlights

- Full training pipeline for OL-FM together with Latent ODE, Neural CDE, and Fourier Neural Operator baselines.
- On-the-fly PDE data generation so no external datasets are needed.
- Automated hyper-parameter searches that keep the training-budget identical across models.
- Matplotlib diagnostics that reproduce the plots in the manuscript (error curves, parameter efficiency, spectral comparisons).

## Repository layout

| Path                                              | Description                                                                                                                    |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `Benchmark.py`                                    | Main Navier–Stokes benchmark pipeline: data generation, OL-FM model, baselines, evaluation, and plotting.                      |
| `Benchmark.ipynb`                                 | Notebook walkthrough of the benchmark, mirroring `Benchmark.py` with extra commentary and cells to probe intermediate tensors. |
| `Benchmark.pdf`                                   | Static export of the benchmark notebook for quick viewing.                                                                     |
| `Cross Resolution.py`                             | Burgers cross-resolution benchmark where models are trained and tested on different grids.                                     |
| `Cross Resolution.ipynb` / `Cross Resolution.pdf` | Notebook and PDF companion for the cross-resolution study.                                                                     |
| `Requirements.txt`                                | Locked dependency list used to reproduce the reported runs.                                                                    |

## Requirements and setup

1. Install Python 3.10+ (the scripts rely on modern PyTorch and scientific Python packages).
2. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r Requirements.txt
   ```

3. (Optional) Ensure a CUDA-capable GPU is visible to PyTorch if you want to match the runtimes in the paper. The scripts automatically fall back to CPU.

> `Requirements.txt` intentionally keeps a few comments documenting typos that were fixed when exporting from the internal environment. Leave those comments in place so dependency resolvers know which packages changed.

## Running the experiments

### 1. Navier–Stokes benchmark (`Benchmark.py`)

This script builds a 2D incompressible Navier–Stokes dataset, trains OL-FM together with Latent ODE, Neural CDE, and FNO baselines, and prints/plots report-ready metrics.

```bash
python Benchmark.py
```

What happens:

- Dataset generation (`NSConfig`) samples initial vorticity fields and integrates the PDE spectrally.
- Each model is trained with the same effective number of gradient updates (controlled by `fair_total_updates`).
- Minimal hyper-parameter searches (`hp_trials_per_model`) tune learning rates, hidden widths, etc.
- After testing, the script aggregates MSEs, prints parameter-efficiency tables, and pops up Matplotlib figures (MSE summaries, RMSE vs. parameters, spectral comparisons, etc.).

Tips:

- Edit `NSConfig` to change grid size, viscosity, training budget, or the number of seeds for robustness studies.
- Set `torch.autograd.set_detect_anomaly(True)` is already enabled; disable it for slightly faster runs after debugging.
- Use the saved predictions (`preds_flat_first`) inside the script if you want to dump numpy arrays for custom plotting.

### 2. Burgers cross-resolution benchmark (`Cross Resolution.py`)

This experiment stresses whether the learned models understand function-space structure by training on one grid and evaluating on another.

```bash
python "Cross Resolution.py"
```

Pipeline summary:

- `BurgersConfig` controls the low- and high-resolution grids, viscosity, spectral encoder, and the number of samples.
- The script trains OL-FM-style latent dynamics along with spectral regression baselines, then evaluates on both the training grid and a finer grid (`cross_grid_size`).
- Output includes quantitative summaries (RMSE, spectral errors, H1 norms) plus Matplotlib plots comparing fields and spectra across resolutions.

Adjustments:

- Modify `cfg_base.cross_grid_size` to test extrapolation to even coarser/finer meshes.
- Increase `num_seeds` for more stable averages or decrease it for smoke tests.

## Working with the notebooks

- Open `Benchmark.ipynb` or `Cross Resolution.ipynb` inside Jupyter or VS Code to step through the experiments interactively. Every section mirrors the corresponding script cells, so you can pause after each block to inspect tensors or tweak parameters.
- The PDFs provide a fixed snapshot of the notebooks in case you only need the figures while reading the paper.

## Expected outputs

Both scripts print structured logs similar to those quoted in the paper:

- Seed-by-seed test MSEs followed by aggregate statistics.
- Parameter counts and ratios between OL-FM and each baseline.
- Capacity-adjusted error metrics and relative efficiency scores.
- Field reconstructions and spectral overlays for qualitative inspection.

If you want to capture the figures for the paper, rerun the scripts and save the Matplotlib windows (`File → Save`) or wrap plotting calls with `plt.savefig`.

## Extending the benchmark suite

- Add new baselines by following the template of `train_latent_ode_with_search` or `train_fno_with_search`, then register the model inside the `model_names` list.
- You can plug in alternative PDEs by swapping out the data-generation utilities (`generate_initial_vorticity`, `simulate_ns_2d`, etc.) while keeping the OL-FM training/evaluation loops untouched.
- For multi-resolution or operator-generalization studies, reuse the spectral encoders/decoders already defined in each script.

## Citation

If you use this codebase in academic work, please cite the OL-FM paper:

```
Singh, Ayush. Operator-Latent Flow Matching: Learning Continuous Dynamics in Function-Space Latents. University of Sydney, 2025.
```

The BibTeX entry will be added here once the paper is formally published.
