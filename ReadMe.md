# Operator–Latent Flow Matching (OL-FM)

Operator–Latent Flow Matching (OL–FM) is a research framework for learning continuous-time
dynamics when the system state is a function (e.g., fields, curves, surfaces). Observations are encoded
into a function-space latent that evolves under an operator-valued ODE, trained via flow matching to
deliver stable long-horizon rollouts, constraint handling, and resolution transfer.

## Why OL–FM?
- **Function-space latents:** encode observations into Sobolev/Hilbert space representations instead of
  finite-dimensional vectors, preserving geometry and smoothness.
- **Neural operator vector field:** evolve the latent with a spectral neural operator that can be shared
  across resolutions and domains.
- **Flow matching training:** regress latent velocities along carefully designed interpolation paths,
  removing the need to backpropagate through ODE solvers and improving conditioning.
- **Stability & constraints:** Lipschitz control in latent space enables guarantees about existence and
  uniqueness of the latent flow, while penalties/projections can enforce structural constraints.
- **Cross-domain versatility:** demonstrated on PDE surrogates (1D Burgers, 2D Navier–Stokes) and
  financial term-structure dynamics with fewer constraint violations than vector-latent baselines.

## Project Structure

The codebase is organized into several directories, each containing specific tests and models:

- **Benchmarking/**: Contains benchmark tests (Test 02, Test 03).
- **Cross-Resolution/**: Focuses on cross-resolution experiments (Test 01, Test 04, Test 05, Test 06, Test 08).
- **Financial Model/**: Includes the financial model implementation (Test 07) and asset generation scripts.
- **Gadi Tests/**: Tests specifically designed or run on the Gadi supercomputer (Test 09, Test 10).

## Installation

To set up the environment, install the required dependencies using pip:

```bash
pip install -r Requirements.txt
```

## Usage

To run a specific test, navigate to the corresponding directory and execute the Python script. For example, to run Test 03 in the Benchmarking directory:

```bash
cd Benchmarking
python "Test 03.py"
```

Please ensure you have the necessary data and permissions if running tests that require specific resources.