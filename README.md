# PyTorch-Accelerated Multi-Objective Evolutionary Optimization for PET Image Reconstruction

A high-performance, PyTorch-accelerated Multi-Objective Evolutionary Algorithm (MOEAP) framework for optimizing Positron Emission Tomography (PET) image reconstruction.

## Overview
This repository contains a specialized framework that navigates the complex trade-off between clinical image quantification (accurate modeling of tracer uptake) and lesion detectability (improved signal-to-noise ratio in localized regions). It leverages an NSGA-II solver engineered from scratch, integrated with highly parallelized tensor operations on GPU to rapidly evaluate population batches.

Key technical achievements of this codebase:
- **PyTorch-accelerated evolutionary framework** (MOEAP) to jointly optimize lesion detectability and signal quantification.
- **Batched tensor operations** applying customized simulated binary crossover (SBC) and hybrid EM-Polynomial mutations on GPU, drastically accelerating typical MOEA bottlenecks.
- **Automated diagnostic pipeline** featuring adaptive convergence termination, dynamic Pareto quality indicators (Hypervolume, Spacing), and a comprehensive tracking module.
- **Analytical Dashboard** generated automatically per run, providing researchers with immediate visual analysis of clinical trade-offs.

## Setup Instructions

### Prerequisites
Make sure you have an environment with Python 3.8+ installed.

### Installation
1. Clone the repository and navigate into the `PET-image-reconstruction` directory.
2. Install the required Python packages. You can use `pip`:

```bash
pip install numpy scipy matplotlib torch
```
*Note: For maximum performance, ensure that you install a version of `torch` compiled with CUDA support if you intend to run this on an Nvidia GPU, or MPS for Apple Silicon.*

## Usage

The main entry point for the pipeline is `run_moeap_torch.py`.

### Basic Execution
To run the framework with the default settings (synthetic 'heart' phantom, population=40, generations=50):
```bash
python run_moeap_torch.py
```

### Advanced Configuration
You can customize the phantom geometry, evolutionary parameters, and the compute device:
```bash
python run_moeap_torch.py --phantom liver --N 100 --gens 200
```
Force the engine to run on CPU:
```bash
python run_moeap_torch.py --device cpu
```

### Full Arguments List
- `--phantom`: Choose the simulated phantom to reconstruct (`heart` or `liver`). Default: `heart`
- `--img_size`: Matrix size for the reconstructed image (e.g., 64 for 64x64). Default: `64`
- `--n_angles`: Number of projection angles for the Radon transform. Default: `90`
- `--counts`: Total simulated photon counts. Default: `4e5`
- `--N`: Population size for the MOEA. Default: `40`
- `--gens`: Maximum number of generations. Default: `50`
- `--seed`: Random seed for reproducibility. Default: `42`
- `--device`: Target execution device (`auto`, `cuda`, `mps`, `cpu`). Default: `auto`
- `--out`: Directory where tracking metrics and analytical dashboards are saved. Default: `./results_torch`

## Directory Structure
- `run_moeap_torch.py`: Application entry point and orchestrator.
- `moeap_torch.py`: Core PyTorch-accelerated logic for the MOEAP main loop, evaluating populations in batched forward passes.
- `pet_simulator.py`: Utilities for generating phantoms and simulating realistic forward/backward projection data.
- `nsga2_core.py`: Fast vectorised implementation of non-dominated sorting and crowding distance based on NSGA-II.
- `polynomial_mutation.py`: Implements custom hybrid (EM + Polynomial) mutations across population tensors.
- `pet_objectives_torch.py`: Defines the dual objective functions (quantification vs. detectability) natively in PyTorch.
- `termination.py`: Provides heuristics and adaptive criteria for early convergence termination.
- `indicators.py`: Code for measuring multi-objective quality indicators during evolution (e.g., Hypervolume, Spacing).
- `moeap_doctor_view.py` and `moeap_visualize.py`: Dedicated analytical reporting modules for compiling the Pareto trade-offs into clincal visualizations.
