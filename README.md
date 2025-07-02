# Implicit Surface Reconstruction Using SDF

This repository implements neural implicit surface reconstruction using Signed Distance Functions (SDF). It allows you to train a neural network to reconstruct 3D surfaces from point clouds or mesh data. The core training and evaluation routines are in `phasetask_run.py`, and there is also an exploratory Jupyter notebook (`PHASETask.ipynb`) for interactive development and visualization.

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-enabled GPU recommended for training (optional)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mr-sarthakgupta/surface-reconstruct.git
   cd surface-reconstruct
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Training and Evaluation

The main script for training and evaluating the surface reconstruction neural network is `phasetask_run.py`. You can run it as follows:

```bash
python phasetask_run.py
```

- The script trains the model, reconstructs the surface, and saves intermediate meshes and model checkpoints in the `intermediates/` and `trained_models/` directories, respectively.
- Evaluates reconstruction using Chamfer distance.

#### Jupyter Notebook

For interactive experimentation and visualization, use the `PHASETask.ipynb` notebook:

```bash
jupyter notebook PHASETask.ipynb
```

### Directory Structure

- `phasetask_run.py` — Main training and evaluation script.
- `PHASETask.ipynb` — Interactive notebook for development and visualization.
- `find_low.py` — Utility to find the mesh with the lowest Chamfer distance.
- `requirements.txt` — Dependency list.

