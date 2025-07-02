# Surface-Reconstruct (Unofficial PyTorch Implementation)

This repository contains an **unofficial PyTorch implementation** of the paper  
**"Learning to Reconstruct Surfaces from Point Clouds in Both Euclidean and Riemannian Spaces"**  
by Zhu et al. (CVPR 2021).  
[[arXiv:2106.07689](https://arxiv.org/abs/2106.07689)]

## Overview

The original paper proposes a novel method for reconstructing surfaces from point clouds in both Euclidean and Riemannian spaces using deep learning techniques. This implementation aims to replicate the main ideas and results of the paper using PyTorch.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/mr-sarthakgupta/surface-reconstruct.git
   cd surface-reconstruct
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
# Example command to train or evaluate the model
python train.py --config configs/your_config.yaml
```

- Please refer to the code and configuration files for details on available arguments and settings.
