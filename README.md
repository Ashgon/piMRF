# Rapid Spatio-temporal MR Fingerprinting Using Physics-Informed Implicit Neural Representation


This repository provides the official implementation of the paper: Rapid Spatio-temporal MR Fingerprinting Using Physics-Informed Implicit Neural Representation (coming soon)

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Run](#Run)
- [Citation](#citation)
- [Contacts](#contacts)

## Introduction
πMRF (Physics-informed implicit neural MRF) is a physics-informed unsupervised framework for accurate quantitative parameter mapping via global spatio-temporal inversion.


## Project Structure

The main components of this repository are organized as follows:

```
piMRF/
├── main.py              # Runnable demo script for running πMRF reconstruction.
├── model/               # Core implementation of networks and solvers.
│   ├── model.py         # Network architectures (NablaBlochNet, DinerSiren) and loss functions.
│   └── piMRF.py         # πMRF reconstruction solver implementation.
├── configs/             # Configuration files.
│   ├── config.json      # Parameter settings for the reconstruction.
│   └── piMRF_config.py  # Configuration loading and merging logic.
├── utils/               # Utility functions and operators.
│   ├── utils.py         # Data processing and visualization utilities.
│   ├── SIM_EPG.py       # Bloch simulation (EPG) implementation.
│   └── Nufft_multi.py   # Multi-coil NUFFT operators.
├── data/                # Example datasets including k-space, trajectories, and maps.
└── results/             # Directory for saving reconstruction outputs and logs.
```

## Getting Started
The hardware and software environment we tested:
- OS: Ubuntu 22.04.5 LTS
- CPU: Intel(R) Xeon(R) Gold 6258R CPU @ 2.70GHz
- GPU: NVIDIA A40 48GB
- CUDA: 13.0
- PyTorch: 2.7.1
- Python: 3.11.13


### Installation
0. Download and Install the appropriate version of NVIDIA driver and CUDA for your GPU.
1. Download and install [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.anaconda.com/miniconda/).
2. Clone this repo and cd to the project path.
```bash
git clone https://github.com/Ashgon/piMRF.git
cd piMRF
```
3. Create and activate the Conda environment:
```bash
conda create --name piMRF python=3.11.13
conda activate piMRF
```
4. Install dependencies:
```bash
pip install -r requirements.txt
```


### Run

To run the reconstruction demo, please use the following command:

```bash
python main.py
```

Reconstruction results are written to the `results/` folder.

## Citation
If you find this code useful, please cite our work:

```
Coming soon.
```


### Contacts
* Chaoguang Gong (23b305002@stu.hit.edu.cn)
* [Haifeng Wang](https://maic.siat.ac.cn/siat/2025-02/28/article_2025022802085936381.html) (hf.wang1@siat.ac.cn)
* [Yue Hu](https://lab.rjmart.cn/10958/imip) (huyue@hit.edu.cn)
