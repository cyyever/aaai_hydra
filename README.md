# aaai_hydra

## Introduction

This repository contains the implementation code of
[HYDRA: Hypergradient Data Relevance Analysis for Interpreting Deep Neural Networks](https://arxiv.org/abs/2102.02515).

HYDRA is a method of neural network interpretability that assesses the contribution of training data. You can play [this demo](https://cyyever.github.io/aaai_hydra) to get a feel for its power.

## Installation

### Dependency

#### Supported Operating Systems

Linux and Windows should work for recent versions of PyTorch.

#### Software dependency

```
PyTorch >= 2.1
A C++20 compiler
```

#### Steps to install

Here it's assumed that pip is used as the package manager.

1. Install PyTorch

```
pip3 install torch --user
```

2. Install a PyTorch extension for storing tensors.

```
git clone --recursive git@github.com:cyyever/torch_cpp_extension.git
cd torch_cpp_extension
mkdir build && cd build
cmake -DBUILD_SHARED_LIBS=on ..
cmake --build . --config release
cd ..
env cmake_build_dir=build python3 setup.py install --user
```

3. Install the dependent libraries.

```
pip3 install -r requirements.txt --user
```

## Citation

If you find our work useful, please cite it:

```
@article{chen2021hydra,
  title={Hydra: Hypergradient data relevance analysis for interpreting deep neural networks},
  author={Chen, Yuanyuan and Li, Boyang and Yu, Han and Wu, Pengcheng and Miao, Chunyan},
  journal={arXiv preprint arXiv:2102.02515},
  year={2021}
}
```
