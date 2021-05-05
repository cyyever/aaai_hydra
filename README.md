# aaai_hydra

## Introduction

This repository contains the implementation code of
[HYDRA: Hypergradient Data Relevance Analysis for Interpreting Deep Neural Networks](https://arxiv.org/abs/2102.02515).

## Installation

### Dependency

#### Supported Operating Systems

Windows is not supported yet. You need a recent Linux distribution.

#### Software dependency

```
pytorch >= 1.7
torchvision >= v0.8.0
gcc >= 10.2
```

#### Steps to install

1. Install pytorch >= 1.7 according to [the instructions](https://pytorch.org/)
2. Install torchvision >= v0.8.0 according to [the instructions](https://github.com/pytorch/vision)

3.

```
git clone --recursive git@github.com:cyyever/naive_python_lib.git
cd naive_python_lib
pip3 install -r requirements.txt --user
python3 setup.py install --user
```

4.

```
git clone --recursive git@github.com:cyyever/naive_cpp_lib.git
cd naive_cpp_lib
mkdir build && cd build
cmake -DBUILD_TORCH=on -DBUILD_TORCH_PYTHON_BINDING=on -DBUILD_SHARED_LIBS=on ..
sudo make install
```

5.

```
git clone --recursive git@github.com:cyyever/naive_pytorch_lib.git
cd naive_pytorch_lib
python3 setup.py install --user
```

## Citation

@article{chen2021hydra,
title={Hydra: Hypergradient data relevance analysis for interpreting deep neural networks},
author={Chen, Yuanyuan and Li, Boyang and Yu, Han and Wu, Pengcheng and Miao, Chunyan},
journal={arXiv preprint arXiv:2102.02515},
year={2021}
}
