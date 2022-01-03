# aaai_hydra

## Introduction

This repository contains the implementation code of
[HYDRA: Hypergradient Data Relevance Analysis for Interpreting Deep Neural Networks](https://arxiv.org/abs/2102.02515).

HYDRA is a method of neural network interpretability that assesses the contribution of training data. You can play [this demo](https://cyyever.github.io/aaai_hydra) to get a feel for its power.

## Installation

### Dependency

#### Supported Operating Systems

Windows is not supported yet. You need a recent Linux distribution.

#### Software dependency

```
pytorch >= 1.7
torchvision >= v0.8.0
a C++20 compiler

```

#### Steps to install

1. Install pytorch according to [the instructions](https://pytorch.org/).
2. Install torchvision according to [the instructions](https://github.com/pytorch/vision).
3. Install GSL according to [the instructions](https://github.com/microsoft/GSL).
4. Install doctest according to [the instructions](https://github.com/doctest/doctest).
5. Install spdlog according to [the instructions](https://github.com/gabime/spdlog).
6. Install pybind11 according to [the instructions](https://github.com/pybind/pybind11).

7.

```
python3 -m pip install --upgrade --user git+ssh://git@github.com/cyyever/naive_python_lib.git@main
```

8.

```
git clone --recursive git@github.com:cyyever/naive_cpp_lib.git
cd naive_cpp_lib
mkdir build && cd build
cmake -DBUILD_SHARED_LIBS=on ..
sudo make install
```

9.

```
git clone --recursive git@github.com:cyyever/algorithm.git
cd algorithm
mkdir build && cd build
cmake -DBUILD_SHARED_LIBS=on ..
sudo make install
```

10.

```
git clone --recursive git@github.com:cyyever/torch_cpp_extension.git
cd torch_cpp_extension
mkdir build && cd build
cmake -DBUILD_SHARED_LIBS=on ..
sudo make install
```

11.

```
git clone --recursive git@github.com:cyyever/torch_toolbox.git
cd cyy_torch_toolbox
python3 setup.py install --user
```

12.

```
git clone --recursive git@github.com:cyyever/torch_algorithm.git
cd cyy_torch_algorithm
python3 setup.py install --user
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
