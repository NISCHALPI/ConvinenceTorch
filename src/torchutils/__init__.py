"""torchutils package.

A package containing utility modules and classes for PyTorch.

Package Structure:
==================
- trainer: Contains the implementation of the NNtrainer class, which is a trainer class for neural networks.

Modules:
========
- trainer: Module containing the implementation of the NNtrainer class, which is a trainer class for neural networks.
    - NNtrainer: A trainer class for neural networks.

Description:
============

The torchutils package provides a collection of utility modules and classes designed to enhance the functionality and ease of use of PyTorch, a popular deep learning framework. It includes the `trainer` module, which contains the implementation of the NNtrainer class.

The NNtrainer class is a flexible and extensible trainer class specifically designed for training neural networks in PyTorch. It provides an abstraction layer that handles common training tasks such as model initialization, optimization, loss computation, device management, and training/validation loops. By using the NNtrainer class, developers can focus on defining their models and customizing the training process without worrying about the boilerplate code.

The torchutils package aims to simplify the process of training neural networks in PyTorch by providing a high-level interface and convenient abstractions. It promotes code reusability, modularity, and extensibility, allowing developers to build and train complex models efficiently.

Usage:
import torchutils

# Access the modules within the package
from torchutils.trainer import NNtrainer

# Use the modules and classes
trainer = NNtrainer(model, optimizer, loss)

"""
from .skeletons import configure_logging

# Configures the loggin module
configure_logging()

# Version
__version__ = "1.0.0"
