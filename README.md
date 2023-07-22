[![Static Badge](https://img.shields.io/badge/pytorch-v2.x-red)](https://pytorch.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 



# ConvinenceTorch - A Lightweight and Minimalistic Convenience Library for PyTorch
<img src="./Logo.jpeg" alt="Logo" width="300">

ConvinenceTorch is a Python package that serves as a lightweight and minimalistic convenience library for PyTorch, a popular deep learning framework. It provides a collection of utility modules and classes to simplify and streamline the training process of neural networks in PyTorch.

## Features

- **NNtrainer**: ConvinenceTorch offers a flexible and extensible `NNtrainer` class designed for training neural networks in PyTorch. The `NNtrainer` handles various common training tasks, such as model initialization, optimization, loss computation, device management, and training/validation loops. By utilizing the `NNtrainer` class, developers can focus on defining their models and customizing the training process without being burdened by boilerplate code.

- **Memory Usage Report**: ConvinenceTorch provides a memory usage report feature that allows users to monitor the memory consumption during the training process. The mem_report() function in NNtrainer provides valuable insights into the memory usage, helping users optimize their models and data loaders efficiently.

- **Metrics and Loss capture**: ConvinenceTorch offers a comprehensive and flexible mechanism for capturing and tracking metrics and losses during the training process. The NNtrainer class allows users to easily specify the metrics they wish to monitor, such as accuracy, F1 score, or any custom evaluation metric. Additionally, users can choose to record the training and validation losses for further analysis.
- **Stochastic Weight Averaging (SWA) Support**: NNtrainer supports Stochastic Weight Averaging, a technique that can improve model generalization. The class includes functionalities for setting SWA models, starting epochs, and SWA-specific validation.
- **Model Checkpointing**: ConvinenceTorch includes model checkpointing functionality, allowing users to save the model's state at specified intervals during training or after every epoch. This feature facilitates easy resumption of training from the last saved checkpoint, enhancing training reliability and fault tolerance.
- **Inference Support** : With NNtrainer, users can make predictions using the trained model on new data, enabling them to leverage their models for inference tasks.

## Installation
This library is designed with a minimalistic approach to cater to educational applications and general research purposes. It boasts well-documented docstrings for every component, ensuring clarity and ease of understanding throughout its usage.

For those seeking to extend its functionalities, performing an editable install and making necessary tweaks to the library is highly recommended. Consequently, an editable install is the preferred method, allowing users to seamlessly customize the library to suit their specific requirements and leverage its full potential.
```bash
$ pip install convinence-torch
```

## Usage

```python
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

# Enable debugging 
import os
os.environ['LOG'] = '10'

# Import trainer 
from torchutils.trainer import NNtrainer

# Setup your data loaders
train_data = torchvision.datasets.FashionMNIST('./data/', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.FashionMNIST('./data/', train=False, transform=torchvision.transforms.ToTensor(), download=True)
trainloader = DataLoader(train_data, batch_size=B, shuffle=True, num_workers=5)
test_loader = DataLoader(test_data, batch_size=B)

# Constants
D = 28 * 28
n = 32
C = 1
classes = 10
eta_0 = 0.005
eta_1 = 0.001

# Define your PyTorch model, optimizer, and loss function
fc_model = nn.Sequential(
    nn.Conv2d(1, n, 3, padding=1),  # 28 * 28
    nn.LeakyReLU(0.1),
    nn.Conv2d(n, n, 3, padding=1),  # 28 * 28
    nn.LeakyReLU(0.1),
    nn.MaxPool2d(2),  # 14 * 14
    nn.Conv2d(n, n // 2, 3, padding=1),
    nn.LeakyReLU(0.1),
    nn.Conv2d(n // 2, n // 2, 3, padding=1),
    nn.LeakyReLU(0.1),
    nn.MaxPool2d(2),  # 7 * 7
    nn.Conv2d(n // 2, n // 4, 3, padding=1),
    nn.LeakyReLU(0.1),
    nn.Flatten(),
    nn.Linear(7 * 7 * n // 4, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 10)
)

# Create an instance of NNtrainer
trainer = NNtrainer(model, optimizer, loss)

# See the memory usage report
trainer.mem_report(batch_shape=(33, 1, 28, 28))

# Add Cosine Annealing
trainer.add_cosineannealing_lrs(lr_initial=eta_0, lr_final=eta_1, dips=5, epoch=20, verbose=True)

# Add Gradient Clipping
trainer.add_gradient_clipping(min=-100, max=100)

# Compile the model before training
trainer.compile()

# Train the model
trainer.train(trainloader=trainloader, valloader=valloader, epoch=20, metrics=['accuracy', 'f1'], record_loss=True, checkpoint_file='train')

# Plot the training and validation metric curves
trainer.plot_train_validation_metric_curve('accuracy')
```

## Contributing
Contributions are welcome! If you find any issues or have suggestions for new features, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.


