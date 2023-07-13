ConvinenceTorch is a Python package that provides utility modules and classes for PyTorch, a popular deep learning framework. It offers a collection of tools to simplify and enhance the training process of neural networks in PyTorch.


## Features

- **NNtrainer**: A flexible and extensible trainer class for neural networks in PyTorch. It handles common training tasks such as model initialization, optimization, loss computation, device management, and training/validation loops. The NNtrainer class allows developers to focus on defining models and customizing the training process without worrying about boilerplate code.

## Installation

You can install ConvinenceTorch using `pip`:

```shell
pip install convinencetorch
```

## Usage 

```python
import torch
from torch.utils.data import DataLoader
from torchutils.trainer import NNtrainer

# Define your PyTorch model, optimizer, and loss function
model = ...
optimizer = ...
loss = ...

# Create a DataLoader for your training data
train_loader = DataLoader(...)

# Create an instance of NNtrainer
trainer = NNtrainer(model, optimizer, loss)

trainer.train(trainloader=trainloader, valloader=valloader, epoch=20,  metrics=['accuracy', 'f1'], record_loss=True, checkpoint_file='train'  , checkpoint_every_x=2)
```

## Contributing
Contributions are welcome! If you find any issues or have suggestions for new features, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.


