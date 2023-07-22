"""This script defines a BaseTrainer class, which is the base class for trainers in PyTorch.

BaseTrainer:
============
    Base class for trainers in PyTorch.
    
Parameters:
-----------
    model (torch.nn.Module): The model to be trained.
    optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
    loss (torch.nn.Module): The loss function to compute the training loss.
    

Attributes:
-----------
    model (torch.nn.Module): The model to be trained.
    optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
    loss (torch.nn.Module): The loss function to compute the training loss.
    _device (torch.device): The device to be used for training. Defaults to GPU:0 if available, otherwise CPU.
    
Properties:
-----------
    device (torch.device): Get or set the device to be used for training.
    

Methods:
--------
    train(*args, **kwargs): Abstract method for training the model.
    validate(*args, **kwargs): Abstract method for validating the model.
    save_model(): Abstract method for saving the trained model.
    
Note: For detailed documentation of the class methods, attributes, and properties, please refer to the docstrings within the code.
"""

import typing as tp
from abc import ABC, abstractmethod

import torch

from .base_log import get_logger

logger = get_logger("base_trainer")


__all__ = ["BaseTrainer"]


class BaseTrainer(ABC):
    """Base class for trainers in PyTorch.

    Args:
    -----
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        loss (torch.nn.Module): The loss function to compute the training loss.

    Attributes:
    -----------
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        loss (torch.nn.Module): The loss function to compute the training loss.
        _device (torch.device): The device to be used for training. Defaults to GPU:0 if available, otherwise CPU.

    Properties:
    -----------
        device (torch.device): Get or set the device to be used for training.

    Methods:
    --------
        train(*args, **kwargs): Abstract method for training the model.
        validate(*args, **kwargs): Abstract method for validating the model.
        save_model(): Abstract method for saving the trained model.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: torch.nn.modules.loss._Loss,
    ) -> None:
        """Constructer for the base trainer.

        Args:
            model (torch.nn.Module): The model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
            loss (torch.nn.Module): The loss function to compute the training loss.
        """
        # Optimizer
        self._model = model
        # OPTIMIZER
        self._optimizer = optimizer
        # Loss
        self.loss = loss

        # Set Device
        self._device: tp.Optional[torch.device] = None
        super().__init__()

    def _move_to_device(self) -> None:
        if self._device is None:
            logger.debug(
                f'Checking if CUDA device is available ? : {torch.cuda.get_device_name(0) if  torch.cuda.is_available() else "Not Available"} '
            )
            self.device = (
                torch.device("cuda:0")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        logger.debug(msg="Moving model to device and linking it to optimizer")
        self._model = self._model.to(device=self.device)

    @property
    def model(self) -> torch.nn.Module:
        """Model property of the base trainer.

        Returns:
            torch.nn.Module: model of the trainer.
        """
        return self._model

    @model.setter
    def model(self) -> tp.NoReturn:
        logger.debug(
            "Cannot assign new model! Instanitate new trainer explicitly for new model"
        )
        raise RuntimeError(
            "Cannot assign new model! Instanitate new trainer explicitly for new model"
        )

    @property
    def device(self) -> tp.Optional[torch.device]:
        """Get or set the device to be used for training.

        Returns:
        --------
            torch.device: The current device.

        """
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        """Set the device to be used for training.

        Args:
            device (torch.device): The device to be set.

        Raises:
            RuntimeError: If the device is not a subclass of torch.device.
        """
        if isinstance(device, torch.device):
            self._device = device
        else:
            logger.error("Device must be a subclass of torch.device")
            raise RuntimeError

    @abstractmethod
    def train(self) -> None:  # type: ignore
        """Abstract train method."""
        pass

    @abstractmethod
    def _validate(self) -> float:  # type: ignore
        """Abstract method to validate.

        Returns:
            float: loss of the model wrt validation set or batch.
        """
        pass

    @staticmethod
    def _weight_init(model: torch.nn.Module) -> None:
        """Initialize the weights of linear and convolutional layers using Xavier initialization.

        Args:
            model (torch.nn.Module): The PyTorch model for weight initialization.

        Returns:
            None

        Extended Summary:
            This function initializes the weights of linear and convolutional layers in the provided PyTorch model
            using Xavier initialization. It applies the initialization to each relevant submodule in a recursive manner.

            Xavier initialization sets the weights according to a normal distribution with zero mean and variance
            calculated based on the number of input and output connections of the layer. The bias is initialized to zero
            using a uniform distribution.

            Note: This function modifies the weights of the model in-place.

        """
        if isinstance(model, (torch.nn.Conv2d, torch.nn.Linear)):
            logger.debug(
                f"Applying xavier normal weight init to Model: {model._get_name()}, ObjID: {id(model)}"
            )
            torch.nn.init.xavier_normal_(model.weight.data)

            if model.bias is not None:
                torch.nn.init.zeros_(model.bias)
