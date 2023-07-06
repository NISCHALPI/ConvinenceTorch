import torch
from abc import ABC, abstractmethod
import typing as tp
from .base_log import get_logger


logger = get_logger('base_trainer')


__all_= ['BaseTrainer']

class BaseTrainer(ABC):
    """Base class for trainers in PyTorch.

    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        loss (torch.nn.Module): The loss function to compute the training loss.

    Attributes:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        loss (torch.nn.Module): The loss function to compute the training loss.
        _device (torch.device): The device to be used for training. Defaults to GPU:0 if available, otherwise CPU.

    Properties:
        device (torch.device): Get or set the device to be used for training.

    Methods:
        train(*args, **kwargs): Abstract method for training the model.
        validate(*args, **kwargs): Abstract method for validating the model.
        save_model(): Abstract method for saving the trained model.

    """

    def __init__(self, model :torch.nn.Module, optimizer  : torch.optim.Optimizer, loss : torch.nn.Module) -> None: 
        
        
        # Optimizer
        self.model = model
        #OPTIMIZER
        self.optimizer =  optimizer
        #Loss 
        self.loss = loss
        
        # PREFER GPU:0 IF AVAILABLE
        self._device : tp.Optional[torch.device]  = None

        
        
        super().__init__()

    def _move_to_device(self) -> None:
        
        if self._device is None:
            self.device =  torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu') 
        
        logger.debug(msg='Moving model to device and linking it to optimizer')
        self.model = self.model.to(device=self.device)
        return
        
        

    @property
    def device(self) -> torch.device:
        """Get or set the device to be used for training.

        Returns:
            torch.device: The current device.

        """
        return self._device
    
    @device.setter
    def device(self, device : torch.device) -> None:
        """Set the device to be used for training.

        Args:
            device (torch.device): The device to be set.

        Raises:
            RuntimeError: If the device is not a subclass of torch.device.

        """
        if isinstance(device, torch.device):
            self._device = device
        else:
            logger.error('Device must be a subclass of torch.device')
            raise RuntimeError

    @abstractmethod
    def train(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def validate(self, *args , **kwargs):
        pass    

    
