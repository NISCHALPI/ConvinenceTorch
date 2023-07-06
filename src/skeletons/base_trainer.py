import torch
from abc import ABC, abstractmethod
import typing as tp
from .base_log import get_logger


logger = get_logger('base_trainer')


__all__= ['BaseTrainer']

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
        
        # Set Device
        self._device : tp.Optional[torch.device]  = None
        super().__init__()

    def _move_to_device(self) -> None:
        
        if self._device is None:
            logger.debug(f'Checking if CUDA device is available ? : {torch.cuda.get_device_name(0) if  torch.cuda.is_available() else "Not Available"} ')
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


    def _weight_init(self, model: torch.nn.Module) -> None:
        """Initialize the weights of linear and convolutional layers using Xavier initialization.

        Args:
            model (torch.nn.Module): The PyTorch model for weight initialization.

        Returns:
            None

        Summary:
            This function initializes the weights of linear and convolutional layers in the provided PyTorch model
            using Xavier initialization. It applies the initialization to each relevant submodule in a recursive manner.

            Xavier initialization sets the weights according to a normal distribution with zero mean and variance
            calculated based on the number of input and output connections of the layer. The bias is initialized to zero
            using a uniform distribution.

            Note: This function modifies the weights of the model in-place.

        """
        if isinstance(model, torch.nn.Linear) or isinstance(model, torch.nn.Conv2d):
            
            logger.debug(f'Applying xavier normal weight init to Model: {model._get_name()}, ObjID: {id(model)}')
            torch.nn.init.xavier_normal_(model.weight.data)
            
            if model.bias is not None:
                torch.nn.init.zeros_(model.bias)
        return

