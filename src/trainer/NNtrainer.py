import typing as tp
import torch
from ..skeletons import get_logger, BaseTrainer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
from tqdm import tqdm


logger = get_logger('NNtrainer')



__all__= ['NNtrainer']


class NNtrainer(BaseTrainer):
    """
    Neural Network Trainer class that extends the BaseTrainer.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    optimizer : torch.optim.Optimizer
        The optimizer for updating model parameters.
    loss : torch.nn.Module
        The loss function to compute the training loss.
    seed : float, optional
        Seed for random number generation. Default is None.
    lr_scheduler : torch.optim.lr_scheduler.LRScheduler, optional
        Learning rate scheduler. Default is None.

    Attributes
    ----------
    model : torch.nn.Module
        The model to be trained.
    optimizer : torch.optim.Optimizer
        The optimizer for updating model parameters.
    loss : torch.nn.Module
        The loss function to compute the training loss.
    seed : float, optional
        Seed for random number generation.
    best : torch.Tensor
        The best loss achieved during training.
    scheduler : torch.optim.lr_scheduler.LRScheduler, optional
        Learning rate scheduler.
    cycle : int
        Training cycle number.

    Methods
    -------
    train(trainloader, valloader=None, epoch=100, show_every_batch=None, early_stopping=False,
          eval_every_epoch=None, save_loss=False, *args, **kwargs)
        Train the model using the provided data loader(s).
    validate(valloader, *args, **kwargs)
        Validate the model using the provided data loader.
    get_loss() -> tp.Union[tp.Tuple[list, list], list, None]
        Return the training and validation loss (if available).
    """

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss: torch.nn.Module, seed : tp.Optional[float] = None, device : tp.Optional[torch.device] = None ,lr_scheduler : tp.Optional[torch.optim.lr_scheduler.LRScheduler] = None) -> None: 
        
        # Initlize
        super().__init__(model, optimizer, loss)
        
        #check optimizer link
        self._check_optimizer_model_link()
        # Get the seed 
        self.seed= seed 
        # set best loss
        self.best : float = torch.inf    
        # set lr scheduler 
        self.scheduler  = lr_scheduler
        self._check_optimizer_lr_link()

        # set device 
        if device is not None:
            self.device = device
        # Moves model to device : default is cuda
        self._move_to_device()

        # Training Cycles
        self.cycle : int = 1 

    def _check_optimizer_model_link(self) -> None:
        """
        Check if the optimizer is linked with the model parameters.
        """
        logger.debug(msg='Checking Optimizer-Model Link in NNtrainer')
        if not self.optimizer.param_groups[0]['params'] == list(self.model.parameters()):
            logger.error('Optimizer passed to NNtrainer is not linked with model parameters. optimizer.step() cannot work')
            raise RuntimeError


    def _check_optimizer_lr_link(self) -> None:    
        """
        Check if the optimizer and scheduler are properly linked.
        """
        if self.scheduler is not None:
            logger.debug(msg='Checking Optimizer-Scheduler Link in NNtrainer')
            if not self.scheduler.optimizer == self.optimizer:
                logger.error(msg='Scheduler not linked with the optimizer! Cannot perform lr.step()')
                raise RuntimeError
        else:
            logger.debug(msg="No lrscheduler is passed in the NNtrainer")



    def train(self, trainloader: DataLoader , valloader : tp.Optional[DataLoader] = None , 
            epoch : int = 100, show_every_batch: tp.Optional[int] = None, restart: bool = False ,
            early_stopping : bool = False , eval_every_epoch : tp.Optional[int] = None 
            ,save_loss : bool = False ,  *args, **kwargs) -> None:
        """
        Train the model using the provided data loader(s).

        Parameters
        ----------
        trainloader : DataLoader
            The data loader for the training data.
        valloader : Optional[DataLoader], optional
            The data loader for the validation data. Default is None.
        epoch : int, optional
            The number of training epochs. Default is 100.
        show_every_batch : Optional[int], optional
            Log the training loss every `show_every_batch` batches. Default is None.
        early_stopping : bool, optional
            Enable early stopping if the loss does not improve. Default is False.
        eval_every_epoch : Optional[int], optional
            Evaluate the model on the validation set every `eval_every_epoch` epochs. Default is None.
        save_loss : bool, optional
            Save the training and validation loss. Default is False.
        *args, **kwargs
            Additional arguments to be passed to the training loop.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the optimizer is not linked with the model parameters.

        Notes
        -----
        - The training loop iterates over the specified number of epochs.
        - During each epoch, the model is trained on the batches provided by the `trainloader`.
        - The loss is computed, gradients are backpropagated, and model parameters are updated using the optimizer.
        - Optionally, the training loss can be logged, and early stopping can be applied if specified.
        - If a validation data loader (`valloader`) is provided, the model can be evaluated on the validation set at specified intervals.
        - If `save_loss` is set to True, the training and validation loss can be saved for further analysis.
        """

        # Set model to training mode
        self.model.train()
        
        # Restart Training 
        if restart:
            logger.debug('Restart flag passed to train! reinitilizing weights using xavier normal and bias to zero')
            self.cycle = 1
            self.model.apply(self._weight_init)


        logger.info(f'--------------START OF  {self.cycle} TRAINING CYCLE---------------------')
        
        # Start Best Loss
        start_loss = self.best

        # if passed seed
        if self.seed:
            torch.manual_seed(self.seed)
        
        # Loss saving attribiutes
        if save_loss:
            setattr(self, 'epoch_loss', None)
            self.epoch_loss = []
            if eval_every_epoch is not None and valloader is not None:
                setattr(self, 'epoch_val_loss', None)
                self.epoch_val_loss = []

        

        
        for epoch in tqdm(range(1, epoch + 1), desc='Epoch', colour='blue', ncols=80 , position=0):        
            # Set running loss to zero
            running_loss = 0

            # Trigger LR Scheduler 
            if self.scheduler is not None:
                self.scheduler.step()

            for idx, (feature, lable)  in enumerate(trainloader):
                # Move to device
                feature = feature.to(self.device)
                lable = lable.to(self.device)
                fp = self.model(feature)
                loss = self.loss(lable , fp)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Log the batch log 
                if show_every_batch is not None:
                    if epoch % show_every_batch == 0:
                        logger.info(f'Epoch {epoch}, Batch: {idx}, Loss: {loss.data.item():.3f}...')
                

                running_loss += loss.data.item()
            
            # Record best loss
            if self.best > running_loss:
                    self.best = running_loss

            
            logger.info(f'Finished epoch {epoch}. Loss: {running_loss:.3f}...')

            # Append Loss
            if hasattr(self , 'epoch_loss'):
                self.epoch_loss.append(running_loss)
            
            # Evaluate on valid set 
            if eval_every_epoch is not None and valloader is not None and save_loss:
                if epoch % eval_every_epoch == 0:
                    self.epoch_val_loss.append(self.validate(valloader))


            # Use Early Stopping if provided 
            if early_stopping:
                if  self.best > start_loss:
                    logger.info(f'Training stopped early due to no improvement in loss.')
                    break
            
            

        logger.info(f'--------------END OF  {self.cycle} TRAINING CYCLE---------------------')
        
        # Increment Cycle
        self.cycle += 1

        return



    def validate(self ,valloader : DataLoader ,  *args, **kwargs) -> float:
        """
        Validate the model using the provided data loader.

        Parameters
        ----------
        valloader : Optional[DataLoader]
            The data loader for the validation data. Default is None.
        *args, **kwargs
            Additional arguments to be passed for validation.

        Returns
        -------
        float
            The validation loss.

        Notes
        -----
        - The model is switched to evaluation mode during the validation process.
        - The validation loss is computed for each batch in the validation data.
        - The total validation loss is returned.
        """
        # Set to eval
        self.model.eval()

        with torch.no_grad(): 

            loss = 0. 
            for feature, lable in valloader:
                feature = feature.to(self.device)
                lable = lable.to(self.device)
                fp = self.model(feature)
                loss += self.loss(lable, fp)
        
        #Set to train 
        self.model.train()

        return float(loss)


    def get_loss(self) -> tp.Union[tp.Tuple[list , list], list, None]:
        """
        Get the training and validation loss (if available).

        Returns
        -------
        Union[Tuple[List, List], List, None]
            - If both training and validation loss are available:
                - A tuple containing two lists: the training loss and the validation loss.
            - If only training loss is available:
                - A list containing the training loss.
            - If only validation loss is available:
                - A list containing the validation loss.
            - If neither training nor validation loss is available:
                - None.
        """
        if hasattr(self, 'epoch_loss') and hasattr(self, 'epoch_val_loss'):
            return self.epoch_loss, self.epoch_val_loss
        elif hasattr(self, 'epoch_loss'):
            return self.epoch_loss
        elif hasattr(self, 'epoch_val_loss'):
            return self.epoch_val_loss
        
        return None
    
    
    def plot_train_validation_error_curve(self) -> None:
        """
        Plot the training and validation error curves.

        Returns
        -------
        None
        """
        plt.figure(figsize=(10, 8))
        plt.grid(visible=True, which='both', axis='both')

        if hasattr(self, 'epoch_loss'):    
            plt.plot(range(1, len(self.epoch_loss) + 1), self.epoch_loss, color='red', linestyle='-', marker='o', markersize=5, label='Training Loss / Epoch', alpha=0.5)

        if hasattr(self, 'epoch_val_loss'):
            plt.plot(range(1, len(self.epoch_val_loss) + 1), self.epoch_val_loss, color='purple', alpha=0.5,linestyle='-', marker='s', markersize=5, label='Validation Loss / Epoch')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Error Curves')
        plt.legend()
        
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(linestyle='dotted', linewidth=0.5)
        plt.tight_layout()
        
        plt.show()


    def predict(self, X: torch.tensor) -> torch.tensor:
        """
        Predicts the output for the input tensor using the trained model.

        Args:
            X (torch.tensor): The input tensor for prediction.

        Returns:
            torch.tensor: The predicted output tensor.

        Note:
            The model needs to be in evaluation mode (self.model.eval()) before making predictions.
            The input tensor X should be moved to the appropriate device (self.device) before prediction.

        Example:
            >>> input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> model = MyModel()
            >>> model.load_state_dict(torch.load("model_weights.pth"))
            >>> output = model.predict(input_tensor)
        """

        # Set to evaluation 
        self.model.eval()

        with torch.no_grad():
            X = X.to(self.device)
            out = self.model(X).cpu()
        
        # Reset to train mode 
        self.model.train() 

        return out
