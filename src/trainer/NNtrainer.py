import typing as tp
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
from tqdm import tqdm
from ..skeletons import get_logger, BaseTrainer
from ..skeletons import Register

logger = get_logger('NNtrainer')



__all__= ['NNtrainer']


class NNtrainer(BaseTrainer):
    """
    A trainer class for neural networks.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model.
    optimizer : torch.optim.Optimizer
        The optimizer for model parameters.
    loss : torch.nn.Module
        The loss function.
    seed : float, optional
        The seed value. Defaults to None.
    device : torch.device, optional
        The device to run the model on. Defaults to None.
    lr_scheduler : torch.optim.lr_scheduler.LRScheduler, optional
        The learning rate scheduler. Defaults to None.

    Attributes
    ----------
    available_metric : dict
        A dictionary of available evaluation metrics.
    best : float
        The best loss achieved during training.
    scheduler : torch.optim.lr_scheduler.LRScheduler
        The learning rate scheduler.
    device : torch.device
        The device to run the model on.
    cycle : int
        The training cycle number.

    Methods
    -------
    train(trainloader, valloader=None, epoch=100, log_every_batch=None, restart=False, early_stopping=False,
          validate_every_epoch=None, record_loss=True, metrics=None, *args, **kwargs)
        Trains the model.
    validate(valloader, metrics=None, *args, **kwargs)
        Validates the model on a validation set.
    get_loss()
        Returns the training loss.
    plot_train_validation_metric_curve(metric='primary')
        Plots the training and validation metric curves.
    predict(X)
        Makes predictions using the trained model.

    Notes
    -----
    This trainer assumes a classification task with multiple classes.
    """
    


    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss: torch.nn.Module, seed : 
                 tp.Optional[float] = None, device : tp.Optional[torch.device] = None ,
                 lr_scheduler : tp.Optional[torch.optim.lr_scheduler.LRScheduler] = None) -> None: 
        """
        Initializes the NNtrainer.

        Parameters
        ----------
        model : torch.nn.Module
            The neural network model.
        optimizer : torch.optim.Optimizer
            The optimizer for model parameters.
        loss : torch.nn.Module
            The loss function.
        seed : float, optional
            The seed value. Defaults to None.
        device : torch.device, optional
            The device to run the model on. Defaults to None.
        lr_scheduler : torch.optim.lr_scheduler.LRScheduler, optional
            The learning rate scheduler. Defaults to None.
        """


        # Initlize
        super().__init__(model, optimizer, loss)
        
        #check optimizer link
        self._check_optimizer_model_link()
        # Get the seed 
        self.seed= seed 
        
        # set lr scheduler 
        self.scheduler  = lr_scheduler
        self._check_optimizer_lr_link()

        # set device 
        if device is not None:
            self.device = device
        # Moves model to device : default is cuda
        self._move_to_device()

        
        # Training Cycles
        self.cycle : int = 0 


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
            epoch : int = 100, log_every_batch: tp.Optional[int] = None, restart: bool = False ,
            validate_every_epoch : int = 1 ,record_loss : bool = True ,
            metrics : tp.Optional[tp.Union[tp.Iterable[str], str]] = None,  *args, **kwargs) -> None:
        
        """
        Trains the model.

        Parameters
        ----------
        trainloader : DataLoader
            The data loader for the training set.
        valloader : DataLoader, optional
            The data loader for the validation set. Defaults to None.
        epoch : int, optional
            The number of epochs to train. Defaults to 100.
        log_every_batch : int, optional
            Log the training loss every specified number of batches. Defaults to None.
        restart : bool, optional
            Restart training from the beginning. Defaults to False.
        early_stopping : bool, optional
            Stop training early if no improvement in loss. Defaults to False.
        validate_every_epoch : int, optional
            Perform validation every specified number of epochs. Defaults to None.
        record_loss : bool, optional
            Record the training and validation loss. Defaults to True.
        metrics : Union[Iterable[str], str], optional
            Evaluation metrics to calculate. Defaults to None.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.
        """


        
        # Set model to training mode
        self.model.train()
        logger.debug(f'Setting model to train for OBID = {id(self)}')



        #Initilize Weights using Xaviers Uniform Weight init 
        if self.cycle == 0:
            self._weight_init(self.model)




        # Restart Training 
        if restart:
            logger.debug('Restart flag passed to train! reinitilizing weights using xavier normal and bias to zero')
            self.cycle = 0
            self.model.apply(self._weight_init)



        # If instantiate the register if record loss at the start
        if record_loss:
            setattr(self, 'register', Register(metrics=metrics, loss=self.loss, epoch=epoch ,cycle= self.cycle + 1))

        # If not, remove previous if present
        else:
            if hasattr(self, 'register'):
                delattr(self, 'register')


        # Start the training 
        logger.info(f'--------------START OF  {self.cycle + 1} TRAINING CYCLE---------------------')
       
        # if passed seed
        if self.seed:
            torch.manual_seed(self.seed)
              

        for e in tqdm(range(1, epoch + 1), desc=f'Train Cycle: {self.cycle + 1} , Epoch', colour='blue', ncols=80 , position=0):        
            
            # Trigger LR Scheduler 
            if self.scheduler is not None:
                self.scheduler.step()

            for idx, (feature, lable)  in enumerate(trainloader):
                # Move to device
                feature = feature.to(self.device)
                lable = lable.to(self.device)
                fp = self.model(feature)
                loss = self.loss(fp, lable)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Log the batch log 
                if log_every_batch is not None:
                    if e % log_every_batch == 0:
                        logger.info(f'Epoch {e}, Batch: {idx}, Loss: {loss.data.item():.3f}...')

                # Register the metrics 
                if record_loss:
                    self.register : Register
                    self.register._record(y_pred=fp, y_true=lable, epoch=e, where=True)
            

            # Evaluate on validation set 
            if validate_every_epoch != 0 and valloader is not None:
                if record_loss:
                    if e % validate_every_epoch == 0:
                        self._validate(valloader=valloader, epoch=e)
                else:
                    raise ValueError('Validation Data Loader Passed but record_loss is False. No point in validataion. Pass record_loss = True explicitly')
        
        # Minimize the Register 
        if record_loss:
            self.register._minimize_per_epoch()


        logger.info(f'--------------END OF  {self.cycle} TRAINING CYCLE---------------------')
        
        # Increment Cycle
        self.cycle += 1

        return


    def _validate(self ,valloader : DataLoader , epoch : tp.Optional[int] = None  , *args, **kwargs) -> float:
        """
        Validates the model on a validation set.

        Parameters
        ----------
        valloader : DataLoader
            The data loader for the validation set.
        metrics : Union[Iterable[str], str], optional
            Evaluation metrics to calculate. Defaults to None.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        float
            The validation loss.
        """
        # Set to eval
        self.model.eval()
        logger.debug(f'Setting model to eval for OBID = {id(self)}')
        
        with torch.no_grad(): 
            loss = 0
            for feature, lable in valloader:
                feature = feature.to(self.device)
                lable = lable.to(self.device)
                fp = self.model(feature)
                loss += self.loss(fp, lable)
            
                if hasattr(self, 'register') and epoch is not None:
                    self.register._record(y_pred=fp, y_true=lable, epoch=epoch, where=False)                
    

        #Set to train 
        self.model.train()
        logger.debug(f'Setting model to train for OBID = {id(self)}')
        return loss


    def get_loss(self) -> tp.Union[dict , None]:
        """
        Returns the training loss.

        Returns
        -------
        Union[dict, None]
            The training loss as a dictionary if available, None otherwise.
        """
        if hasattr(self, 'register'):
            return self.register.records
    
        return None 
    
    
    def plot_train_validation_metric_curve(self , metric : tp.Optional[str] = None) -> None:
        """
        Plots the training and validation metric curves.

        Parameters
        ----------
        metric : str, optional
            The metric to plot. Defaults to loss.
        """
        if hasattr(self, 'register'):
            self.register.plot_train_validation_metric_curve(metric=metric)

        else:
            raise RuntimeError('Record Loss was not passed to train method. Metrics not Recorded!')


    def predict(self, X: torch.tensor) -> torch.tensor:
        """
        Makes predictions using the trained model.

        Parameters
        ----------
        X : torch.tensor
            The input tensor for making predictions.

        Returns
        -------
        torch.tensor
            The predicted tensor.
        """

        # Set to evaluation
        logger.debug(f'Setting model to eval for OBID = {id(self)}')
        self.model.eval()

        with torch.no_grad():
            X = X.to(self.device)
            out = self.model(X).cpu()
        
        # Reset to train mode
        logger.debug(f'Setting model to train for OBID = {id(self)}')
        self.model.train() 

        return out

