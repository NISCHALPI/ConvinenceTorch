import typing as tp
import torch
from ..skeletons import get_logger, BaseTrainer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
from tqdm import tqdm
import sklearn.metrics as met
import numpy as np

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
    
    # Metrics 
    available_metric = {
        'accuracy' : met.accuracy_score , 
        'f1' : met.f1_score , 
        'roc' : met.roc_auc_score , 
        'L1' : met.mean_absolute_error,
        'precision':met.precision_score, 
        'recall' : met.recall_score
        }



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
            early_stopping : bool = False , validate_every_epoch : tp.Optional[int] = None 
            ,record_loss : bool = True , metrics : tp.Optional[tp.Union[tp.Iterable[str], str]] = None,  *args, **kwargs) -> None:
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


        logger.info(f'--------------START OF  {self.cycle} TRAINING CYCLE---------------------')
        
        # Start Best Loss
        start_loss = self.best

        # if passed seed
        if self.seed:
            torch.manual_seed(self.seed)
        
        # Loss saving attribiutes setup
        if record_loss and self.cycle == 0:
            logger.debug(f'Recording primary traning loss for NNTrainer : OBJID {id(self)}')
            setattr(self, 'records', {})
            # Set primary train loss 
            self.records['train'] = {}
            self.records['train']['primary'] = []

           
            if metrics is not None: 
                if isinstance(metrics , str):
                    metrics = [metrics]

                # Filter metric 
                metrics = list(filter(self._check_metric, metrics))
                
                for metric in metrics:
                    self.records['train'][metric] = []
            
            # Set primary validation loss
            if validate_every_epoch is not None and valloader is not None:
                logger.debug(f'Recording primary validation loss for NNTrainer : OBJID {id(self)}')
                self.records['validation'] = {}
                self.records['validation']['primary'] = []

                if metrics is not None:
                    for metric in metrics:
                            self.records['validation'][metric] = []
            
        
        
        for epoch in tqdm(range(1, epoch + 1), desc='Epoch', colour='blue', ncols=80 , position=0):        
            
            # Set running loss to zero
            running_loss = 0
            

            # running additional metric
            additional_running_loss = {}
            
            if metrics is not None and record_loss:
                additional_running_loss = {name : [] for name in metrics}
            

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
                

                #Record training additional metrics 
                if record_loss and metrics is not None:
                    self._record_running_metric(fp, lable, additional_running_loss)
                
                # Log the batch log 
                if log_every_batch is not None:
                    if epoch % log_every_batch == 0:
                        logger.info(f'Epoch {epoch}, Batch: {idx}, Loss: {loss.data.item():.3f}...')
                

                running_loss += loss.data.item()
            

            # Record best loss
            if  running_loss < self.best:
                    self.best = running_loss
            
            # Collect Running loss for training
            if record_loss and metrics is not None:
                self._collect_running_metric(additional_running_loss)
            
            # Log 
            logger.info(f'Finished epoch {epoch}. Loss: {running_loss:.3f}...')


            
            # Append  Primary Loss
            if hasattr(self , 'records'):
                self.records['train']['primary'].append(running_loss)
            

            # Evaluate on valid set 
            if validate_every_epoch is not None and valloader is not None and record_loss:
                if epoch % validate_every_epoch == 0:
                    self.records['validation']['primary'].append(self.validate(valloader, metrics))


            # Use Early Stopping if provided 
            if early_stopping:
                if  self.best > start_loss:
                    logger.info(f'Training stopped early due to no improvement in loss.')
                    break
            
            

        logger.info(f'--------------END OF  {self.cycle} TRAINING CYCLE---------------------')
        
        # Increment Cycle
        self.cycle += 1

        return



    def validate(self ,valloader : DataLoader , metrics : list = None , *args, **kwargs) -> float:
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
            loss = 0.

            # Metric Definations
            running_valid_metric = {}
            if metrics is not None:
                running_valid_metric = {name : [] for name in metrics}


            for feature, lable in valloader:
                feature = feature.to(self.device)
                lable = lable.to(self.device)
                fp = self.model(feature)
                loss += self.loss(fp, lable)
                
                # recored running metric
                self._record_running_metric(fp, lable, running_dict=running_valid_metric)

            
            # Collect avereges 
            self._collect_running_metric(running_valid_metric, what=False)
                
                
        #Set to train 
        self.model.train()
        logger.debug(f'Setting model to train for OBID = {id(self)}')

        return float(loss)


    def get_loss(self) -> tp.Union[dict , None]:
        """
        Returns the training loss.

        Returns
        -------
        Union[dict, None]
            The training loss as a dictionary if available, None otherwise.
        """
        if hasattr(self, 'records'):
            return self.records
    
        return None 
    
    
    def plot_train_validation_metric_curve(self , metric : str = 'primary') -> None:
        """
        Plots the training and validation metric curves.

        Parameters
        ----------
        metric : str, optional
            The metric to plot. Defaults to 'primary'.
        """
        plt.figure(figsize=(10, 8))
        plt.grid(visible=True, which='both', axis='both')

        if hasattr(self, 'records') and metric in self.records['train'].keys():
                
            plt.plot(range(1, len(self.records['train'][metric]) + 1), self.records['train'][metric], color='red', linestyle='-', marker='o', markersize=5, label='Training Loss / Epoch', alpha=0.5)

            if 'validation' in self.records.keys() and metric in self.records['validation'].keys():
                plt.plot(range(1, len(self.records['validation'][metric]) + 1), self.records['validation'][metric], color='purple', alpha=0.5,linestyle='-', marker='s', markersize=5, label='Validation Loss / Epoch')

        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'Training and Validation {metric} Curves')
        plt.legend()
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(linestyle='dotted', linewidth=0.5)
        plt.tight_layout()
        
        plt.show()


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


    def _record_running_metric(self,  y_pred : torch.Tensor , y_true : torch.Tensor, running_dict: dict) -> None:
        """
        Records the running metric during training.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted tensor.
        y_true : torch.Tensor
            The true tensor.
        running_dict : dict
            The dictionary to store the running metric values.
        """
        for metric in running_dict.keys():
            running_dict[metric].append(self._eval_metric(y_pred=y_pred, y_true=y_true, metric=metric))
        
        return

    def _collect_running_metric(self, running_dict : dict, what: bool = True):
        """
        Collects the running metric during training.

        Parameters
        ----------
        running_dict : dict
            The dictionary containing the running metric values.
        what : bool, optional
            Specifies whether to collect the metric for training or validation. Defaults to True.
        """
        key = 'train' if what else 'validation'
        
        for metric in running_dict.keys():
            self.records[key][metric].append(sum(running_dict[metric])/len(running_dict[metric]))

        return

    def _eval_metric(self, y_pred : torch.Tensor , y_true : torch.Tensor, metric : str) -> float:
        """
        Evaluates the metric between predicted and true tensors.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted tensor.
        y_true : torch.Tensor
            The true tensor.
        metric : str
            The metric to evaluate.

        Returns
        -------
        float
            The evaluated metric value.
        """

        # Move to CPU and Record
        y_pred_cpu = y_pred.data.cpu()
        y_true_cpu = y_true.data.cpu()

        # Apply softmax if logits 
        if isinstance(self.loss , (torch.nn.CrossEntropyLoss , )):
            y_pred_cpu = torch.nn.functional.softmax(y_pred_cpu, dim=1).argmax(dim=1)
        
        y_pred_cpu = y_pred_cpu.numpy().ravel()
        y_true_cpu   = y_true_cpu.numpy().ravel()


        return self.available_metric[metric](y_true_cpu, y_pred_cpu)
    
    def _check_metric(self, metric : str) -> bool:
        """
        Checks if the given metric is available.

        Parameters
        ----------
        metric : str
            The metric to check.

        Returns
        -------
        bool
            True if the metric is available, False otherwise.
        """
        
        if metric in self.available_metric.keys():
            return True
        
        Warning(f'{metric} is not available')
        
        return False

        

