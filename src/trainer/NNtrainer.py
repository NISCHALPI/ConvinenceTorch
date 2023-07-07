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
          eval_every_epoch=None, record_loss=False, *args, **kwargs)
        Train the model using the provided data loader(s).
    validate(valloader, *args, **kwargs)
        Validate the model using the provided data loader.
    get_loss() -> tp.Union[tp.Tuple[list, list], list, None]
        Return the training and validation loss (if available).
    """
    available_metric = {'accuracy' : met.accuracy_score , 
            'f1' : met.f1_score , 'roc' : met.roc_auc_score , 
            'L1' : met.mean_absolute_error, 'precision':met.precision_score, 
            'recall' : met.recall_score}

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss: torch.nn.Module, seed : 
                 tp.Optional[float] = None, device : tp.Optional[torch.device] = None ,
                 lr_scheduler : tp.Optional[torch.optim.lr_scheduler.LRScheduler] = None) -> None: 
        
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
            epoch : int = 100, show_every_batch: tp.Optional[int] = None, restart: bool = False ,
            early_stopping : bool = False , eval_every_epoch : tp.Optional[int] = None 
            ,record_loss : bool = False , metrics : tp.Optional[tp.Union[tp.Iterable[str], str]] = None,  *args, **kwargs) -> None:
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
        record_loss : bool, optional
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
        - If `record_loss` is set to True, the training and validation loss can be saved for further analysis.
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
            if eval_every_epoch is not None and valloader is not None:
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
                if show_every_batch is not None:
                    if epoch % show_every_batch == 0:
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
            if eval_every_epoch is not None and valloader is not None and record_loss:
                if epoch % eval_every_epoch == 0:
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
        if hasattr(self, 'records'):
            return self.records
    
        return None 
    
    
    def plot_train_validation_metric_curve(self , metric : str = 'primary') -> None:
        """
        Plot the training and validation error curves.

        Returns
        -------
        None
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
        
        for metric in running_dict.keys():
            running_dict[metric].append(self._eval_metric(y_pred=y_pred, y_true=y_true, metric=metric))
        
        return

    def _collect_running_metric(self, running_dict : dict, what: bool = True):
        
        key = 'train' if what else 'validation'
        
        for metric in running_dict.keys():
            self.records[key][metric].append(sum(running_dict[metric])/len(running_dict[metric]))

        return

    def _eval_metric(self, y_pred : torch.Tensor , y_true : torch.Tensor, metric : str) -> float:
         

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
        if metric in self.available_metric.keys():
            return True
        
        Warning(f'{metric} is not available')
        
        return False

        

