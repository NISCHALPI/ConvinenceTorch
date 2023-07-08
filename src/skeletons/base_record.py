from statistics import mean
import sklearn.metrics as met 
import  torch 
import typing as tp
import matplotlib.pyplot as plt




__all__ = ['Register']



class Register(object):
    """
    Class for recording and evaluating metrics during training.

    Parameters
    ----------
    metrics : Union[List[str], str, None]
        The metrics to record and evaluate. If None, only the loss metric is used.
        If a string, it represents a single metric. If a list of strings, it represents multiple metrics.
    loss : torch.nn.modules.loss._Loss
        The loss function used in training.
    epoch : int
        The total number of training epochs.

    Attributes
    ----------
    available_metric : dict
        Dictionary mapping metric names to metric functions.

    records : dict
        Dictionary storing the recorded metrics for each epoch and dataset split.

    minimized_record : dict
        Dictionary storing the mean value of recorded metrics per epoch and dataset split.

    Methods
    -------
    _init_metrics(metrics: Union[List[str], str]) -> None
        Initialize the metrics to be recorded and evaluated.

    _key(epoch: int) -> str
        Returns the key string for a given epoch.

    _init_records() -> None
        Initialize the records dictionary.

    _check_metric(metric_name: str) -> bool
        Check if the given metric is available.

    _eval_sk_metric(y_pred: torch.Tensor, y_true: torch.Tensor, metric_name: str) -> float
        Evaluate the metric between predicted and true tensors.

    _record(y_pred: torch.Tensor, y_true: torch.Tensor, epoch: int, where: bool = True) -> None
        Record the metrics for a given epoch and dataset split.

    _minimize_per_epoch() -> None
        Calculate the mean value of recorded metrics per epoch and dataset split.

    Properties
    ----------
    records : dict
        Get the records dictionary.

    minimized_record : dict
        Get the minimized_record dictionary.
    """


    # Available metrics metric System
    available_metric = {
        'accuracy' : met.accuracy_score , 
        'f1' : met.f1_score , 
        'roc' : met.roc_auc_score , 
        'L1' : met.mean_absolute_error,
        'precision':met.precision_score, 
        'recall' : met.recall_score
        }



    def __init__(self, metrics : tp.Union[tp.List[str], str, None] , loss: torch.nn.modules.loss._Loss , epoch : int, cycle : tp.Optional[int] = None) -> None:
        """
        Initialize a Register object.

        Parameters
        ----------
        metrics : Union[List[str], str, None]
            The metrics to record and evaluate. If None, only the loss metric is used.
            If a string, it represents a single metric. If a list of strings, it represents multiple metrics.
        loss : torch.nn.modules.loss._Loss
            The loss function used in training.
        epoch : int
            The total number of training epochs.
        """
        
        self._metrics = None
        self._loss = loss
        self.available_metric[loss._get_name()]= loss
        self._records  = {}
        self._epoch = epoch
        self._init_metrics(metrics)
        self._init_records()
        self.cycle = cycle

        return
    
    def _init_metrics(self, metrics: tp.Union[tp.List[str], str]) -> None:
        """
        Initialize the metrics to be recorded and evaluated.

        Parameters
        ----------
        metrics : Union[List[str], str]
            The metrics to record and evaluate.
        """

        # Init metric 
        if metrics is None:
            self._metrics = [self._loss._get_name()]
        elif isinstance(metrics , str):
           self._metrics = [metrics, self._loss._get_name()]
        elif isinstance(metrics , list):
            self._metrics = [*metrics, self._loss._get_name()]
        else:
            raise ValueError('Metric type should be one of  List[str] | str | None ')

        # Check New Metrics
        for name in self._metrics:
            self._check_metric(name)

        return None
    
    def _key(self, epoch : int) -> str:
        """
        Returns the key string for a given epoch.

        Parameters
        ----------
        epoch : int
            The epoch number.

        Returns
        -------
        str
            The key string.
        """

        return f'Epoch_{epoch}'


    def _init_records(self) -> None:
        """
        Initialize the records dictionary.
        """

        # initalize the records 
        for index in ['train', 'valid']:
            self._records[index] = {}
            for name in self._metrics:
                self._records[index][name] = {} 
                for e in range(1, self._epoch + 1):
                    self._records[index][name][self._key(e)] = []
        

        
        return None

    def _check_metric(self, metric_name : str) -> bool:
        """
        Checks if the given metric is available.

        Parameters
        ----------
        metric_name : str
            The metric to check.

        Returns
        -------
        bool
            True if the metric is available, False otherwise.
        """

        if metric_name in self.available_metric.keys() or metric_name == self._loss._get_name():
            return True
        else:
            raise RuntimeError(f'{metric_name} is not available!')
        
    def _eval_metric(self, y_pred: torch.Tensor , y_true: torch.Tensor, metric_name: str) -> float:
        """
        Evaluates the metric between predicted and true tensors.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted tensor.
        y_true : torch.Tensor
            The true tensor.
        metric_name : str
            The metric to evaluate.

        Returns
        -------
        float
            The evaluated metric value.
        """
        
        # move to cpu 
        y_pred_cpu = y_pred.data.cpu()
        y_true_cpu = y_true.data.cpu()

        

         # Check if metric is from sklearn or loss func from pytorch
        if issubclass(self.available_metric[metric_name].__class__ , torch.nn.Module):
            return float(self.available_metric[metric_name](y_pred_cpu, y_true_cpu))
        
        else:
            # Apply softmax if logits from cross entropy
            if isinstance(self._loss , (torch.nn.CrossEntropyLoss , )):
                y_pred_cpu = torch.nn.functional.softmax(y_pred_cpu, dim=1).argmax(dim=1)

            y_pred_cpu = y_pred_cpu.numpy().ravel()
            y_true_cpu = y_true_cpu.numpy().ravel()
            return float(self.available_metric[metric_name](y_true_cpu, y_pred_cpu))


    def _record(self, y_pred: torch.Tensor, y_true: torch.Tensor, epoch: int,  where : bool = True) -> None:
        """
        Record the metrics for a given epoch and dataset split.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted tensor.
        y_true : torch.Tensor
            The true tensor.
        epoch : int
            The epoch number.
        where : bool, optional
            Indicates whether the record is for the training set (True) or validation set (False).
            Defaults to True.
        """

        key = 'train' if where else 'valid'
        
        for name in self._metrics:    
                val =  self._eval_metric(y_pred=y_pred, y_true=y_true, metric_name=name)
                self._records[key][name][self._key(epoch)].append(val) 

        return None

    def _minimize_per_epoch(self) -> None:
        """
        Calculate the mean value of recorded metrics per epoch and dataset split.
        """
        
        # Set minimized attribute 
        setattr(self, '_records_per_epoch', {})
        # Minimize to Mean
        for index in ['train', 'valid']:
            self._records_per_epoch[index] = {}
            for name in self._metrics:
                self._records_per_epoch[index][name] = [] 
                for e in range(1, self._epoch + 1):
                    # If not empty, calculate mean and append 
                    if len(self._records[index][name][self._key(e)]) != 0:
                        self._records_per_epoch[index][name].append(mean(self._records[index][name][self._key(e)]))        
        return



    def plot_train_validation_metric_curve(self , metric : tp.Optional[str] = None) -> None:
        """
        Plots the training and validation metric curves.

        Parameters
        ----------
        metric : str, optional
            The metric to plot. Defaults to Loss func.
        """
        if metric is None:
            metric = self._loss._get_name()

        plt.figure(figsize=(10, 8))
        plt.grid(visible=True, which='both', axis='both')

        # Get Keys 
        trainkey , validkey = self.minimized_record.keys()
        
        
        plt.plot(range(1, len(self.minimized_record[trainkey][metric]) + 1), 
                self.minimized_record[trainkey][metric], color='red',
                linestyle='-', marker='o', markersize=5, label='Training Curve',
                alpha=0.5)

        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'Training and Validation {metric} Curves')


        # If validation is non empty
        if len(self.minimized_record[validkey]) != 0:
            plt.plot(range(1, len(self.minimized_record[validkey][metric]) + 1), 
                    self.minimized_record[validkey][metric], color='purple',
                    alpha=0.5,linestyle='-', marker='s', markersize=5, 
                    label='Validation Curve')
        else:
            raise RuntimeWarning('Validation is not recorded! Only training was recorded by register')

        
        plt.legend()
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(linestyle='dotted', linewidth=0.5)
        plt.tight_layout()
        plt.show()


    @property
    def records(self) -> dict:
        return self._records

    @property
    def minimized_record(self) -> dict:
        if hasattr(self, '_records_per_epoch'):
            return self._records_per_epoch
        else:
            raise RuntimeError('Run Register._minimize_per_epoch in the last epoch of training loop')
        
    
    def __getitem__(self, key) -> dict:
        return self.records[key]
    

    def __repr__(self) -> str:
        if self.cycle is not None:
            return f'Record(train, valid), Cycle = {self.cycle}'
        else:
            return 'Record(train, valid)'


        