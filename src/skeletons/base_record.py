import numpy as np
import sklearn.metrics as met 
import  torch 
from .base_log import get_logger
import typing as tp




__all__ = ['Register']

logger = get_logger(name='Register')



class Register(object):

    # Available metrics metric System
    available_metric = {
        'accuracy' : met.accuracy_score , 
        'f1' : met.f1_score , 
        'roc' : met.roc_auc_score , 
        'L1' : met.mean_absolute_error,
        'precision':met.precision_score, 
        'recall' : met.recall_score
        }



    def __init__(self, metrics : tp.Union[tp.List[str], str, None] , loss: torch.nn.modules.loss._Loss) -> None:
        
        self.metrics = None
        self.loss = loss
        self.record  = {'train': {}, 'validation':{}}
        self._init_metrics(metrics, loss._get_name())
        self._init_records()
        
        # Add loss to available metric
        self.available_metric[loss._get_name()]= self.loss
        
        pass
    


    def _init_metrics(self, metrics: tp.Union[tp.List[str], str] , primary_name : str) -> None:
        
        # Init metric 
        if metrics is None:
            metrics = [self.loss._get_name()]
        elif isinstance(metrics , str):
            metrics = [metrics, self.loss._get_name()]
        
        elif isinstance(metrics , list):
            metrics.append(self.loss._get_name())

        else:
            raise ValueError('Metric type should be one of  List[str] | str | None ')

        # Check New Metrics
        for name in metrics:
            self._check_metric(name)

        # Add metrics to self.metrics 
        if self.metrics is not None:
            self.metrics = list(set().union(self.metrics, metrics))
        else:
            self.metrics = metrics
    

        return None
    
    def _init_records(self) -> None:
        
        # initalize the records 
        for name in self.metrics:
            for index in ['train', 'vaildation']:
                # If not present add else dont 
                if name not in self.record[index].keys():
                    self.record[index][name] = np.empty(0)
        return None

    def _check_metric(self, metric_name : str) -> bool:
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

        if metric_name in self.available_metric.keys() or metric_name == self.loss._get_name():
            return True
        else:
            raise RuntimeError(f'{metric_name} is not available!')
        
    def cycle_update_records(self, metrics : tp.Union[tp.List[str], str, None]) -> None:
        self._init_metrics(metrics=metrics)
        self._init_records()
        return
    
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
          # Move to CPU and Record
        y_pred_cpu = y_pred.data.cpu()
        y_true_cpu = y_true.data.cpu()

        # Apply softmax if logits 
        if isinstance(self.loss , (torch.nn.CrossEntropyLoss , )):
            y_pred_cpu = torch.nn.functional.softmax(y_pred_cpu, dim=1).argmax(dim=1)
        
        y_pred_cpu = y_pred_cpu.numpy().ravel()
        y_true_cpu   = y_true_cpu.numpy().ravel()


        return self.available_metric[metric_name](y_true_cpu, y_pred_cpu)
        
    def record(self, y_pred: torch.Tensor, y_true: torch.Tensor, where : bool = True) -> None:
        

        key = 'train' if where else 'validation'

        for name in self.metrics:
            #  If it is loss func 
            if issubclass(self.available_metric[name], torch.nn.Module):
                val =  float(self.loss(y_pred, y_true).data())
                np.append(self.record[key][name] , val, axis=0)

            # If scikit learn func 
            else:
                val = self._eval_metric(y_pred=y_pred, y_true=y_true, metric_name=name)
                np.append(self.record[key][name] , val, axis=0)

        return None
