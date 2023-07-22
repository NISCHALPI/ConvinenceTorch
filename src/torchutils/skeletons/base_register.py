"""This script defines a Register class for recording and evaluating metrics during training.

Register:
==========
    Class for recording and evaluating metrics during training.
    
Parameters:
-----------
    metrics (Union[List[str], str, None]): The metrics to record and evaluate. If None, only the loss metric is used.
    loss (torch.nn.modules.loss._Loss): The loss function used in training.
    epoch (int): The total number of training epochs.
    cycle (Optional[int]): The cycle number (default: None).
    multiclass_reduction_strategy (str): The strategy for reducing multiclass metrics (default: 'micro').
    

Attributes:
-----------
    available_metric (dict): Dictionary mapping metric names to metric functions.
    records (dict): Dictionary storing the recorded metrics for each epoch and dataset split.
    minimized_record (dict): Dictionary storing the mean value of recorded metrics per epoch and dataset split.
    

Methods:
--------
    _multi_classification_enable(func: Callable, avg: str) -> Callable: Decorator that enables multi-classification support for a given sklearn metric.
    _init_multiclass() -> None: Wraps the multiclassification strategy.
    _init_metrics(metrics: Union[List[str], str]) -> None: Initialize the metrics to be recorded and evaluated.
    _key(epoch: int) -> str: Returns the key string for a given epoch.
    _init_records() -> None: Initialize the records dictionary.
    _check_metric(metric_name: str) -> bool: Checks if the given metric is available.
    _eval_metric(y_pred: torch.Tensor, y_true: torch.Tensor, metric_name: str) -> float: Evaluates the metric between predicted and true tensors.
    _record(y_pred: torch.Tensor, y_true: torch.Tensor, epoch: int, where: bool = True) -> None: Record the metrics for a given epoch and dataset split.
    _minimize_per_epoch() -> None: Calculate the mean value of recorded metrics per epoch and dataset split.
    plot_train_validation_metric_curve(metric: Optional[str] = None) -> None: Plots the training and validation metric curves.
    
Properties:
-----------
    records (dict): Get the records dictionary.
    minimized_record (dict): Get the minimized_record dictionary.

Note: For detailed documentation of the class methods, attributes, and properties, please refer to the docstrings within the code.
"""


import functools
import typing as tp
from statistics import mean

import matplotlib.pyplot as plt  # type:ignore[import]
import numpy as np  # type:ignore[import]
import pandas as pd  # type:ignore[import]
import sklearn.metrics as met  # type:ignore[import]
import torch

__all__ = ["Register"]


class Register:
    """Class for recording and evaluating metrics during training.

    Parameters
    ----------
    metrics : Union[List[str], str, None]
        The metrics to record and evaluate. If None, only the loss metric is used.
        If a string, it represents a single metric. If a list of strings, it represents multiple metrics.
    loss : torch.nn.modules.loss._Loss
        The loss function used in training.
    epoch : int
        The total number of training epochs.

    Attributes:
    ----------
    available_metric : dict
        Dictionary mapping metric names to metric functions.

    records : dict
        Dictionary storing the recorded metrics for each epoch and dataset split.

    minimized_record : dict
        Dictionary storing the mean value of recorded metrics per epoch and dataset split.

    Methods:
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
        "accuracy": met.accuracy_score,
        "f1": met.f1_score,
        "l1": met.mean_absolute_error,
        "precision": met.precision_score,
        "recall": met.recall_score,
    }

    def __init__(
        self,
        metrics: tp.Union[tp.List[str], str, None],
        loss: torch.nn.modules.loss._Loss,
        epoch: int,
        cycle: tp.Optional[int] = None,
        multiclass_reduction_strategy: str = "micro",
    ) -> None:
        """Initialize a Register object.

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
        self._metrics: tp.List[str]
        self._loss = loss
        # Set loss to available metric dict
        self.available_metric[loss._get_name()] = loss
        # Set multiclass reduction strategy
        self.multiclass_strategy = multiclass_reduction_strategy
        self._init_multiclass()

        # Record
        self._records: tp.Dict[
            str, tp.Union[tp.Dict[str, tp.Dict[str, list[float]]], list[float]]
        ]

        assert epoch > 0, "Epoch cannot be negative"
        self._epoch = epoch
        self._init_metrics(metrics)
        self._init_records()
        self.cycle = cycle

    @staticmethod
    def _multi_classification_enable(func: tp.Callable, avg: str) -> tp.Callable:
        """Decorator that enables multi-classification support for a given sklearn metric.

        Args:
            func (callable): The function to be decorated.
            avg (str): The averaging strategy for the multi-classification scoring.

        Returns:
            callable: The decorated function.

        Raises:
        -------
            None.

        Examples:
        ---------
            @multi_classification_enable
            def my_classification_function(y_true, y_pred, avg=None):
                # Your classification logic here
                pass

        """

        @functools.wraps(func)
        def wrapper(*args: np.ndarray, **kwargs: str) -> float:
            # Declare Nonlocal
            nonlocal avg
            # Add to kwargs
            kwargs["average"] = avg
            # Evaluate
            return func(*args, **kwargs)

        return wrapper

    def _init_multiclass(self) -> None:
        # Wraps the multiclassification strategy
        for metric_name in ["f1", "recall", "precision"]:
            self.available_metric[metric_name] = self._multi_classification_enable(
                self.available_metric[metric_name], self.multiclass_strategy
            )

    def _init_metrics(self, metrics: tp.Union[tp.List[str], str, None]) -> None:
        """Initialize the metrics to be recorded and evaluated.

        Parameters
        ----------
        metrics : Union[List[str], str]
            The metrics to record and evaluate.
        """
        # Init metric
        if metrics is None:
            self._metrics = [self._loss._get_name()]  # ty
        elif isinstance(metrics, str):
            self._metrics = [self._loss._get_name(), metrics]
        elif isinstance(metrics, list):
            self._metrics = [self._loss._get_name(), *metrics]
        else:
            raise ValueError("Metric type should be one of  List[str] | str | None ")

        # Check New Metrics
        for name in self._metrics:
            self._check_metric(name)

    def _key(self, epoch: int) -> str:
        """Returns the key string for a given epoch.

        Parameters
        ----------
        epoch : int
            The epoch number.

        Returns:
        -------
        str
            The key string.
        """
        return f"Epoch_{epoch}"

    def _init_records(self) -> None:
        """Initialize the records dictionary."""
        # Initialize training time record dict
        self._records = {}
        self._records["time"] = {}

        # Initialize time to zero
        for e in range(0, self._epoch):
            self._records["time"][self._key(e)] = 0.0  # type: ignore

        # initalize the records
        for index in ["train", "valid"]:
            self._records[index] = {}
            for name in self._metrics:
                self._records[index][name] = {}  # type: ignore
                for e in range(0, self._epoch):
                    self._records[index][name][self._key(e)] = []  # type: ignore

    def _check_metric(self, metric_name: str) -> bool:
        """Checks if the given metric is available.

        Parameters
        ----------
        metric_name : str
            The metric to check.

        Returns:
        -------
        bool
            True if the metric is available, False otherwise.
        """
        if (
            metric_name in self.available_metric.keys()
            or metric_name == self._loss._get_name()
        ):
            return True

        raise RuntimeError(f"{metric_name} is not available!")

    def _eval_metric(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, metric_name: str
    ) -> float:
        """Evaluates the metric between predicted and true tensors.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted tensor.
        y_true : torch.Tensor
            The true tensor.
        metric_name : str
            The metric to evaluate.

        Returns:
        -------
        float
            The evaluated metric value.
        """
        # move to cpu
        y_pred_cpu = y_pred.data.cpu()
        y_true_cpu = y_true.data.cpu()

        # Check if metric is from pytorch
        if issubclass(self.available_metric[metric_name].__class__, torch.nn.Module):
            return float(self.available_metric[metric_name](y_pred_cpu, y_true_cpu))

        # If metric is from scikit-learn
        # Apply softmax if logits having column greater than one
        if (
            isinstance(self._loss, (torch.nn.CrossEntropyLoss,))
            and y_pred_cpu.shape[-1] > 1
        ):
            y_pred_cpu = torch.nn.functional.softmax(y_pred_cpu, dim=-1).argmax(
                dim=-1
            )  # Extracts indexes

        y_pred_cpu = y_pred_cpu.numpy().flatten()
        y_true_cpu = y_true_cpu.numpy().flatten()

        return float(self.available_metric[metric_name](y_true_cpu, y_pred_cpu))

    def _record_batch(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, epoch: int, where: bool = True
    ) -> None:
        """Record the metrics for a given epoch and dataset split.

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
        key = "train" if where else "valid"

        for name in self._metrics:
            val = self._eval_metric(y_pred=y_pred, y_true=y_true, metric_name=name)
            self._records[key][name][self._key(epoch)].append(val)  # type: ignore

    def _record_train_time_per_epoch(self, epoch: int, time: float) -> None:
        """Records epoch train time.

        Args:
            epoch (int): crossponding epoch
            time (float): time from time.time()
        """
        self._records["time"][self._key(epoch=epoch)] = time  # type: ignore[assignment,call-overload]

        pass

    def _minimize_per_epoch(self) -> None:
        """Calculate the mean value of recorded metrics per epoch and dataset split.

        This method calculates the mean value of recorded metrics per epoch and dataset split
        and stores the results in a dataframe named `self._records_per_epoch`.

        Returns:
            None
        """
        # Set minimized attribute
        self._records_per_epoch: tp.Dict[str, pd.DataFrame] = {}

        # Minimize to Mean
        col_names = ["time"]
        col_names += self._metrics

        # For Train
        # Index
        index_train = [self._key(e) for e in range(0, self._epoch)]
        # Set up the train
        self._records_per_epoch["train"] = pd.DataFrame(
            data=np.zeros(shape=(len(index_train), len(col_names))),
            index=index_train,
            columns=col_names,
        )
        # Fill train dataframe
        for metric_name in self._records["train"].keys():  # type: ignore
            for epoch in self._records["train"][metric_name].keys():  # type: ignore
                # Non Empty check
                if len(self._records["train"][metric_name][epoch]) != 0:
                    self._records_per_epoch["train"][metric_name][epoch] = mean(self._records["train"][metric_name][epoch])  # type: ignore

        # Adds time
        self._records_per_epoch["train"]["time"] = self._records["time"]

        # Normalize time to start from zero
        self._records_per_epoch["train"]["time"] -= self._records_per_epoch["train"][
            "time"
        ].min()

        # Get non empty validation epoch
        non_empty_epoch = list(
            filter(
                lambda X: len(self._records["valid"][self._loss._get_name()][X]) != 0,
                self._records["valid"][self._loss._get_name()].keys(),
            )
        )

        ## For Validation
        if len(non_empty_epoch) != 0:
            self._records_per_epoch["valid"] = pd.DataFrame(
                data=np.zeros(shape=(len(non_empty_epoch), len(col_names[1:]))),
                index=non_empty_epoch,
                columns=col_names[1:],
            )
            # Fill valid dataframe
            for metric_name in self._records["valid"].keys():  # type: ignore
                for epoch in non_empty_epoch:  # type: ignore
                    # IF non empty register
                    self._records_per_epoch["valid"][metric_name][epoch] = mean(self._records["valid"][metric_name][epoch])  # type: ignore

        return

    def plot_train_validation_metric_curve(
        self, metric: tp.Optional[str] = None
    ) -> plt.Axes:
        """Plots the training and validation metric curves.

        Parameters
        ----------
        metric : str, optional
            The metric to plot. Defaults to Loss func.
        """
        if metric is None:
            metric = self._loss._get_name()

        fig = plt.figure(figsize=(10, 8), dpi=150)
        ax = fig.add_axes(111)
        ax.grid(visible=True, which="both", axis="both")

        # Get Keys
        trainkey, validkey = "train", "valid"

        ax.plot(
            range(0, len(self.minimized_record[trainkey][metric])),
            self.minimized_record[trainkey][metric].values,
            color="red",
            linestyle="-",
            marker="o",
            markersize=5,
            label="Training Curve",
            alpha=0.5,
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_title(f"Training and Validation {metric} Curves")

        # If validation is non empty
        if validkey in self.minimized_record.keys() != 0:
            ax.plot(
                range(0, len(self.minimized_record[validkey][metric])),
                self.minimized_record[validkey][metric].values,
                color="purple",
                alpha=0.5,
                linestyle="-",
                marker="s",
                markersize=5,
                label="Validation Curve",
            )

        ax.grid(linestyle="dotted", linewidth=0.5)
        return ax

    @property
    def records(self) -> dict:
        """Record property of the register onject.

        Returns:
            dict: A dictionary of recorded metric
        """
        return self._records

    @property
    def minimized_record(self) -> tp.Dict[str, pd.DataFrame]:
        """Minimized records for the register. This is a per epoch record.

        Raises:
            RuntimeError: if _minimize_per_epoch is not run before acessing

        Returns:
            dict: minimized record of the recorded metric
        """
        if hasattr(self, "_records_per_epoch"):
            return self._records_per_epoch

        raise RuntimeError(
            "Run Register._minimize_per_epoch in the last epoch of training loop"
        )

    def __getitem__(self, key: str) -> pd.DataFrame:
        """Magic getitem method.

        Args:
            key (str): train | valid

        Returns:
            dict: records relevent to the key
        """
        return self.minimized_record[key]

    def __repr__(self) -> str:
        """The representation of the register.

        Returns:
            str: string representation of the register.
        """
        if self.cycle is not None:
            return f"Register(train, valid, cycle = {self.cycle} , multiclass_strategy: {self.multiclass_strategy})"

        return (
            f"Register(train, valid, multiclass_strategy: {self.multiclass_strategy})"
        )

    @property
    def metrics(self) -> tp.List[str]:
        """Property of available metric.

        Returns:
        -------
            copy of all the available metrics since list is mutable
        """
        return self._metrics.copy()

    @metrics.setter
    def metrics(self) -> tp.NoReturn:
        """Setter."""
        raise RuntimeError("Cannot set metrics property")
