"""This script defines an NNtrainer class, which is a trainer class for neural networks.

NNtrainer:
==========

    A trainer class for neural networks.
    
    Parameters:
    ----------
        model (torch.nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer): The optimizer for model parameters.
        loss (torch.nn.Module): The loss function.
        seed (float, optional): The seed value. Defaults to None.
        device (torch.device, optional): The device to run the model on. Defaults to None.
        lr_scheduler (torch.optim.lr_scheduler.LRScheduler, optional): The learning rate scheduler. Defaults to None.
    

Attributes:
    -----------
        available_metric (dict): A dictionary of available evaluation metrics.
        best (float): The best loss achieved during training.
        scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler.
        device (torch.device): The device to run the model on.
        cycle (int): The training cycle number.
    

Methods:
    --------
        train(trainloader, valloader=None, epoch=100, log_every_batch=None, restart=False, early_stopping=False,
              validate_every_epoch=None, record_loss=True, metrics=None, *args, **kwargs): Trains the model.
        validate(valloader, metrics=None, *args, **kwargs): Validates the model on a validation set.
        get_loss(): Returns the training loss.
        plot_train_validation_metric_curve(metric='primary'): Plots the training and validation metric curves.
        predict(X): Makes predictions using the trained model.
    
    Note: For detailed documentation of the class methods, attributes, and properties, please refer to the docstrings within the code.
"""

import typing as tp
import warnings
from time import time

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore[import]

from ..skeletons import BaseTrainer, MemPool, Register, RingBuffer, get_logger

logger = get_logger("NNtrainer")


__all__ = ["NNtrainer"]


class NNtrainer(BaseTrainer):
    """A trainer class for neural networks.

    ...

    Attributes:
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

    Methods:
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
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: torch.nn.modules.loss._Loss,
        seed: tp.Optional[float] = None,
        device: tp.Optional[torch.device] = None,
        lr_scheduler: tp.Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        max_persistence_over_cyces: int = 5,
    ) -> None:
        """Initializes the NNtrainer.

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
        max_persistence_over_cyces : int
            Maximum amount of registers hold over cycle
        """
        # Initlize
        super().__init__(model, optimizer, loss)

        # check optimizer link
        self._check_optimizer_model_link()
        # Get the seed
        self.seed = seed

        # set lr scheduler
        self.scheduler = lr_scheduler
        self._check_optimizer_lr_link()

        # set device
        if device is not None:
            self.device = device

        # Moves model to device : default is cuda
        self._move_to_device()

        # set ring buffer attribute
        self._regiser_buffer = RingBuffer(max_persistence=max_persistence_over_cyces)

        # Training Cycles
        self.cycle: int = 0

        return

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Getter method for the optimizer property.

        Returns:
            torch.optim.Optimizer: The current optimizer used by the trainer.

        Example:
            # Initialize trainer
            trainer = MyTrainer(model, optimizer, loss_function)

            # Access the optimizer
            current_optimizer = trainer.optimizer
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optim: torch.optim.Optimizer) -> None:
        """Setter method for the optimizer property.

        Parameters:
            optim (torch.optim.Optimizer): The new optimizer to be set for the trainer.

        Note:
            This method sets a new optimizer for the trainer. If a learning rate scheduler is already
            attached to the current optimizer, it will also be linked to the new optimizer to ensure
            consistent behavior during training.

        Example:
            # Initialize trainer
            trainer = MyTrainer(model, optimizer, loss_function)

            # Create a new optimizer
            new_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Set the new optimizer for the trainer
            trainer.optimizer = new_optimizer
        """
        if self.scheduler is not None:
            self.scheduler.optimizer = optim

        self._optimizer = optim

        return

    @property
    def register_buffer(self) -> RingBuffer:
        """Access register ring buffer property.

        _This gives user acces to the old training cycle registers.

        Returns:
        -------
            Ring Buffer of the registers
        """
        return self._regiser_buffer

    @register_buffer.setter
    def register_buffer(self) -> tp.NoReturn:
        """Cannot set register buffer. Fixed for a trainer instance.

        Returns:
        -------
            tp.NoReturn
        """
        raise RuntimeError("Cannot set Register Ring Buffer")

    def _check_optimizer_model_link(self) -> None:
        """Check if the optimizer is linked with the model parameters."""
        logger.debug(msg="Checking Optimizer-Model Link in NNtrainer")
        if not self.optimizer.param_groups[0]["params"] == list(
            self.model.parameters()
        ):
            logger.error(
                "Optimizer passed to NNtrainer is not linked with model parameters. optimizer.step() cannot work"
            )
            raise RuntimeError

    def _check_optimizer_lr_link(self) -> None:
        """Check if the optimizer and scheduler are properly linked."""
        if self.scheduler is not None:
            logger.debug(msg="Checking Optimizer-Scheduler Link in NNtrainer")
            if not self.scheduler.optimizer == self.optimizer:
                logger.error(
                    msg="Scheduler not linked with the optimizer! Cannot perform lr.step()"
                )
                raise RuntimeError
        else:
            logger.debug(msg="No lrscheduler is passed in the NNtrainer")

    def train(  # type: ignore[override]
        self,
        trainloader: DataLoader,
        valloader: tp.Optional[DataLoader] = None,
        epoch: int = 100,
        log_every_x_batch: tp.Optional[int] = None,
        validate_every_x_epoch: int = 1,
        record_loss: bool = True,
        metrics: tp.Optional[tp.Union[tp.List[str], str]] = None,
        checkpoint_file: tp.Optional[str] = None,
        checkpoint_every_x: int = -1,
        multiclass_reduction_strategy_for_metric: str = "micro",
        weight_init: bool = False,
        swa_model: tp.Optional[torch.nn.Module] = None,
        swa_scheduler: tp.Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        swa_start_epoch: tp.Optional[int] = None,
    ) -> None:
        """Trains the model.

        Parameters
        ----------
        trainloader : DataLoader
            The data loader for the training set.
        valloader : DataLoader, optional
            The data loader for the validation set. Defaults to None.
        epoch : int, optional
            The number of epochs to train. Defaults to 100.
        log_every_x_batch : int, optional
            Log training progress every x batches. Defaults to None.
        restart : bool, optional
            Restart training from the beginning. Defaults to False.
        validate_every_x_epoch : int, optional
            Perform validation every x epochs. Defaults to 1.
        record_loss : bool, optional
            Record training loss and metrics. Defaults to True.
        metrics : Union[Iterable[str], str], optional
            Evaluation metrics to calculate. Defaults to None.
        checkpoint_file : str, optional
            File name to save model checkpoints. Defaults to None.
        checkpoint_every_x : int, optional
            Save model checkpoint every x epochs. Defaults to -1.

        Returns:
        -------
        None
        """
        # Set model to training mode
        self.model.train()
        logger.debug(f"Setting model to train for OBID = {id(self)}")

        # Initilize Weights using Xaviers Uniform Weight init
        if weight_init:
            self.model.apply(self._weight_init)

        # If record loss, set ring-buffer to online
        if record_loss:
            self._regiser_buffer._online = True
            # Enqueue register in the ring buffer
            self._regiser_buffer.enqueue(
                Register(
                    metrics=metrics,
                    loss=self.loss,
                    epoch=epoch,
                    cycle=self.cycle + 1,
                    multiclass_reduction_strategy=multiclass_reduction_strategy_for_metric,
                )
            )

            logger.debug("Register Buffer is online!")
            logger.debug(
                f"Enquing register in the ring buffer {self._regiser_buffer.peek.__repr__()}"
            )

        # if passed seed
        if self.seed:
            torch.manual_seed(self.seed)

        # SWA SPECIFIC Setup
        swa_flag = False
        if swa_model is not None:
            assert isinstance(
                swa_model, torch.optim.swa_utils.AveragedModel
            ), "swa model is not an instanc of torch.optim.swa_utils.AveragedModel"
            assert isinstance(
                swa_scheduler, torch.optim.swa_utils.SWALR
            ), "swa model is not an instanc of torch.optim.swa_utils.SWALR"
            assert 1 <= swa_start_epoch < epoch, "swa epoch must be in [1, epoch)"
            self.swa_model = swa_model
            self.swa_scheduler = swa_scheduler
            self.swa_start_epoch = swa_start_epoch

            if not hasattr(self, 'swa_register_buffer'):
                self.swa_register_buffer = RingBuffer(
                    self._regiser_buffer.max_persistence
                )

            if record_loss and valloader is not None:
                self.swa_register_buffer.enqueue(
                    Register(
                        metrics=metrics,
                        loss=self.loss,
                        epoch=epoch,
                        cycle=self.cycle,
                        multiclass_reduction_strategy=multiclass_reduction_strategy_for_metric,
                    )
                )

            swa_flag = True
            self.swa_register_buffer._online = True

        # Start the training cycle
        self._run_train_cycle(
            trainloader=trainloader,
            valloader=valloader,
            epoch_min=0,
            epoch_max=epoch,
            validate_every_epoch=validate_every_x_epoch,
            checkpoint_file=checkpoint_file,
            log_every_batch=log_every_x_batch,
            save_every=checkpoint_every_x,
            swa_flag=swa_flag,
        )

        return

    def train_from_checkpoint(  # type: ignore[override]
        self,
        check_point: str,
        trainloader: DataLoader,
        valloader: tp.Optional[DataLoader] = None,
        epoch: int = 100,
        log_every_x_batch: tp.Optional[int] = None,
        validate_every_x_epoch: int = 1,
        record_loss: bool = True,
        metrics: tp.Optional[tp.Union[tp.List[str], str]] = None,
        checkpoint_file: tp.Optional[str] = None,
        checkpoint_every_x: int = -1,
        multiclass_reduction_strategy_for_metric: str = "micro",
    ) -> None:
        """Resume training from a saved checkpoint.

        Parameters
        ----------
        check_point : str
            The file path of the checkpoint.
        trainloader : DataLoader

        valloader : DataLoader, optional
            The data loader for the validation set. Defaults to None.
        epoch : int, optional
            The number of epochs to train. Defaults to 100.
        log_every_x_batch : int, optional
            Log training progress every x batches. Defaults to None.
        validate_every_x_epoch : int, optional
            Perform validation every x epochs. Defaults to 1.
        record_loss : bool, optional
            Record training loss and metrics. Defaults to True.
        metrics : Union[Iterable[str], str], optional
            Evaluation metrics to calculate. Defaults to None.
        checkpoint_file : str, optional
            File name to save model checkpoints. Defaults to None.
        checkpoint_every_x : int, optional
            Save model checkpoint every x epochs. Defaults to -1.

        Returns:
        -------
        None
        """
        # Load the checkpoint file
        checkpoint_dict = torch.load(f=check_point, map_location=self.device)
        logger.debug("Checkpoint file loaded")

        # Manual Seed
        if self.seed:
            torch.manual_seed(self.seed)

        # Set Cycle to previous cycle
        self.cycle = checkpoint_dict["cycle"]

        # Assert that current epoch is larger
        assert (
            epoch > checkpoint_dict["epoch"]
        ), f'Passed Epoch:{epoch} must be strictly greater that the saved Epoch:{checkpoint_dict["epoch"]}'

        logger.debug("Loading State for model and optimizer")
        # Load the states of the optimizer and model
        self.model.load_state_dict(checkpoint_dict["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint_dict["optimizer_state_dic"])

        # Set model to training mode
        self.model.train()
        logger.debug(f"Setting model to train for OBID = {id(self)}")

        logger.debug("Resuming the Registry")

        # Resume Ring Buffer| This is a new buffer that will not load the old buffer from checkpoint
        if record_loss:
            self._regiser_buffer._online = True
            print(self._regiser_buffer)
            self._regiser_buffer.enqueue(
                Register(
                    metrics=metrics,
                    loss=self.loss,
                    epoch=(epoch - checkpoint_dict["epoch"]),
                    multiclass_reduction_strategy=multiclass_reduction_strategy_for_metric,
                    cycle=self.cycle + 1,
                )
            )

            logger.debug("Register Buffer is online!")
            logger.debug(
                f"Enquing register in the ring buffer {self._regiser_buffer.__repr__()}"
            )

        logger.info(
            f'Resuming Training from Epoch:{checkpoint_dict["epoch"]} to Epoch:{epoch}'
        )

        # Run the training cycle
        self._run_train_cycle(
            trainloader=trainloader,
            valloader=valloader,
            epoch_min=0,
            epoch_max=(epoch - checkpoint_dict["epoch"]),
            validate_every_epoch=validate_every_x_epoch,
            log_every_batch=log_every_x_batch,
            checkpoint_file=checkpoint_file,
            save_every=checkpoint_every_x,
        )

        return

    def _run_train_cycle(
        self,
        trainloader: DataLoader,
        valloader: tp.Optional[DataLoader],
        epoch_min: int,
        epoch_max: int,
        validate_every_epoch: int,
        checkpoint_file: tp.Optional[str],
        log_every_batch: tp.Optional[int],
        save_every: int = -1,
        swa_flag: bool = False,
    ) -> None:
        """Run the training cycle.

        Parameters
        ----------
        trainloader : DataLoader
            The data loader for the training set.
        valloader : DataLoader
            The data loader for the validation set.
        epoch_min : int
            The minimum epoch number.
        epoch_max : int
            The maximum epoch number.
        validate_every_epoch : int
            Perform validation every x epochs.
        checkpoint_file : str, optional
            File name to save model checkpoints.
        record_loss : bool
            Record training loss and metrics.
        log_every_batch : int, optional
            Log training progress every x batches.
        save_every : int, optional
            Save model checkpoint every x epochs. Defaults to -1.

        Returns:
        -------
        None
        """
        # Start the training
        logger.info(
            f"--------------START OF  {self.cycle + 1} TRAINING CYCLE---------------------"
        )

        assert (
            validate_every_epoch >= 0
        ), f"Got {validate_every_epoch} for validate_every_x_epoch but it must be in [0, inf) range where 0 means do not validate"

        # Start the cycle
        for e in tqdm(
            range(epoch_min, epoch_max),
            desc=f"Train Cycle: {self.cycle + 1} , Epoch",
            colour="blue",
            ncols=80,
            position=0,
        ):
            for idx, (feature, lable) in enumerate(trainloader):
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
                    if (e + 1) % log_every_batch == 0:
                        logger.info(
                            f"Epoch {e}, Batch: {idx}, Loss: {loss.data.item():.3f}..."
                        )

                # Register the metrics if buffer is online
                if self._regiser_buffer._online:
                    self._regiser_buffer.peek._record_batch(
                        y_pred=fp.data, y_true=lable, epoch=e, where=True
                    )

            # Evaluate on validation set
            if validate_every_epoch != 0 and valloader is not None:
                if self._regiser_buffer._online:
                    if (e + 1) % validate_every_epoch == 0:
                        self._validate(valloader=valloader, epoch=e, swa_flag=swa_flag)
                else:
                    raise ValueError(
                        "Validation Data Loader Passed but record_loss is False. No point in validataion. Pass record_loss = True explicitly"
                    )

            # SWA Flags and more
            if swa_flag and (e + 1) >= self.swa_start_epoch:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
            else:
                # Trigger LR Scheduler after validation | if none , trigger based on loss on training for ReduceLROnPlateau
                if self.scheduler is not None:
                    if isinstance(
                        self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        assert (
                            validate_every_epoch == 1
                        ), "Cannot use one validation loss for one epoch and training loss for another epoch to ReduceLROnPlateau Scheduler. Use validate_every_epoch = 1 in training."
                        if validate_every_epoch != 0 and valloader is not None:
                            self.scheduler.step(
                                sum(
                                    self._regiser_buffer.peek.records["valid"][
                                        self.loss._get_name()
                                    ][f"Epoch_{e}"]
                                )
                            )
                        else:
                            self.scheduler.step(
                                sum(
                                    self._regiser_buffer.peek.records["train"][
                                        self.loss._get_name()
                                    ][f"Epoch_{e}"]
                                )
                            )
                    else:
                        self.scheduler.step()

            # Save Checkpoint if available
            if checkpoint_file:
                # This refers to saving last epoch
                if save_every == -1:
                    self._save_checkpoint(
                        epoch=e, filename=checkpoint_file + "_checkpoint.bin"
                    )
                # Save every epoch
                elif (e + 1) % save_every == 0:
                    self._save_checkpoint(
                        epoch=e, filename=checkpoint_file + f"_checkpoint{e}.bin"
                    )

            # Record traning time per epoch
            if self._regiser_buffer._online:
                self._regiser_buffer.peek._record_train_time_per_epoch(
                    epoch=e, time=time()
                )
                if swa_flag:
                    self._regiser_buffer.peek._record_train_time_per_epoch(
                        epoch=e, time=time()
                    )

        # Minimize the Register
        if self._regiser_buffer._online:
            self._regiser_buffer.peek._minimize_per_epoch()

        if swa_flag:
            # SWA Batch Statistics update
            torch.optim.swa_utils.update_bn(trainloader, self.swa_model)
            self.swa_register_buffer.peek._minimize_per_epoch()

        # Log
        logger.info(
            f"--------------END OF  {self.cycle + 1} TRAINING CYCLE---------------------"
        )

        # Increment Cycle
        self.cycle += 1

        # Set Buffer to offline after training
        self._regiser_buffer._online = False

    def _validate(  # type: ignore[override]
        self,
        valloader: DataLoader,
        epoch: tp.Optional[int] = None,
        swa_flag: bool = False,
    ) -> float:
        """Validates the model on a validation set.

        Parameters
        ----------
        valloader : DataLoader
            The data loader for the validation set.
        metrics : Union[Iterable[str], str], optional
            Evaluation metrics to calculate. Defaults to None.

        Returns:
        -------
        float
            The validation loss.
        """
        # Set to eval
        self.model.eval()
        if swa_flag:
            self.swa_model.eval()

        logger.debug(f"Setting model to eval for OBID = {id(self)}")

        with torch.no_grad():
            loss = 0
            for feature, lable in valloader:
                feature = feature.to(self.device)
                lable = lable.to(self.device)
                fp = self.model(feature)
                loss += self.loss(fp, lable)

                if self._regiser_buffer._online and epoch is not None:
                    self._regiser_buffer.peek._record_batch(
                        y_pred=fp, y_true=lable, epoch=epoch, where=False
                    )

                if swa_flag:
                    fp_swa = self.swa_model(feature)
                    if self.swa_register_buffer._online and epoch is not None:
                        self.swa_register_buffer.peek._record_batch(
                            y_pred=fp_swa, y_true=lable, epoch=epoch, where=False
                        )

        # Set to train
        self.model.train()
        if swa_flag:
            self.swa_model.train()

        logger.debug(f"Setting model to train for OBID = {id(self)}")
        return loss

    def plot_train_validation_metric_curve(
        self, metric: tp.Optional[str] = None
    ) -> None:
        """Plots the training and validation metric curves for last recorded cycle.

        Parameters
        ----------
        metric : str, optional
            The metric to plot. Defaults to loss.
        """
        if metric is None:
            metric = self.loss._get_name()

        if not self._regiser_buffer.is_empty():
            ax = self._regiser_buffer.peek.plot_train_validation_metric_curve(
                metric=metric
            )
        else:
            raise RuntimeError(
                "Register buffer is empty. Metrics not Recorded! Pass record_loss=True"
            )

        if hasattr(self, 'swa_register_buffer'):
            ax.plot(
                range(0, len(self.swa_register_buffer.peek['valid'][metric])),
                self.swa_register_buffer.peek['valid'][metric].values,
                color="green",
                marker="h",
                markersize=5,
                label="SWA Model Validation Curve",
                alpha=0.5,
            )

            ax.legend()
            plt.tight_layout()
            plt.plot()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Makes predictions using the trained model.

        Parameters
        ----------
        X : torch.tensor
            The input tensor for making predictions.

        Returns:
        -------
        torch.tensor
            The predicted tensor.
        """
        # Set to evaluation
        logger.debug(f"Setting model to eval for OBID = {id(self)}")
        self.model.eval()

        with torch.no_grad():
            X = X.to(self.device)
            out = self.model(X).cpu()

        # Reset to train mode
        logger.debug(f"Setting model to train for OBID = {id(self)}")
        self.model.train()

        return out

    def _save_checkpoint(self, epoch: int, filename: str) -> None:
        """Save checkpoint of the model.

        This function saves the model parameters, optimizer state, and optionally the register attribute (if present) as a checkpoint file.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        filename : str
            The filename to save the checkpoint.

        Returns:
        -------
        None
        """
        # Saving parameters
        save_dict = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dic": self.optimizer.state_dict(),
            "cycle": self.cycle,
        }

        # If has non empty ring register buffer attribute, save it as well
        if not self._regiser_buffer.is_empty():
            save_dict["register_buffer"] = self._regiser_buffer

        torch.save(save_dict, f=filename)

    def num_model_parameters(self) -> int:
        """Returns the total number of parameters in the model.

        Returns:
        -------
        int
            The total number of parameters in the model.
        """
        return sum(map(torch.numel, self.model.parameters()))

    def mem_report(
        self,
        batch_shape: tp.Union[torch.Size, tp.Iterable[int]],
        batch_dtype: torch.dtype = torch.float32,
        recduction: str = "MB",
    ) -> float:
        """Calculate memory usage for a given batch shape and data type.

        Parameters
        ----------

            batch_shape (Union[torch.Size, Iterable[int]]): Shape of the input batch.
            batch_dtype (torch.dtype, optional): Data type of the input batch (default=torch.float32).
            reduction (str, optional): Reduction format for memory usage (default='MB').

        Returns:
        -------
            float: Total memory usage per batch in the specified reduction format.

        Note:
        -----
            This method initializes a memory pool for the trainer and model specified in the object,
            and then calculates the memory usage based on the given input batch shape and data type.
        """
        logger.debug(
            f"initiating a memory pool for the trainer_id {id(self)} and for model {id(self.model)} on device {self.device}"
        )
        mempool = MemPool(self.model, self.device)  # type: ignore[arg-type]

        # CalcMemUse
        return mempool.calc_memusage(batch_shape, batch_dtype, recduction)

    def compile(self, *args, **kwargs) -> None:  # noqa
        """Compiles the module with provided arguments and keyword arguments.

        This method compiles the model and sets the compiled model as the new model for the instance.
        The compilation process may involve optimizations and other backend-specific operations.

        Parameters:
        -----------
            *args: Variable-length argument list.
                Additional arguments that can be passed to the torch.compile.

            **kwargs: Variable-length keyword argument list.
                Additional keyword arguments that can be passed to the torch.compile.

        Returns:
        --------
            None
        """
        logger.debug(
            "Compiled the model and reset self.model with self.model = torch.compile(self.model)"
        )
        self._model = torch.compile(self.model, *args, **kwargs)

        return

    def no_scheduler(self) -> None:
        """Disables the learning rate scheduler for the trainer.

        Note:
        -----
            This method sets the learning rate scheduler to None, effectively disabling any
            learning rate schedule during training. The optimizer's learning rate will remain
            constant throughout the training process.

        Example:
        --------
            # Initialize trainer
            trainer = MyTrainer(model, optimizer, loss_function)

            # Disable the learning rate scheduler
            trainer.no_scheduler()

            # Start training with a constant learning rate
            trainer.train()
        """
        self.scheduler = None
        pass

    def _link_scheduler(
        self,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        initial_lr: tp.Optional[float] = None,
    ) -> None:
        """Links the specified learning rate scheduler to the trainer.

        Parameters:
        -----------
            scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler to be linked.
            initial_lr (float, optional): The initial learning rate to set for the optimizer. Default is None.

        Raises:
        -------
            AssertionError: If initial_lr is not greater than 0.

        Note:
        -----
            This method links the provided learning rate scheduler to the trainer's optimizer.
            If an initial_lr is provided, it sets this learning rate for all parameter groups in the optimizer.
            Otherwise, the initial learning rate in the optimizer remains unchanged.
        """
        assert initial_lr > 0, "Initial Lr should not be in (0, inf)"

        if self.scheduler is not None:
            warnings.warn(
                f"LR scheduler is already set! Overwriting current lr scheduler with {scheduler.__class__}"
            )

        self.scheduler = scheduler
        self._check_optimizer_lr_link()

        # Set initial to all param group
        if initial_lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = initial_lr

        return

    def add_expdecay_lrs(
        self, lr_initial: float, lr_final: float, epoch: int, verbose: bool = False
    ) -> None:
        """Adds an exponential learning rate scheduler to the trainer's optimizer.

        Parameters:
        -----------
            lr_initial (float): The initial learning rate.
            lr_final (float): The final learning rate. Should be smaller than lr_initial.
            epoch (int): The total number of epochs for the exponential decay.
            verbose (bool, optional): Whether to print verbose output during training. Default is False.

        Raises:
        -------
            AssertionError: If lr_final is not smaller than lr_initial.

        Note:
        -----
            This method sets an exponential learning rate scheduler using torch.optim.lr_scheduler.ExponentialLR
            to the trainer's optimizer. The learning rate will decay exponentially from lr_initial to lr_final
            over the specified number of epochs.

        Example:
        --------
            # Initialize trainer
            trainer = MyTrainer(model, optimizer, loss_function)

            # Add exponential learning rate scheduler
            trainer.add_expdecay_lrs(lr_initial=0.1, lr_final=0.01, epoch=10, verbose=True)

            # Start training
            trainer.train()
        """
        assert lr_final < lr_initial, "lr_final should be less than lr_initial"

        # gamma calculation
        gamma_ = pow(lr_final / lr_initial, 1 / epoch)

        exp_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=gamma_, verbose=verbose
        )

        # Link Scheduler
        self._link_scheduler(exp_scheduler, initial_lr=lr_initial)
        logger.debug(
            f'{self.scheduler.__class__} set to optimizer {self.optimizer.__repr__()}'
        )

        return

    def add_stepdecay_lrs(
        self,
        lr_initial: float,
        lr_final: float,
        step: int,
        epoch: int,
        verbose: bool = False,
    ) -> None:
        """Adds a step decay learning rate scheduler to the trainer's optimizer.

        Parameters:
        -----------
            lr_initial (float): The initial learning rate.
            lr_final (float): The final learning rate. Should be smaller than lr_initial.
            steps (int): The number of drops in lrs
            epoch (int): The total number of epochs for the step decay.
            verbose (bool, optional): Whether to print verbose output during training. Default is False.

        Raises:
        -------
            AssertionError: If lr_final is not smaller than lr_initial or if step_size is not smaller than epoch.

        Note:
        -----
            This method sets a step decay learning rate scheduler using torch.optim.lr_scheduler.StepLR
            to the trainer's optimizer. The learning rate will be reduced from lr_initial to lr_final in
            a stepwise manner every step_size epochs.

        Example:
        --------
            # Initialize trainer
            trainer = MyTrainer(model, optimizer, loss_function)

            # Add step decay learning rate scheduler
            trainer.add_stepdecay_lrs(lr_initial=0.1, lr_final=0.01, step_size=5, epoch=20, verbose=True)

            # Start training
            trainer.train()

        """
        assert lr_final < lr_initial, "lr_final should be less than lr_initial"
        assert 1 <= step < epoch, "Step size should be in [1, inf)"

        # Calculate gamma
        gamma = pow(lr_final / lr_initial, 1 / step)

        scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, epoch // step, gamma, verbose=verbose
        )

        self._link_scheduler(scheduler=scheduler, initial_lr=lr_initial)
        logger.debug(
            f'{self.scheduler.__class__} set to optimizer {id(self.optimizer)}'
        )

        return

    def add_cosineannealing_lrs(
        self,
        lr_initial: float,
        lr_final: float,
        dips: int,
        epoch: int,
        verbose: bool = False,
    ) -> None:
        """Adds a Cosine Annealing learning rate scheduler to the trainer's optimizer.

        Parameters:
        -----------
            lr_initial (float): The initial learning rate.
            lr_final (float): The final learning rate.
            dips (int): no of dips in the cosine. Use dips < epoch // 4
            epoch (int) : max no of epoch
            verbose (bool, optional): Whether to print verbose output during training. Default is False.

        Note:
        -----
            This method sets a Cosine Annealing learning rate scheduler using torch.optim.lr_scheduler.CosineAnnealingLR
            to the trainer's optimizer. The learning rate will anneal from lr_initial to lr_final following a cosine
            annealing schedule over the specified number of epochs.

        Example:
        --------
            # Initialize trainer
            trainer = MyTrainer(model, optimizer, loss_function)

            # Add Cosine Annealing learning rate scheduler
            trainer.add_cosineannealing_lrs(lr_initial=0.1, lr_final=0.01, epoch=50, verbose=True)

            # Start training
            trainer.train()

        """
        # Calculate T_max for the CosineAnnealingLR scheduler
        T_max = epoch / (dips * 2 - 1)

        cos_annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=T_max, eta_min=lr_final, verbose=verbose
        )
        # Link Scheduler
        self._link_scheduler(cos_annealing_scheduler, initial_lr=lr_initial)
        logger.debug(
            f'Cosine Annealing LRscheduler set to optimizer {id(self.optimizer)}'
        )
        return

    def add_gradient_clipping(self, min: float, max: float) -> None:
        """Add gradient clipping to the model.

        This method adds a hook to the backward pass of the model's parameters, which clips the gradients
        during backpropagation to prevent exploding gradients. Gradient clipping helps stabilize the training
        process and can prevent the model from diverging during optimization.

        Parameters
        ----------
        min : float
        The minimum value to which gradients will be clipped.
        Gradients with values lower than this will be set to `min`.
        max : float
        The maximum value to which gradients will be clipped.
        Gradients with values higher than this will be set to `max`.

        Returns:
        -------
        None

        Notes:
        -----
        Gradient clipping is often used to mitigate the issue of exploding gradients, which can occur when
        training deep neural networks. By setting a range for the gradients, the model's parameters are
        adjusted in a way that prevents them from taking very large steps during optimization, thus promoting
        more stable and reliable training.

        """
        # Add a hook for the backward method to clip gradients
        for param_groups in self.model.parameters():
            param_groups.register_hook(
                lambda gradients: torch.clip(gradients, min, max)
            )

        logger.debug(f'Clipped the gradients to min : {min} max : {max}')

        return
