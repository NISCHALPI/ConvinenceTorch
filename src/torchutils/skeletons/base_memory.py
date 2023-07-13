"""This script defines two classes: MemNode and MemPool, used for tracking memory usage of neural network modules.

MemNode:
========

    Memory Node class to track memory usage of a neural network module.
    It takes a base model as input and provides methods to calculate and display memory statistics.
    The key attributes of the class are:
    - _model: The base model being tracked.
    - _input: Input tensor used for calculating stats.
    - _dtype_size: Size of the data type in bytes.
    - _param_mem: Memory usage of parameters if available.
    - _output: Output tensor of the base model.
    
    The class provides the following methods:
    - _calc_stats(input: torch.Tensor) -> None: Calculates memory statistics for the base model.
    - _to(size_in_bytes: int, reduction: str = 'MB') -> float: Converts the size in bytes to the specified reduction (MB, KB, B, GB).
    - _prettyprint(reduction: str = 'MB') -> None: Prints a formatted summary of memory usage.
    - forward(input: torch.Tensor, reduction: str = 'MB') -> torch.Tensor: Runs the base model on the input and returns the output.

MemPool:
========

    Memory Pool class to track memory usage of a neural network model.
    It takes a model and a device as input and provides methods to calculate memory usage for different input shapes.
    The key attributes of the class are:
    - _model: The model being tracked.
    - _device: The device on which the model is located.
    - _pooled_memnodes: List of MemNode instances for each base module.
    
    The class provides the following methods:
    - _init_pool() -> None: Initializes the memory pool by creating MemNode instances for each base module.
    - _calc_memusage(input_shape: tp.Union[torch.Size, tp.Iterable[int]], input_dtype: torch.dtype = torch.float32, reduction: str = 'MB') -> float:
        Calculates memory usage for a given input shape and data type.

Note: For detailed documentation of the methods and their parameters, please refer to the docstrings within the code.
"""

import typing as tp
from queue import Queue

import torch

from .base_log import get_logger

logger = get_logger("base_memory")


__all__ = ["MemNode", "MemPool"]


class MemNode:
    """Memory Node class to track memory usage of a neural network module.

    Args:
    -----
        base_model (torch.nn.Module): Base model to be tracked.

    Attributes:
    -----------
        _model (torch.nn.Module): Base model being tracked.
        _input (torch.Tensor): Input tensor used for calculating stats.
        _dtype_size (int): Size of the data type in bytes.
        _param_mem (int, optional): Memory usage of parameters if available.
        _output (torch.Tensor): Output tensor of the base model.

    Methods:
    --------
        _calc_stats(input: torch.Tensor) -> None:
            Calculates memory statistics for the base model.

        _to(size_in_bytes: int, reduction: str = 'MB') -> float:
            Converts the size in bytes to the specified reduction (MB, KB, B, GB).

        _prettyprint(reduction: str = 'MB') -> None:
            Prints a formatted summary of memory usage.

        forward(input: torch.Tensor, reduction: str = 'MB') -> torch.Tensor:
            Runs the base model on the input and returns the output.
    """

    def __init__(self, base_model: torch.nn.Module) -> None:
        """Constructer for mem node.

        Args:
            base_model (torch.nn.Module): Need a base model which is defined not to contain any submodules.
        """
        self._model = base_model

    def _calc_stats(self, input_tensor: torch.Tensor) -> None:
        # Need Inputs to calculate the stats
        self._input = input_tensor
        self._dtype_size = input_tensor.element_size()

        # Calculate the parameter memory
        if len(list(self._model.parameters())) != 0:
            logger.debug(
                f"For {self._model._get_name()} , {id(self._model)}, parameters are detected!"
            )
            setattr(
                self,
                "_param_mem",
                sum(map(torch.numel, self._model.parameters())) * self._dtype_size,
            )

        # Calculate output with no gradient tracking
        with torch.no_grad():
            self._output = self._model(self._input)
            logger.debug(
                f"Forward pass for For {self._model._get_name()} , {id(self._model)}, sucesseful!"
            )
        # Calculate the output memory usage
        self._out_mem = torch.numel(self._output) * self._dtype_size

        # Set status dict attribute
        self._status_dict = {"name": self._model._get_name(), "out_mem": self._out_mem}
        if hasattr(self, "_parm_mem"):
            self._status_dict["param_mem"] = self._param_mem

    @staticmethod
    def _to(size_in_bytes: int, reduction: str = "MB") -> float:
        if reduction == "MB":
            return size_in_bytes / (1024) ** 2

        if reduction == "KB":
            return size_in_bytes / 1024

        if reduction == "B":
            return size_in_bytes

        if reduction == "GB":
            return size_in_bytes / (1024**3)

        raise RuntimeError("reduction must be one of MB | GB | B | KB")

    def _prettyprint(self, reduction: str = "MB") -> None:
        print(
            "------------------------------------------------------------------------------"
        )
        print(f"For Module: {self._model._get_name()}")
        print(
            f"        Total Output Mem : { (self._to(self._out_mem, reduction=reduction) * 2):.3f}  {reduction}"
        )
        if hasattr(self, "_param_mem"):
            print(
                f"       Total Param Mem :{(self._to(self._param_mem, reduction=reduction) * 2):.3f} {reduction}"
            )

        print(
            "------------------------------------------------------------------------------"
        )

    def forward(
        self, input_tensor: torch.Tensor, reduction: str = "MB"
    ) -> torch.Tensor:
        """Executes a forward pass for the node.

        Args:
            input_tensor (torch.Tensor): Input for the Node
            reduction (str, optional): Type of recduction to perform. Defaults to "MB".

        Returns:
            torch.Tensor: output of the forward pass.
        """
        # Run Status Tracker
        self._calc_stats(input_tensor)

        # Run PrettyPrint
        self._prettyprint(reduction)

        # Return Status dict
        return self._output


class MemPool:
    """Memory Pool class to track memory usage of a neural network model.

    Args:
        model (torch.nn.Module): Model to be tracked.
            The model can be any torch.nn.Module or a nested module.
        device (torch.device): Device on which the model is located.
            This should explicitly be an instance of torch.device.

    Attributes:
        _model (torch.nn.Module): Model being tracked.
        _device (torch.device): Device on which the model is located.
        _node_queue (Queue): Queue of MemNode instances for each base module.

    Methods:
        _init_pool() -> None:
            Initializes the memory pool by creating MemNode instances for each base module.

        calc_memusage(
            input_shape: Union[torch.Size, Iterable[int]],
            input_dtype: torch.dtype = torch.float32,
            reduction: str = "MB"
        ) -> float:
            Calculates memory usage for a given input shape and data type.

    """

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        """Constructor for MemPool class.

        Args:
            model (torch.nn.Module): Model to be tracked.
                The model can be any torch.nn.Module or a nested module.
            device (torch.device): Device on which the model is located.
                This should explicitly be an instance of torch.device.
        """
        self._model = model
        self._device = device
        self._init_pool()

    def _init_pool(self) -> None:
        # Gets both the sequentail and nested modules
        self._node_queue = Queue()

        for (
            _,
            module,
        ) in (
            self._model._modules.items()
        ):  # torch.nn.Module._modules returns an order dics which only contatins base models
            self._node_queue.put(MemNode(module))

        logger.debug(
            f"Init Queue for MemPool {id(self)} has {self._node_queue.qsize} modules"
        )

    def calc_memusage(
        self,
        input_shape: tp.Union[torch.Size, tp.Iterable[int]],
        input_dtype: torch.dtype = torch.float32,
        reduction: str = "MB",
    ) -> float:
        """Calculate the memory usage for a given input shape and data type.

        Parameters:
            input_shape (Union[torch.Size, Iterable[int]]): The shape of the input tensor for the model.
            input_dtype (torch.dtype, optional): Data type of the input tensor. Defaults to torch.float32.
            reduction (str, optional): The reduction to be used in the memory size representation. Defaults to 'MB'.

        Returns:
            float: Total memory usage per batch_size for the given input shape and dtype in the specified reduction.
        """
        input_ = torch.rand(*input_shape, dtype=input_dtype, device=self._device)

        print("---------------------Memory-Usage-Per-Batch---------------------------")
        input_size = MemNode._to(
            input_.numel() * input_.element_size(), reduction=reduction
        )
        print(f"Input Layer Memory: {input_size} {reduction}")

        total_mem = input_size

        # Temp status dict in order from MemNodes
        status_dicts = []
        temp_input = input_

        # Mem Forward Loop
        while not self._node_queue.empty():
            mem_node = self._node_queue.get()
            temp_input = mem_node.forward(temp_input, reduction=reduction)
            status_dicts.append(mem_node._status_dict)
            total_mem += (
                2 * status_dicts[-1]["out_mem"]
            )  # Memory With Gradient i.e output_size * 2

        # Add parameter weights to total memory
        total_mem += (
            sum(map(torch.numel, self._model.parameters())) * input_.element_size()
        )

        # Convert from bytes to appoprate reduction
        total_mem = MemNode._to(total_mem, reduction=reduction)

        print(
            "--------------------------------SUMMARY---------------------------------"
        )
        print(
            f"Total memory usage per batch_size {input_shape} for dtype {input_dtype} : {total_mem:.2f} {reduction}"
        )

        return total_mem
