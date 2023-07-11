"""
This script defines two classes: MemNode and MemPool, used for tracking memory usage of neural network modules.

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

import torch
import typing as tp 
from .base_log import get_logger



logger = get_logger('base_memory')


__all__ = ['MemNode', 'MemPool']




class MemNode(object):
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
        self._model = base_model
        

    def _calc_stats(self, input_tensor: torch.Tensor) -> None:
        
    
        # Need Inputs to calculate the stats 
        self._input  = input_tensor
        self._dtype_size = input_tensor.element_size()
        
        # Calculate the parameter memory 
        if len(list(self._model.parameters())) != 0:
            logger.debug(f'For {self._model._get_name()} , {id(self._model)}, parameters are detected!')
            setattr(self, '_param_mem', sum(map(torch.numel, self._model.parameters())) * self._dtype_size)
        
        # Calculate output with no gradient tracking 
        with torch.no_grad():
            self._output=  self._model(self._input)
            logger.debug(f'Forward pass for For {self._model._get_name()} , {id(self._model)}, sucesseful!')
        

        # Calculate the output memory usage 
        self._out_mem= torch.numel(self._output) * self._dtype_size
        
    
        return
     
    @staticmethod
    def _to(size_in_bytes: int , reduction: str = 'MB') -> float:
        
        if reduction == 'MB':
            return size_in_bytes / (1024) ** 2
        
        elif reduction == 'KB':
            return size_in_bytes / 1024

        elif reduction == 'B':
            return size_in_bytes

        elif reduction == 'GB':
            return size_in_bytes / (1024 ** 3)

        raise RuntimeError('reduction must be one of MB | GB | B | KB')

     
    def _prettyprint(self, reduction: str = 'MB') -> None:
        
        print('------------------------------------------------------------------------------')
        print(f'For Module: {self._model._get_name()}')
        print(f'        Total Output Mem : { (self._to(self._out_mem, reduction=reduction) * 2):.3f}  {reduction}')
        if hasattr(self, '_param_mem'):
            print(f'       Total Param Mem :{(self._to(self._param_mem, reduction=reduction) * 2):.3f} {reduction}')

        print('------------------------------------------------------------------------------')
        
        return
    
    def forward(self, input_tensor: torch.Tensor, reduction : str = 'MB') -> torch.Tensor:
        
        # Run Status Tracker 
        self._calc_stats(input_tensor)

        # Run PrettyPrint 
        self._prettyprint(reduction)

        return self._output


class MemPool(object):
    """Memory Pool class to track memory usage of a neural network model.
    
    Args:
    -----
        model (torch.nn.Module): Model to be tracked.
        device (torch.device): Device on which the model is located.
    
    Attributes:
    -----------
        _model (torch.nn.Module): Model being tracked.
        _device (torch.device): Device on which the model is located.
        _pooled_memnodes (list): List of MemNode instances for each base module.
    
    Methods:
    --------
    
        _init_pool() -> None:
            Initializes the memory pool by creating MemNode instances for each base module.
        
        _calc_memusage(input_shape: tuple, input_dtype: torch.dtype = torch.float32, reduction: str = 'MB') -> None:
            Calculates memory usage for a given input shape and data type.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self._model = model
        self._device = device
        self._init_pool()

    def _init_pool(self) -> None:

        # Gets both the sequentail and nested modules
        self._pooled_memnodes = []
        
        for module in self._model.modules():
            if not isinstance(module, torch.nn.Sequential):
                self._pooled_memnodes.append(MemNode(module))

        logger.debug(f'Init Pool for MemPool {id(self)} has {len(self._pooled_memnodes)} modules')
        return
    
    

    def _calc_memusage(self, input_shape: tp.Union[torch.Size, tp.Iterable[int]] , input_dtype: torch.dtype = torch.float32, reduction : str = 'MB') -> float:
        
        input_ = torch.rand(*input_shape, dtype=input_dtype, device=self._device)
        
        print('---------------------Memory-Usage-Per-Batch---------------------------')
        print(f'Input Layer Memory: {MemNode._to(input_.numel() * input_.element_size(), reduction=reduction)} {reduction}')

        # Temp
        temp_out = input_
        total_mem = MemNode._to(input_.numel() * input_.element_size(), reduction=reduction)
        
       
        # Mem Forward Loop
        for memnodes in self._pooled_memnodes:
            temp_out = memnodes.forward(temp_out , reduction=reduction)
            
            total_mem += 2 * MemNode._to( memnodes._out_mem , reduction= reduction)
            if hasattr(memnodes, '_param_mem'):
                total_mem += 2 * MemNode._to(memnodes._param_mem, reduction= reduction)
    
        print('--------------------------------SUMMARY---------------------------------')
        print(f'Total memory usage per batch_size {input_shape} for dtype {input_dtype} : {total_mem:.2f} {reduction}')
        
        
        return total_mem



