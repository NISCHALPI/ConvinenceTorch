import pytest
import torch
from torchutils.skeletons import MemNode, MemPool


@pytest.mark.memory()
@pytest.mark.parametrize(
    ("base_model", "inp_shape", "out_shape"),
    argvalues=[
        (torch.nn.Linear(12, 12), (11, 12), (11, 12)),
        (torch.nn.Tanh(), (1, 1), (1, 1)),
        (torch.nn.Conv2d(3, 64, 3, padding=1), (1, 3, 256, 256), (1, 64, 256, 256)),
    ],
)
def test_MemNode(base_model: torch.nn.Module, inp_shape, out_shape):
    # Create an instance of MemNode
    mem_node = MemNode(base_model)

    # Generate random input tensor
    input_tensor = torch.randn(*inp_shape)

    assert issubclass(base_model.__class__, torch.nn.Module)

    # Test _calc_stats method
    mem_node._calc_stats(input_tensor)
    if len(list(base_model.parameters())) != 0:
        assert hasattr(mem_node, "_param_mem")
    assert hasattr(mem_node, "_out_mem")
    assert hasattr(mem_node, "_status_dict")

    # Test _to method
    size_in_bytes = 1024
    reduction = "MB"
    result = mem_node._to(size_in_bytes, reduction)
    assert result == pytest.approx(1 / 1024)

    # Test _prettyprint method
    reduction = "MB"
    mem_node._prettyprint(reduction)

    # Test forward method
    reduction = "MB"
    output = mem_node.forward(input_tensor, reduction)
    assert output.shape == out_shape
    assert output.grad is None


@pytest.mark.memory()
@pytest.mark.parametrize(
    ("model", "input_shape", "pool_length"),
    argvalues=[
        (torch.nn.Sequential(torch.nn.Linear(10, 10)), (10,), 1),
        (
            torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Flatten(),
                torch.nn.Linear(64 * 16 * 16, 10),
            ),
            (1, 3, 32, 32),
            6,
        ),
    ],
)
def test_MemPool(model: torch.nn.Module, input_shape, pool_length):
    # Create an instance of MemPool
    device = torch.device("cpu")
    mem_pool = MemPool(model, device)

    assert isinstance(model, torch.nn.Module)

    # Test _init_pool method
    assert mem_pool._node_queue.qsize() == pool_length

    # Test _calc_memusage method
    input_dtype = torch.float32
    reduction = "MB"
    total_mem = mem_pool.calc_memusage(
        input_shape, input_dtype=input_dtype, reduction=reduction
    )
    assert isinstance(total_mem, float)
