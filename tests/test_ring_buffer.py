import pytest
import torch
from torchutils.skeletons import Register, RingBuffer


@pytest.fixture()
def empty_buffer():
    return RingBuffer(max_persistence=5)


@pytest.fixture()
def full_buffer():
    buffer = RingBuffer(max_persistence=3)
    buffer.enqueue(
        Register(
            metrics="accuracy", loss=torch.nn.CrossEntropyLoss(), epoch=20, cycle=1
        )
    )
    buffer.enqueue(
        Register(metrics="precision", loss=torch.nn.MSELoss(), epoch=30, cycle=2)
    )
    buffer.enqueue(
        Register(metrics="recall", loss=torch.nn.L1Loss(), epoch=40, cycle=3)
    )
    return buffer


@pytest.mark.buffer()
def test_enqueue(empty_buffer):
    register = Register(
        metrics="accuracy", loss=torch.nn.CrossEntropyLoss(), epoch=20, cycle=1
    )
    assert empty_buffer.is_empty()
    empty_buffer.enqueue(register)
    assert not empty_buffer.is_empty()
    assert empty_buffer.size == 1
    assert empty_buffer[0].metrics[-1] == "accuracy"


@pytest.mark.buffer()
def test_dequeue(full_buffer):
    assert not full_buffer.is_empty()
    register = full_buffer.dequeue()
    assert full_buffer.size == 2
    assert register.metrics[-1] == "accuracy"
    assert full_buffer[0].metrics[-1] == "precision"


@pytest.mark.buffer()
def test_clear(full_buffer):
    assert not full_buffer.is_empty()
    full_buffer.clear()
    assert full_buffer.is_empty()
    assert full_buffer.size == 0


@pytest.mark.buffer()
def test_is_full(full_buffer):
    assert full_buffer.is_full()
    full_buffer.dequeue()
    assert not full_buffer.is_full()


@pytest.mark.buffer()
def test_peek(full_buffer):
    register = full_buffer.peek
    assert full_buffer.size == 3
    assert register.metrics[-1] == "recall"


@pytest.mark.buffer()
def test_max_persistence(empty_buffer):
    empty_buffer.max_persistence = 3
    assert empty_buffer.max_persistence == 3
    empty_buffer.max_persistence = 1
    assert empty_buffer.size == 0
    empty_buffer.enqueue(
        Register(
            metrics="accuracy", loss=torch.nn.CrossEntropyLoss(), epoch=20, cycle=1
        )
    )
    empty_buffer.enqueue(
        Register(metrics="precision", loss=torch.nn.MSELoss(), epoch=30, cycle=2)
    )
    empty_buffer.enqueue(
        Register(metrics="recall", loss=torch.nn.L1Loss(), epoch=40, cycle=3)
    )
    assert empty_buffer.size == 1
    assert empty_buffer[0].metrics[-1] == "recall"


@pytest.mark.buffer()
def test_invalid_enqueue(empty_buffer):
    register1 = Register(
        metrics="accuracy", loss=torch.nn.CrossEntropyLoss(), epoch=20, cycle=1
    )
    register2 = Register(metrics="accuracy", loss=torch.nn.MSELoss(), epoch=30, cycle=1)
    empty_buffer.enqueue(register1)
    with pytest.raises(AssertionError):
        empty_buffer.enqueue(register2)


@pytest.mark.buffer()
def test_invalid_max_persistence(empty_buffer):
    with pytest.raises(AssertionError):
        empty_buffer.max_persistence = 0
