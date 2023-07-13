"""Ring buffer to extend register persistence in the trainer over a cycle.

This module provides a RingBuffer class that implements a ring buffer data structure. The ring buffer is designed to store and manage a collection of Register objects, extending their persistence over a specified number of cycles.

The RingBuffer allows enqueueing new Register objects and automatically removes the oldest Register when the buffer reaches its maximum persistence limit. It also provides methods to check if the buffer is empty or full, access Register objects by index, retrieve a string representation of the buffer's content, and perform enqueue and dequeue operations.

Example usage:
    buffer = RingBuffer(max_persistence=5)
    buffer.enqueue(register1)
    buffer.enqueue(register2)
    ...
    buffer.enqueue(registerN)
    register = buffer[0]
    print(buffer)
    ...
"""

import typing as tp

from .base_register import Register

__all__ = ["RingBuffer"]


class RingBuffer:
    """Ring buffer to extend register persistence in the trainer over a cycle.

    This class implements a ring buffer data structure that is designed to store and manage a collection of Register objects, extending their persistence over a specified number of cycles.

    Args:
        max_persistence (int): Maximum number of cycles to persist the registers.

    Attributes:
        buffer (List[Register]): The buffer that stores the Register objects.

    Methods:
        is_empty(): Check if the buffer is empty.
        is_full(): Check if the buffer is full.
        enqueue(item: Register) -> Optional[Register]: Enqueue a new Register object into the buffer, removing the oldest Register if the buffer is full.
        clear(): Clear the buffer.
        dequeue() -> Register: Dequeue the oldest Register object from the buffer.
        __getitem__(index: int) -> Register: Get the Register object at the specified index.
        __repr__() -> str: Get a string representation of the buffer.
        peek: Get the most recent Register object in the buffer without removing it.
        size: Get the current size of the buffer.

    Example usage:
        buffer = RingBuffer(max_persistence=5)
        buffer.enqueue(register1)
        buffer.enqueue(register2)
        ...
        buffer.enqueue(registerN)
        register = buffer[0]
        print(buffer)
    """

    def __init__(self, max_persistence: int = 5) -> None:
        """Initialize the RingBuffer.

        Args:
            max_persistence (int): Maximum number of cycles to persist the registers.

        Returns:
            None
        """
        assert (
            max_persistence > 0
        ), f"Max persistence must be greater than 0 but got {max_persistence}"
        self._max_persistence = max_persistence
        self._buffer: tp.List[Register] = []
        # This is info for the trainer if Buffer is online for current cycle
        self._online = False

    def is_empty(self) -> bool:
        """Check if the buffer is empty.

        Returns:
            bool: True if the buffer is empty, False otherwise.
        """
        return len(self._buffer) == 0

    def is_full(self) -> bool:
        """Check if the buffer is full.

        Returns:
            bool: True if the buffer is full, False otherwise.
        """
        return len(self._buffer) == self._max_persistence

    def enqueue(self, item: Register) -> tp.Optional[Register]:
        """Enqueue a new Register object into the buffer, removing the oldest Register if the buffer is full.

        Args:
            item (Register): The Register object to enqueue.

        Returns:
            Optional[Register]: The oldest Register object that was removed from the buffer if it was full, None otherwise.
        """
        assert hasattr(
            item, "cycle"
        ), "This buffer is meant to be used with registers that have a 'cycle' attribute"

        for _registers in self.buffer:
            assert (
                item.cycle != _registers.cycle
            ), "Cannot enqueue Registers having same cycle"

        old_register = None

        if self.is_full():
            old_register = self._buffer.pop(0)

        # Add to buffer
        self._buffer.append(item)

        return old_register

    def clear(self) -> None:
        """Clear the buffer.

        Returns:
            None
        """
        self._buffer = []

    def dequeue(self) -> Register:
        """Dequeue the oldest Register object from the buffer.

        Returns:
            Register: The oldest Register object from the buffer.

        Raises:
            IndexError: If the buffer is empty.
        """
        if self.is_empty():
            raise IndexError("Cannot dequeue from an empty buffer.")
        return self._buffer.pop(0)

    def __getitem__(self, index: int) -> Register:
        """Get the Register object at the specified index.

        Args:
            index (int): The index of the Register object.

        Returns:
            Register: The Register object at the specified index.

        """
        return self._buffer[index]

    def __repr__(self) -> str:
        """Get a string representation of the buffer.

        Returns:
            str: A string representation of the buffer.

        """
        rep = f"Size: {len(self._buffer)}\n"
        if not self.is_empty():
            for idx, register in enumerate(reversed(self._buffer)):
                rep += f"{idx} " + register.__repr__() + "\n"

        return rep

    @property
    def peek(self) -> Register:
        """Get the most recent Register object in the buffer without removing it.

        Returns:
            Register: The most recent Register object in the buffer, or None if the buffer is empty.


        Raises:
        ------
        IndexError
            if the Buffer is empty
        """
        if self.is_empty():
            raise IndexError("Ring Buffer is empty cannot peek!")

        return self.buffer[-1]

    @property
    def size(self) -> int:
        """Get the current size of the buffer.

        Returns:
            int: The size of the buffer.
        """
        return len(self.buffer)

    @size.setter
    def size(self) -> tp.NoReturn:
        """Set the size of the buffer (not supported).

        Raises:
            AttributeError: This method is not supported.
        """
        raise AttributeError("Cannot set a buffer size")

    @property
    def buffer(self) -> tp.List[Register]:
        """Get the buffer that stores the Register objects.

        Returns:
            List[Register]: The buffer.
        """
        return self._buffer

    @buffer.setter
    def buffer(self) -> tp.NoReturn:
        """Set the buffer (not supported).

        Raises:
            AttributeError: This method is not supported. Use enqueue() instead.
        """
        raise AttributeError("Cannot set a buffer. Use enqueue!")

    @property
    def max_persistence(self) -> int:
        """Get the maximum number of cycles to persist the registers.

        Returns:
        int: The maximum number of cycles to persist the registers.
        """
        return self._max_persistence

    @max_persistence.setter
    def max_persistence(self, value: int) -> None:
        """Set the maximum number of cycles to persist the registers.

        Args:
        value (int): The new maximum number of cycles.

        Returns:
        None

        Raises:
        AssertionError: If the given value is not greater than 0.

        Notes:
        If the new maximum persistence is smaller than the current size of the buffer, the oldest registers will be dequeued until the buffer size is equal to the new maximum persistence.
        """
        assert value > 0, f"max_persistence cannot must be in [0,inf), but got {value}"

        while self.size > value:
            self.dequeue()

        self._max_persistence = value
