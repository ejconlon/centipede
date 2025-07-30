from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import override

from centipede.minipat.arc import Arc
from centipede.minipat.ev import Ev
from centipede.minipat.pat import Pat
from centipede.spiny.heapmap import PHeapMap


# sealed
class Stream[T](metaclass=ABCMeta):
    """A stream of events in time."""

    @abstractmethod
    def unstream(self, arc: Arc) -> PHeapMap[Arc, Ev[T]]:
        """Emit all events that start or end in the given arc."""
        ...


@dataclass(frozen=True)
class PatStream[T](Stream[T]):
    pat: Pat[T]

    @override
    def unstream(self, arc: Arc) -> PHeapMap[Arc, Ev[T]]:
        raise Exception("TODO")
