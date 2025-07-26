from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol, TypeVar

__all__ = ["Comparable", "Impossible", "Unit"]


@dataclass(frozen=True)
class Unit:
    @staticmethod
    def instance() -> Unit:
        return _UNIT


_UNIT = Unit()


class Impossible(Exception):
    pass


_T = TypeVar("_T", contravariant=True)


class Comparable(Protocol[_T]):
    @abstractmethod
    def __lt__(self, other: _T) -> bool: ...
    @abstractmethod
    def __le__(self, other: _T) -> bool: ...
    @abstractmethod
    def __gt__(self, other: _T) -> bool: ...
    @abstractmethod
    def __ge__(self, other: _T) -> bool: ...
