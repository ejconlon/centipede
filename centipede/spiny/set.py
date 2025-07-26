""" Persistent set implementation based on weight-balanced trees """

from dataclasses import dataclass


#sealed
class Set[T]:
    pass


@dataclass(frozen=True)
class SetEmpty[T]:
    pass


@dataclass(frozen=True)
class SetBranch[T]:
    _size: int
    _left: Set[T]
    _value: T
    _right: Set[T]
