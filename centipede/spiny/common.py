from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generator

__all__ = ["Box", "Impossible", "Ordering", "Unit", "compare", "compare_lex"]


class Impossible(Exception):
    pass


@dataclass
class Box[T]:
    value: T


@dataclass(frozen=True)
class Unit:
    @staticmethod
    def instance() -> Unit:
        return _UNIT


_UNIT = Unit()


class Ordering(Enum):
    Lt = -1
    Eq = 0
    Gt = 1


def compare[T](a: T, b: T) -> Ordering:
    # Unsafe eq/lt because generic protocols are half-baked
    if getattr(a, "__eq__")(b):
        return Ordering.Eq
    elif getattr(a, "__lt__")(b):
        return Ordering.Lt
    else:
        return Ordering.Gt


def compare_lex[T](agen: Generator[T], bgen: Generator[T]) -> Ordering:
    while True:
        try:
            a = next(agen)
        except StopIteration:
            try:
                _ = next(bgen)
                return Ordering.Lt
            except StopIteration:
                return Ordering.Eq
        try:
            b = next(bgen)
            r = compare(a, b)
            if r != Ordering.Eq:
                return r
        except StopIteration:
            return Ordering.Gt
