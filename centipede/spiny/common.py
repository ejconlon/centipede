from typing import Protocol


class Irrelevant(NotImplementedError):
    pass


class Impossible(Exception):
    pass


class Todo(Exception):
    pass


class Comparable(Protocol):
    def __lt__[S](self: S, other: S) -> bool:
        raise Irrelevant

    def __le__[S](self: S, other: S) -> bool:
        raise Irrelevant

    def __gt__[S](self: S, other: S) -> bool:
        raise Irrelevant

    def __ge__[S](self: S, other: S) -> bool:
        raise Irrelevant
