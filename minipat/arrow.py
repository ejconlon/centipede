"""Arrow and Isomorphism abstractions for functional transformations.

This module provides generalized abstractions for transformations between types:
- Arrow: A one-directional transformation from type A to type B
- Iso: A bidirectional transformation (isomorphism) between types A and B

These abstractions generalize patterns found throughout the codebase:
- PatBinder in minipat.pat follows the Arrow pattern
- ElemParser in minipat.combinators follows the Iso pattern
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Any, Callable, Optional, cast, override

from spiny import PSeq
from spiny.common import Singleton


class Arrow[A, B](metaclass=ABCMeta):
    """Abstract base class for one-directional transformations.

    An Arrow represents a computation that transforms values of type A into
    values of type B. This is a fundamental abstraction in functional programming
    that generalizes function application.

    Arrows support:
    - Identity: A neutral element that returns input unchanged
    - Composition: Chaining arrows together to create new transformations
    - Conversion to Iso: Combining with another arrow to form an isomorphism

    Type Parameters:
        A: The input type
        B: The output type

    Examples:
        >>> double = Arrow.function(lambda x: x * 2)
        >>> add_one = Arrow.function(lambda x: x + 1)
        >>> double_then_add = double.andThen(add_one)
        >>> double_then_add.apply(3)  # Returns 7
    """
    @abstractmethod
    def apply(self, value: A) -> B:
        raise NotImplementedError()

    @staticmethod
    def identity() -> Arrow[A, A]:
        return IdArrow()

    @staticmethod
    def function(app: Callable[[A], B]) -> Arrow[A, B]:
        return FnArrow(app)

    def andThen[C](self, other: Arrow[B, C]) -> Arrow[A, C]:
        return ChainArrow.link(self, other)

    def iso_forward(self, other: Arrow[B, A]) -> Iso[A, B]:
        return FnIso(self.apply, other.apply)

    def iso_backward(self, other: Arrow[B, A]) -> Iso[B, A]:
        return FnIso(other.apply, self.apply)


class Iso[A, B](metaclass=ABCMeta):
    """Abstract base class for bidirectional transformations (isomorphisms).

    An Iso represents a bidirectional transformation between types A and B,
    where the transformation can be applied in both directions. This means
    there exists both a forward transformation (A -> B) and a backward
    transformation (B -> A) that are inverses of each other.

    Isomorphisms are useful when you need to:
    - Parse and render data (e.g., string <-> structured data)
    - Convert between equivalent representations
    - Implement reversible encodings

    Isos support:
    - Identity: A neutral element that returns input unchanged
    - Composition: Chaining isos together to create new transformations
    - Conversion to Arrow: Extracting either direction as a one-way transformation

    Type Parameters:
        A: The first type in the isomorphism
        B: The second type in the isomorphism

    Examples:
        >>> celsius_fahrenheit = Iso.functions(
        ...     lambda c: c * 9/5 + 32,  # forward: Celsius to Fahrenheit
        ...     lambda f: (f - 32) * 5/9  # backward: Fahrenheit to Celsius
        ... )
        >>> celsius_fahrenheit.forward(0)  # Returns 32
        >>> celsius_fahrenheit.backward(32)  # Returns 0
    """
    @abstractmethod
    def forward(self, value: A) -> B:
        raise NotImplementedError()

    @abstractmethod
    def backward(self, value: B) -> A:
        raise NotImplementedError()

    @staticmethod
    def identity() -> Iso[A, A]:
        return IdIso()

    @staticmethod
    def functions(fwd: Callable[[A], B], bwd: Callable[[B], A]) -> Iso[A, B]:
        return FnIso(fwd, bwd)

    def andThen[C](self, other: Iso[B, C]) -> Iso[A, C]:
        return ChainIso.link(self, other)

    def arrow_forward(self) -> Arrow[A, B]:
        return IsoForwardArrow(self)

    def arrow_backward(self) -> Arrow[B, A]:
        return IsoBackwardArrow(self)


class FnArrow[A, B](Arrow[A, B]):
    def __init__(self, fn: Callable[[A], B]) -> None:
        self._fn = fn

    @override
    def apply(self, value: A) -> B:
        return self._fn(value)


class IdArrow[A](Arrow[A, A], Singleton):
    @override
    def apply(self, value: A) -> A:
        return value


class ChainArrow[A, C](Arrow[A, C]):
    def __init__(self, chain: PSeq[Arrow[Any, Any]]) -> None:
        self._chain = chain

    @staticmethod
    def link[B](a1: Arrow[A, B], a2: Arrow[B, C]) -> Arrow[A, C]:
        chain: PSeq[Arrow[Any, Any]]
        if isinstance(a1, IdArrow):
            return cast(Arrow[A, C], a2)
        elif isinstance(a2, IdArrow):
            return cast(Arrow[A, C], a1)
        elif isinstance(a1, ChainArrow):
            if isinstance(a2, ChainArrow):
                chain = a1._chain.concat(a2._chain)
            else:
                chain = a1._chain.snoc(a2)
        elif isinstance(a2, ChainArrow):
            chain = a2._chain.cons(a1)
        else:
            chain = PSeq.mk([a1, a2])
        return ChainArrow(chain)

    @override
    def apply(self, value: A) -> C:
        result: Any = value
        for arrow in self._chain:
            result = arrow.apply(result)
        return cast(C, result)


class FnIso[A, B](Iso[A, B]):
    def __init__(self, fwd: Callable[[A], B], bwd: Callable[[B], A]) -> None:
        self._fwd = fwd
        self._bwd = bwd

    @override
    def forward(self, value: A) -> B:
        return self._fwd(value)

    @override
    def backward(self, value: B) -> A:
        return self._bwd(value)


class IdIso[A](Iso[A, A], Singleton):
    @override
    def forward(self, value: A) -> A:
        return value

    @override
    def backward(self, value: A) -> A:
        return value


class ChainIso[A, C](Iso[A, C]):
    def __init__(self, chain: PSeq[Iso[Any, Any]]) -> None:
        self._chain = chain

    @staticmethod
    def link[B](i1: Iso[A, B], i2: Iso[B, C]) -> Iso[A, C]:
        chain: PSeq[Iso[Any, Any]]
        if isinstance(i1, IdIso):
            return cast(Iso[A, C], i2)
        elif isinstance(i2, IdIso):
            return cast(Iso[A, C], i1)
        elif isinstance(i1, ChainIso):
            if isinstance(i2, ChainIso):
                chain = i1._chain.concat(i2._chain)
            else:
                chain = i1._chain.snoc(i2)
        elif isinstance(i2, ChainIso):
            chain = i2._chain.cons(i1)
        else:
            chain = PSeq.mk([i1, i2])
        return ChainIso(chain)

    @override
    def forward(self, value: A) -> C:
        result: Any = value
        for iso in self._chain:
            result = iso.forward(result)
        return cast(C, result)

    @override
    def backward(self, value: C) -> A:
        result: Any = value
        for iso in reversed(list(self._chain)):
            result = iso.backward(result)
        return cast(A, result)


class IsoForwardArrow[A, B](Arrow[A, B]):
    def __init__(self, iso: Iso[A, B]) -> None:
        self._iso = iso

    @override
    def apply(self, value: A) -> B:
        return self._iso.forward(value)


class IsoBackwardArrow[A, B](Arrow[B, A]):
    def __init__(self, iso: Iso[A, B]) -> None:
        self._iso = iso

    @override
    def apply(self, value: B) -> A:
        return self._iso.backward(value)


class Kleisli[A, B, FB](metaclass=ABCMeta):
    """Abstract base class for Kleisli arrows (monadic computations).

    A Kleisli arrow represents a computation that takes a value of type A
    and returns a value of type B wrapped in a monadic context F. This is
    a generalization of functions that captures effects like optionality,
    multiple results, or stateful computations.

    The key insight is that Kleisli arrows compose naturally even when the
    output is wrapped in a context, enabling clean composition of effectful
    computations.

    Type Parameters:
        A: The input type
        B: The unwrapped output type
        FB: The wrapped output type F[B] (where F is a monad)

    The bind method must be implemented to specify how to chain computations
    in the specific monadic context.
    """
    @abstractmethod
    def apply(self, value: A) -> FB:
        """Apply the Kleisli arrow to produce a monadic result."""
        raise NotImplementedError()

    @abstractmethod
    def bind[FC](self, context: FB, fn: Callable[[B], FC]) -> FC:
        """Bind operation for the specific monad.

        This defines how to chain computations in the monadic context.

        Args:
            context: A value in the monadic context F[B]
            fn: A function from B to F[C]

        Returns:
            A value in the monadic context F[C]
        """
        raise NotImplementedError()

    def andThen[C, FC](self, other: Kleisli[B, C, FC]) -> Kleisli[A, C, FC]:
        """Compose two Kleisli arrows."""
        return ChainKleisli.link(self, other)


def _identity(result):
    return result


def _compose(kleisli, tail, result):
    return kleisli.bind(result, tail)


def _top(kleisli, tail, value):
    result = kleisli.apply(value)
    return tail(result)


class ChainKleisli[A, C, FC](Kleisli[A, C, FC]):
    """Composed Kleisli arrow created from chaining multiple arrows."""
    def __init__(self, chain: PSeq[Kleisli[Any, Any, Any]]) -> None:
        self._chain = chain
        self._fn: Optional[Callable[[A], FC]] = None

    @staticmethod
    def link[B, FB](k1: Kleisli[A, B, FB], k2: Kleisli[B, C, FC]) -> Kleisli[A, C, FC]:
        """Link two Kleisli arrows together, optimizing for identity arrows."""
        chain: PSeq[Kleisli[Any, Any, Any]]
        if isinstance(k1, ChainKleisli):
            if isinstance(k2, ChainKleisli):
                chain = k1._chain.concat(k2._chain)
            else:
                chain = k1._chain.snoc(k2)
        elif isinstance(k2, ChainKleisli):
            chain = k2._chain.cons(k1)
        else:
            chain = PSeq.mk([k1, k2])
        return ChainKleisli(chain)

    def _call_fn(self, value: A) -> FC:
        if self._fn is None:
            tail = None
            for kleisli in reversed(self._chain):
                if tail is None:
                    tail = _identity
                else:
                    tail = partial(_compose, kleisli, tail)
            self._fn = partial(_top, self._chain[0], tail)
        return self._fn(value)

    @override
    def apply(self, value: A) -> FC:
        return self._call_fn(value)

    @override
    def bind[FD](self, context: FC, fn: Callable[[C], FD]) -> FD:
        # Use the bind from the last arrow in the chain
        x = self._chain.unsnoc()
        assert x is not None
        _, last_kleisli = x
        return last_kleisli.bind(context, fn)
