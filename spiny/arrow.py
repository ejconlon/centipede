"""Arrow and Isomorphism abstractions for functional transformations.

This module provides generalized abstractions for transformations between types,
inspired by concepts from category theory and functional programming:

- **Arrow[A, B]**: A one-directional transformation from type A to type B.
  Arrows compose naturally and support identity operations, making them useful
  for building complex data pipelines and transformations.

- **Iso[A, B]**: A bidirectional transformation (isomorphism) between types A and B.
  Isomorphisms represent lossless conversions where you can transform back and forth
  between two representations without losing information.

- **ArrowM[A, B, FB]**: Monadic arrows (Kleisli arrows) that produce results in a
  monadic context. These enable composition of computations with effects like
  optionality, multiple results, or state.

- **SeqArrowM[A, B]**: A specific monadic arrow for the sequence (list) monad,
  enabling non-deterministic computations where each step can produce multiple results.

These abstractions generalize patterns found throughout the codebase:
- PatBinder in minipat.pat follows the ArrowM pattern
- Element parsers in minipat.combinators use the Iso pattern

Examples:
    Basic arrow composition::

        >>> from spiny.arrow import Arrow
        >>> double = Arrow.function(lambda x: x * 2)
        >>> add_one = Arrow.function(lambda x: x + 1)
        >>> pipeline = double.and_then(add_one)
        >>> pipeline.apply(5)  # (5 * 2) + 1 = 11
        11

    Isomorphic transformations::

        >>> from spiny.arrow import Iso
        >>> binary_decimal = Iso.functions(
        ...     lambda n: bin(n)[2:],  # decimal to binary string
        ...     lambda s: int(s, 2)    # binary string to decimal
        ... )
        >>> binary_decimal.forward(10)
        '1010'
        >>> binary_decimal.backward('1010')
        10

    Monadic computations with SeqArrowM::

        >>> from spiny.arrow import SeqArrowM
        >>> from spiny import PSeq
        >>> class SplitArrow(SeqArrowM):
        ...     def apply(self, x):
        ...         return PSeq.mk([x, x + 10])  # produce two results
        >>> arrow = SplitArrow()
        >>> list(arrow.apply(5))
        [5, 15]
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
        >>> double_then_add = double.and_then(add_one)
        >>> double_then_add.apply(3)  # Returns 7
    """

    @abstractmethod
    def apply(self, value: A) -> B:
        """Apply the arrow transformation to a value.

        Args:
            value: The input value of type A.

        Returns:
            The transformed value of type B.
        """
        raise NotImplementedError()

    @staticmethod
    def identity() -> Arrow[A, A]:
        """Create an identity arrow that returns its input unchanged.

        The identity arrow is the neutral element for arrow composition.
        Composing any arrow with identity (on either side) returns the
        original arrow.

        Returns:
            An identity arrow that maps any value to itself.

        Example:
            >>> id_arrow = Arrow.identity()
            >>> id_arrow.apply(42)
            42
        """
        return IdArrow()

    @staticmethod
    def function(app: Callable[[A], B]) -> Arrow[A, B]:
        """Create an arrow from a regular Python function.

        Args:
            app: A function that transforms values from type A to type B.

        Returns:
            An arrow wrapping the given function.

        Example:
            >>> square = Arrow.function(lambda x: x ** 2)
            >>> square.apply(4)
            16
        """
        return FnArrow(app)

    def and_then[C](self, other: Arrow[B, C]) -> Arrow[A, C]:
        """Compose this arrow with another arrow sequentially.

        Creates a new arrow that first applies this arrow's transformation,
        then applies the other arrow's transformation to the result.

        Args:
            other: An arrow that transforms from type B to type C.

        Returns:
            A composed arrow that transforms from type A to type C.

        Example:
            >>> double = Arrow.function(lambda x: x * 2)
            >>> add_ten = Arrow.function(lambda x: x + 10)
            >>> pipeline = double.and_then(add_ten)
            >>> pipeline.apply(3)  # (3 * 2) + 10 = 16
            16
        """
        return ChainArrow.link(self, other)

    def iso_forward(self, other: Arrow[B, A]) -> Iso[A, B]:
        """Create an isomorphism using this arrow as the forward direction.

        Args:
            other: An arrow for the backward direction (B -> A).

        Returns:
            An isomorphism with this arrow as forward and other as backward.

        Example:
            >>> encode = Arrow.function(lambda x: x * 2)
            >>> decode = Arrow.function(lambda x: x / 2)
            >>> iso = encode.iso_forward(decode)
            >>> iso.forward(5)
            10
            >>> iso.backward(10)
            5.0
        """
        return FnIso(self.apply, other.apply)

    def iso_backward(self, other: Arrow[B, A]) -> Iso[B, A]:
        """Create an isomorphism using this arrow as the backward direction.

        Args:
            other: An arrow for the forward direction (B -> A).

        Returns:
            An isomorphism with other as forward and this arrow as backward.
        """
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
        """Apply the forward transformation.

        Args:
            value: The input value of type A.

        Returns:
            The transformed value of type B.
        """
        raise NotImplementedError()

    @abstractmethod
    def backward(self, value: B) -> A:
        """Apply the backward transformation.

        Args:
            value: The input value of type B.

        Returns:
            The transformed value of type A.
        """
        raise NotImplementedError()

    @staticmethod
    def identity() -> Iso[A, A]:
        """Create an identity isomorphism that returns its input unchanged.

        The identity isomorphism is the neutral element for isomorphism composition.
        Both forward and backward transformations return the input unchanged.

        Returns:
            An identity isomorphism that maps any value to itself.

        Example:
            >>> id_iso = Iso.identity()
            >>> id_iso.forward(42)
            42
            >>> id_iso.backward(42)
            42
        """
        return IdIso()

    @staticmethod
    def functions(fwd: Callable[[A], B], bwd: Callable[[B], A]) -> Iso[A, B]:
        """Create an isomorphism from a pair of functions.

        The functions should be inverses of each other for a true isomorphism,
        meaning that `bwd(fwd(x)) == x` and `fwd(bwd(y)) == y`.

        Args:
            fwd: Function for the forward transformation (A -> B).
            bwd: Function for the backward transformation (B -> A).

        Returns:
            An isomorphism using the provided functions.

        Example:
            >>> km_miles = Iso.functions(
            ...     lambda km: km * 0.621371,  # km to miles
            ...     lambda mi: mi * 1.60934    # miles to km
            ... )
            >>> km_miles.forward(100)  # 100 km to miles
            62.1371
            >>> km_miles.backward(62.1371)  # back to km
            100.0
        """
        return FnIso(fwd, bwd)

    def and_then[C](self, other: Iso[B, C]) -> Iso[A, C]:
        """Compose this isomorphism with another isomorphism.

        Creates a new isomorphism that applies this transformation first,
        then the other transformation. The backward direction applies
        the transformations in reverse order.

        Args:
            other: An isomorphism from type B to type C.

        Returns:
            A composed isomorphism from type A to type C.

        Example:
            >>> celsius_kelvin = Iso.functions(
            ...     lambda c: c + 273.15,
            ...     lambda k: k - 273.15
            ... )
            >>> kelvin_fahrenheit = Iso.functions(
            ...     lambda k: (k - 273.15) * 9/5 + 32,
            ...     lambda f: (f - 32) * 5/9 + 273.15
            ... )
            >>> celsius_fahrenheit = celsius_kelvin.and_then(kelvin_fahrenheit)
            >>> celsius_fahrenheit.forward(0)  # 0°C to °F
            32.0
            >>> celsius_fahrenheit.backward(32)  # 32°F to °C
            0.0
        """
        return ChainIso.link(self, other)

    def arrow_forward(self) -> Arrow[A, B]:
        """Extract the forward transformation as an arrow.

        Returns:
            An arrow that applies only the forward transformation.

        Example:
            >>> iso = Iso.functions(lambda x: x * 2, lambda x: x / 2)
            >>> forward_arrow = iso.arrow_forward()
            >>> forward_arrow.apply(5)
            10
        """
        return IsoForwardArrow(self)

    def arrow_backward(self) -> Arrow[B, A]:
        """Extract the backward transformation as an arrow.

        Returns:
            An arrow that applies only the backward transformation.

        Example:
            >>> iso = Iso.functions(lambda x: x * 2, lambda x: x / 2)
            >>> backward_arrow = iso.arrow_backward()
            >>> backward_arrow.apply(10)
            5.0
        """
        return IsoBackwardArrow(self)


class FnArrow[A, B](Arrow[A, B]):
    """Arrow implementation wrapping a regular Python function."""

    def __init__(self, fn: Callable[[A], B]) -> None:
        """Initialize with a function.

        Args:
            fn: The function to wrap as an arrow.
        """
        self._fn = fn

    @override
    def apply(self, value: A) -> B:
        """Apply the wrapped function."""
        return self._fn(value)


class IdArrow[A](Arrow[A, A], Singleton):
    """Identity arrow that returns its input unchanged.

    This is the neutral element for arrow composition.
    Composing any arrow with IdArrow yields the original arrow.
    """

    @override
    def apply(self, value: A) -> A:
        """Return the input value unchanged."""
        return value


class ChainArrow[A, C](Arrow[A, C]):
    """Composed arrow created by chaining multiple arrows together.

    Optimizes composition by flattening nested chains and eliminating
    identity arrows.
    """

    def __init__(self, chain: PSeq[Arrow[Any, Any]]) -> None:
        """Initialize with a sequence of arrows to compose."""
        self._chain = chain

    @staticmethod
    def link[B](a1: Arrow[A, B], a2: Arrow[B, C]) -> Arrow[A, C]:
        """Link two arrows together with optimization.

        Handles special cases:
        - Identity arrows are eliminated
        - Nested chains are flattened

        Args:
            a1: First arrow (A -> B).
            a2: Second arrow (B -> C).

        Returns:
            Optimized composed arrow (A -> C).
        """
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
    """Isomorphism implementation using a pair of functions."""

    def __init__(self, fwd: Callable[[A], B], bwd: Callable[[B], A]) -> None:
        """Initialize with forward and backward functions.

        Args:
            fwd: Function for forward transformation (A -> B).
            bwd: Function for backward transformation (B -> A).
        """
        self._fwd = fwd
        self._bwd = bwd

    @override
    def forward(self, value: A) -> B:
        return self._fwd(value)

    @override
    def backward(self, value: B) -> A:
        return self._bwd(value)


class IdIso[A](Iso[A, A], Singleton):
    """Identity isomorphism that returns its input unchanged.

    This is the neutral element for isomorphism composition.
    Both forward and backward transformations are the identity function.
    """

    @override
    def forward(self, value: A) -> A:
        """Return the input unchanged."""
        return value

    @override
    def backward(self, value: A) -> A:
        """Return the input unchanged."""
        return value


class ChainIso[A, C](Iso[A, C]):
    """Composed isomorphism created by chaining multiple isomorphisms.

    Forward transformation applies isomorphisms left to right.
    Backward transformation applies them right to left.
    """

    def __init__(self, chain: PSeq[Iso[Any, Any]]) -> None:
        """Initialize with a sequence of isomorphisms to compose."""
        self._chain = chain

    @staticmethod
    def link[B](i1: Iso[A, B], i2: Iso[B, C]) -> Iso[A, C]:
        """Link two isomorphisms together with optimization.

        Handles special cases:
        - Identity isomorphisms are eliminated
        - Nested chains are flattened

        Args:
            i1: First isomorphism (A <-> B).
            i2: Second isomorphism (B <-> C).

        Returns:
            Optimized composed isomorphism (A <-> C).
        """
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
    """Arrow adapter that applies only the forward direction of an isomorphism."""

    def __init__(self, iso: Iso[A, B]) -> None:
        """Initialize with an isomorphism."""
        self._iso = iso

    @override
    def apply(self, value: A) -> B:
        """Apply the forward transformation of the isomorphism."""
        return self._iso.forward(value)


class IsoBackwardArrow[A, B](Arrow[B, A]):
    """Arrow adapter that applies only the backward direction of an isomorphism."""

    def __init__(self, iso: Iso[A, B]) -> None:
        """Initialize with an isomorphism."""
        self._iso = iso

    @override
    def apply(self, value: B) -> A:
        """Apply the backward transformation of the isomorphism."""
        return self._iso.backward(value)


class ArrowM[A, B, FB](metaclass=ABCMeta):
    """Abstract base class for monadic arrows (Kleisli arrows).

    ArrowM represents a computation that takes a value of type A and returns
    a value of type B wrapped in a monadic context F. This is a generalization
    of functions that captures effects like optionality, multiple results,
    or stateful computations.

    Monadic arrows are also known as Kleisli arrows in category theory.
    They enable composition of computations with effects, where each step
    can produce results in a monadic context (like Optional, List, or IO).

    The key insight is that monadic arrows compose naturally even when the
    output is wrapped in a context, enabling clean composition of effectful
    computations without nested monadic structures.

    Type Parameters:
        A: The input type.
        B: The unwrapped output type.
        FB: The wrapped output type F[B] where F is a monad (e.g., List[B], Optional[B]).

    Subclasses must implement:
        - apply: Transform input A to monadic result FB.
        - unsafe_bind: Define how to chain computations in the monadic context.
        - identity: Create an identity arrow for the monad.

    Example:
        A parser that might fail could be modeled as ArrowM[str, int, Optional[int]],
        taking a string and returning an optional integer.
    """

    @abstractmethod
    def apply(self, value: A) -> FB:
        """Apply the monadic arrow to produce a result in context.

        Args:
            value: The input value of type A.

        Returns:
            The result of type B wrapped in monadic context F.
        """
        raise NotImplementedError()

    @abstractmethod
    def unsafe_bind[FC](self, context: FB, fn: Callable[[B], FC]) -> FC:
        """Bind operation for the specific monad.

        This defines how to chain computations in the monadic context.
        This is unsafe because we can't actually tell the type system
        that the "F" in FB and FC is the same. Typically subclasses
        will define this and a well-typed bind wrapper, leaving apply
        abstract.

        Args:
            context: A value in the monadic context F[B]
            fn: A function from B to F[C]

        Returns:
            A value in the monadic context F[C]
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def pure(cls) -> ArrowM[A, A, Any]:
        """Create a pure arrow for this monad.

        The pure arrow wraps values in the monadic context without
        additional transformation.

        Returns:
            A pure monadic arrow.
        """
        raise NotImplementedError()

    def and_then[C, FC](self, other: ArrowM[B, C, FC]) -> ArrowM[A, C, FC]:
        """Compose this monadic arrow with another.

        This is the monadic equivalent of function composition. The result
        of this arrow (in context F[B]) is passed through the monad's bind
        operation to the next arrow.

        Args:
            other: A monadic arrow from B to C in context F.

        Returns:
            A composed monadic arrow from A to C in context F.

        Example:
            >>> # If we had Optional monadic arrows:
            >>> parse_int: ArrowM[str, int, Optional[int]]
            >>> double: ArrowM[int, int, Optional[int]]
            >>> parse_and_double = parse_int.and_then(double)
            >>> # This would parse a string and double it if successful
        """
        return ChainArrowM.link(self, other)


def _identity(result: Any) -> Any:
    """Identity function for monadic composition."""
    return result


def _compose(
    kleisli: ArrowM[Any, Any, Any], tail: Callable[[Any], Any], result: Any
) -> Any:
    """Compose a monadic arrow with a continuation using bind."""
    return kleisli.unsafe_bind(result, tail)


def _top(kleisli: ArrowM[Any, Any, Any], tail: Callable[[Any], Any], value: Any) -> Any:
    """Apply the first arrow in a chain and pass result to continuation."""
    result = kleisli.apply(value)
    return tail(result)


class ChainArrowM[A, C, FC](ArrowM[A, C, FC]):
    """Composed monadic arrow created from chaining multiple monadic arrows.

    Implements efficient composition of monadic computations by building
    a function that threads the monadic context through all arrows in
    the chain using bind operations.
    """

    def __init__(self, chain: PSeq[ArrowM[Any, Any, Any]]) -> None:
        """Initialize with a sequence of monadic arrows to compose."""
        self._chain = chain
        self._fn: Optional[Callable[[A], FC]] = None

    @classmethod
    @override
    def pure(cls) -> ArrowM[A, A, Any]:
        """Create a pure arrow by delegating to the first arrow in the chain."""
        # This shouldn't be called directly on ChainArrowM, but we need to implement it
        # Delegate to the actual monad type
        raise NotImplementedError("ChainArrowM.pure should not be called directly")

    @staticmethod
    def link[B, FB](k1: ArrowM[A, B, FB], k2: ArrowM[B, C, FC]) -> ArrowM[A, C, FC]:
        """Link two monadic arrows together with optimization.

        Flattens nested chains to avoid deep nesting.

        Args:
            k1: First monadic arrow (A -> F[B]).
            k2: Second monadic arrow (B -> F[C]).

        Returns:
            Composed monadic arrow (A -> F[C]).
        """
        chain: PSeq[ArrowM[Any, Any, Any]]
        if isinstance(k1, ChainArrowM):
            if isinstance(k2, ChainArrowM):
                chain = k1._chain.concat(k2._chain)
            else:
                chain = k1._chain.snoc(k2)
        elif isinstance(k2, ChainArrowM):
            chain = k2._chain.cons(k1)
        else:
            chain = PSeq.mk([k1, k2])
        return ChainArrowM(chain)

    def _call_fn(self, value: A) -> FC:
        if self._fn is None:
            tail = None
            for kleisli in reversed(self._chain):
                if tail is None:
                    tail = _identity
                else:
                    tail = partial(_compose, kleisli, tail)
            assert tail is not None
            self._fn = partial(_top, self._chain[0], tail)
        return self._fn(value)

    @override
    def apply(self, value: A) -> FC:
        return self._call_fn(value)

    @override
    def unsafe_bind[FD](self, context: FC, fn: Callable[[C], FD]) -> FD:
        # Use the bind from the last arrow in the chain
        x = self._chain.unsnoc()
        assert x is not None
        _, last_kleisli = x
        return last_kleisli.unsafe_bind(context, fn)


class SeqArrowM[A, B](ArrowM[A, B, PSeq[B]]):
    """Monadic arrow for the sequence (list) monad.

    SeqArrowM represents non-deterministic computations where each step
    can produce multiple results. This implements the list monad pattern,
    where bind (flatMap) applies a function to each element and flattens
    the results.

    This is useful for:
    - Non-deterministic computations
    - Search problems with multiple solutions
    - Generating combinations or permutations
    - Modeling computations with multiple possible outcomes

    Type Parameters:
        A: The input type.
        B: The output element type.

    The monadic context is always PSeq[B] (a sequence of B values).

    Example:
        >>> from spiny import PSeq
        >>> class Duplicate(SeqArrowM[int, int]):
        ...     def apply(self, x: int) -> PSeq[int]:
        ...         return PSeq.mk([x, x * 2, x * 3])
        >>> dup = Duplicate()
        >>> list(dup.apply(5))
        [5, 10, 15]
    """

    @classmethod
    @override
    def pure(cls) -> SeqArrowM[A, A]:
        """Create a pure arrow for the sequence monad."""
        return IdSeqArrowM()

    def unsafe_bind[FC](self, context: PSeq[B], fn: Callable[[B], FC]) -> FC:
        """Bind operation for the sequence monad.

        Applies the function to each element in the sequence and flattens
        the results into a single sequence.

        Args:
            context: A sequence of values.
            fn: A function that produces a sequence for each value.

        Returns:
            The flattened sequence of all results.
        """
        fn1 = cast(Callable[[B], PSeq[Any]], fn)
        acc: PSeq[Any] = PSeq.empty()
        for b in context:
            acc += fn1(b)
        return cast(FC, acc)

    def bind[C](self, context: PSeq[B], fn: Callable[[B], PSeq[C]]) -> PSeq[C]:
        """Type-safe bind operation for sequences.

        Args:
            context: A sequence of B values.
            fn: A function from B to a sequence of C values.

        Returns:
            A flattened sequence of C values.

        Example:
            >>> arrow = SeqArrowM()
            >>> context = PSeq.mk([1, 2, 3])
            >>> result = arrow.bind(context, lambda x: PSeq.mk([x, x * 10]))
            >>> list(result)
            [1, 10, 2, 20, 3, 30]
        """
        return self.unsafe_bind(context, fn)


class IdSeqArrowM[T](SeqArrowM[T, T]):
    """Identity arrow for the sequence monad."""

    @override
    def apply(self, value: T) -> PSeq[T]:
        """Wrap a value in a singleton sequence."""
        return PSeq.mk([value])
