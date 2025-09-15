from __future__ import annotations

from typing import Callable

from spiny import PSeq
from spiny.arrow import (
    Arrow,
    ChainArrow,
    ChainArrowM,
    ChainIso,
    FnArrow,
    IdArrow,
    IdIso,
    Iso,
    IsoBackwardArrow,
    IsoForwardArrow,
    SeqArrowM,
)


class TestArrow:
    def test_identity(self) -> None:
        arrow: Arrow[int, int] = Arrow.identity()
        assert arrow.apply(42) == 42
        arrow_str: Arrow[str, str] = Arrow.identity()
        assert arrow_str.apply("test") == "test"
        assert isinstance(arrow, IdArrow)

    def test_function(self) -> None:
        double: Arrow[int, int] = Arrow.function(lambda x: x * 2)
        assert double.apply(5) == 10
        assert double.apply(3) == 6
        assert isinstance(double, FnArrow)

    def test_and_then(self) -> None:
        double: Arrow[int, int] = Arrow.function(lambda x: x * 2)
        add_one: Arrow[int, int] = Arrow.function(lambda x: x + 1)
        composed = double.and_then(add_one)

        assert composed.apply(3) == 7
        assert composed.apply(5) == 11
        assert isinstance(composed, ChainArrow)

    def test_and_then_identity_left(self) -> None:
        identity: Arrow[int, int] = Arrow.identity()
        add_one: Arrow[int, int] = Arrow.function(lambda x: x + 1)
        composed = identity.and_then(add_one)

        assert composed.apply(5) == 6
        assert composed is add_one

    def test_and_then_identity_right(self) -> None:
        add_one: Arrow[int, int] = Arrow.function(lambda x: x + 1)
        identity: Arrow[int, int] = Arrow.identity()
        composed = add_one.and_then(identity)

        assert composed.apply(5) == 6
        assert composed is add_one

    def test_chain_multiple(self) -> None:
        add_one: Arrow[int, int] = Arrow.function(lambda x: x + 1)
        double: Arrow[int, int] = Arrow.function(lambda x: x * 2)
        subtract_three: Arrow[int, int] = Arrow.function(lambda x: x - 3)

        composed = add_one.and_then(double).and_then(subtract_three)
        assert composed.apply(5) == 9

    def test_iso_forward(self) -> None:
        double: Arrow[float, float] = Arrow.function(lambda x: x * 2)
        halve: Arrow[float, float] = Arrow.function(lambda x: x / 2)
        iso = double.iso_forward(halve)

        assert iso.forward(4.0) == 8.0
        assert iso.backward(8.0) == 4.0

    def test_iso_backward(self) -> None:
        double: Arrow[float, float] = Arrow.function(lambda x: x * 2)
        halve: Arrow[float, float] = Arrow.function(lambda x: x / 2)
        iso = double.iso_backward(halve)

        assert iso.forward(8.0) == 4.0
        assert iso.backward(4.0) == 8.0


class TestIso:
    def test_identity(self) -> None:
        iso: Iso[int, int] = Iso.identity()
        assert iso.forward(42) == 42
        assert iso.backward(42) == 42
        assert isinstance(iso, IdIso)

    def test_functions(self) -> None:
        celsius_fahrenheit = Iso.functions(
            lambda c: c * 9 / 5 + 32,
            lambda f: (f - 32) * 5 / 9,
        )

        assert celsius_fahrenheit.forward(0) == 32
        assert celsius_fahrenheit.forward(100) == 212
        assert celsius_fahrenheit.backward(32) == 0
        assert celsius_fahrenheit.backward(212) == 100

    def test_and_then(self) -> None:
        double_halve: Iso[float, float] = Iso.functions(
            lambda x: x * 2, lambda x: x / 2
        )
        add_sub: Iso[float, float] = Iso.functions(lambda x: x + 10, lambda x: x - 10)
        composed = double_halve.and_then(add_sub)

        assert composed.forward(5.0) == 20.0
        assert composed.backward(20.0) == 5.0

    def test_and_then_identity_left(self) -> None:
        identity: Iso[float, float] = Iso.identity()
        double_halve: Iso[float, float] = Iso.functions(
            lambda x: x * 2, lambda x: x / 2
        )
        composed = identity.and_then(double_halve)

        assert composed.forward(5.0) == 10.0
        assert composed.backward(10.0) == 5.0
        assert composed is double_halve

    def test_and_then_identity_right(self) -> None:
        double_halve: Iso[float, float] = Iso.functions(
            lambda x: x * 2, lambda x: x / 2
        )
        identity: Iso[float, float] = Iso.identity()
        composed = double_halve.and_then(identity)

        assert composed.forward(5.0) == 10.0
        assert composed.backward(10.0) == 5.0
        assert composed is double_halve

    def test_chain_multiple(self) -> None:
        iso1: Iso[float, float] = Iso.functions(lambda x: x + 1, lambda x: x - 1)
        iso2: Iso[float, float] = Iso.functions(lambda x: x * 2, lambda x: x / 2)
        iso3: Iso[float, float] = Iso.functions(lambda x: x - 3, lambda x: x + 3)

        composed = iso1.and_then(iso2).and_then(iso3)
        assert composed.forward(5.0) == 9.0
        assert composed.backward(9.0) == 5.0

    def test_arrow_forward(self) -> None:
        iso: Iso[float, float] = Iso.functions(lambda x: x * 2, lambda x: x / 2)
        arrow = iso.arrow_forward()

        assert arrow.apply(5.0) == 10.0
        assert isinstance(arrow, IsoForwardArrow)

    def test_arrow_backward(self) -> None:
        iso: Iso[float, float] = Iso.functions(lambda x: x * 2, lambda x: x / 2)
        arrow = iso.arrow_backward()

        assert arrow.apply(10.0) == 5.0
        assert isinstance(arrow, IsoBackwardArrow)


class TestChainArrow:
    def test_link_both_chains(self) -> None:
        a1: Arrow[float, float] = Arrow.function(lambda x: x + 1)
        a2: Arrow[float, float] = Arrow.function(lambda x: x * 2)
        chain1 = ChainArrow.link(a1, a2)

        a3: Arrow[float, float] = Arrow.function(lambda x: x - 3)
        a4: Arrow[float, float] = Arrow.function(lambda x: x / 2)
        chain2 = ChainArrow.link(a3, a4)

        result = ChainArrow.link(chain1, chain2)
        assert result.apply(5.0) == 4.5

    def test_link_first_chain(self) -> None:
        a1: Arrow[int, int] = Arrow.function(lambda x: x + 1)
        a2: Arrow[int, int] = Arrow.function(lambda x: x * 2)
        chain = ChainArrow.link(a1, a2)

        a3: Arrow[int, int] = Arrow.function(lambda x: x - 3)
        result = ChainArrow.link(chain, a3)
        assert result.apply(5) == 9

    def test_link_second_chain(self) -> None:
        a1: Arrow[int, int] = Arrow.function(lambda x: x + 1)

        a2: Arrow[int, int] = Arrow.function(lambda x: x * 2)
        a3: Arrow[int, int] = Arrow.function(lambda x: x - 3)
        chain = ChainArrow.link(a2, a3)

        result = ChainArrow.link(a1, chain)
        assert result.apply(5) == 9


class TestChainIso:
    def test_link_both_chains(self) -> None:
        i1: Iso[float, float] = Iso.functions(lambda x: x + 1, lambda x: x - 1)
        i2: Iso[float, float] = Iso.functions(lambda x: x * 2, lambda x: x / 2)
        chain1: Iso[float, float] = ChainIso.link(i1, i2)

        i3: Iso[float, float] = Iso.functions(lambda x: x - 3, lambda x: x + 3)
        i4: Iso[float, float] = Iso.functions(lambda x: x / 2, lambda x: x * 2)
        chain2: Iso[float, float] = ChainIso.link(i3, i4)

        result = ChainIso.link(chain1, chain2)
        assert result.forward(5.0) == 4.5
        assert result.backward(4.5) == 5.0

    def test_link_first_chain(self) -> None:
        i1: Iso[float, float] = Iso.functions(lambda x: x + 1, lambda x: x - 1)
        i2: Iso[float, float] = Iso.functions(lambda x: x * 2, lambda x: x / 2)
        chain: Iso[float, float] = ChainIso.link(i1, i2)

        i3: Iso[float, float] = Iso.functions(lambda x: x - 3, lambda x: x + 3)
        result = ChainIso.link(chain, i3)
        assert result.forward(5.0) == 9.0
        assert result.backward(9.0) == 5.0

    def test_link_second_chain(self) -> None:
        i1: Iso[float, float] = Iso.functions(lambda x: x + 1, lambda x: x - 1)

        i2: Iso[float, float] = Iso.functions(lambda x: x * 2, lambda x: x / 2)
        i3: Iso[float, float] = Iso.functions(lambda x: x - 3, lambda x: x + 3)
        chain: Iso[float, float] = ChainIso.link(i2, i3)

        result = ChainIso.link(i1, chain)
        assert result.forward(5.0) == 9.0
        assert result.backward(9.0) == 5.0


class ConcreteSeqArrowM(SeqArrowM[int, int]):
    def __init__(self, fn: Callable[[int], PSeq[int]] | None = None) -> None:
        self._fn = fn or (lambda x: PSeq.mk([x]))

    def apply(self, value: int) -> PSeq[int]:
        return self._fn(value)


class TestSeqArrowM:
    def test_apply_single_result(self) -> None:
        # Test basic apply with single result
        arrow = ConcreteSeqArrowM(lambda x: PSeq.mk([x * 2]))
        result = arrow.apply(5)
        assert list(result) == [10]

    def test_apply_multiple_results(self) -> None:
        # Test apply that produces multiple results
        arrow = ConcreteSeqArrowM(lambda x: PSeq.mk([x, x + 1, x + 2]))
        result = arrow.apply(5)
        assert list(result) == [5, 6, 7]

    def test_bind(self) -> None:
        arrow = ConcreteSeqArrowM(lambda x: PSeq.mk([x, x + 1, x + 2]))
        context = PSeq.mk([1, 2, 3])
        result = arrow.bind(context, lambda x: PSeq.mk([x * 2, x * 3]))

        # For each element in context, the function produces 2 values
        # So we expect 3 * 2 = 6 values total
        expected = [2, 3, 4, 6, 6, 9]
        assert list(result) == expected

    def test_bind_empty_context(self) -> None:
        arrow = ConcreteSeqArrowM(lambda x: PSeq.mk([x * 2]))
        context: PSeq[int] = PSeq.empty()
        result = arrow.bind(context, lambda x: PSeq.mk([x + 1]))
        assert list(result) == []

    def test_bind_empty_result(self) -> None:
        arrow = ConcreteSeqArrowM(lambda x: PSeq.mk([x]))
        context = PSeq.mk([1, 2, 3])
        result: PSeq[int] = arrow.bind(context, lambda x: PSeq.empty())
        assert list(result) == []

    def test_unsafe_bind(self) -> None:
        arrow = ConcreteSeqArrowM(lambda x: PSeq.mk([x * 2]))
        context = PSeq.mk([1, 2, 3])
        # unsafe_bind should work the same as bind for SeqArrowM
        result = arrow.unsafe_bind(context, lambda x: PSeq.mk([x + 10]))
        assert list(result) == [11, 12, 13]

    def test_and_then(self) -> None:
        # Test that and_then creates a ChainArrowM
        double = ConcreteSeqArrowM(lambda x: PSeq.mk([x * 2]))
        add_one = ConcreteSeqArrowM(lambda x: PSeq.mk([x + 1]))
        composed = double.and_then(add_one)

        # The chain should be created properly
        assert isinstance(composed, ChainArrowM)
        assert len(composed._chain) == 2

    def test_monadic_behavior(self) -> None:
        # Test real monadic behavior - each step can produce multiple results
        # This simulates non-deterministic computation

        # First arrow: given x, produce x and x+10
        split = ConcreteSeqArrowM(lambda x: PSeq.mk([x, x + 10]))

        # When we apply split(5), we get [5, 15]
        # Then for each value, multiply function produces 2 values:
        # - 5 -> [10, 15]
        # - 15 -> [30, 45]
        # Total: [10, 15, 30, 45]

        result1 = split.apply(5)
        assert list(result1) == [5, 15]

        # Now test bind with the result of split
        # bind takes the context [5, 15] and applies the function to each
        result2 = split.bind(result1, lambda y: PSeq.mk([y * 2, y * 3]))
        assert list(result2) == [10, 15, 30, 45]

    def test_list_comprehension_analogy(self) -> None:
        # SeqArrowM is like list comprehensions in Haskell
        # [x * y | x <- [1,2,3], y <- [10,20]]

        arrow = ConcreteSeqArrowM(lambda x: PSeq.mk([x * 10, x * 20]))

        # Given context [1, 2, 3], apply function that produces two values for each
        context = PSeq.mk([1, 2, 3])
        result = arrow.bind(context, lambda x: PSeq.mk([x * 10, x * 20]))

        # 1 -> [10, 20]
        # 2 -> [20, 40]
        # 3 -> [30, 60]
        assert list(result) == [10, 20, 20, 40, 30, 60]


class AddArrowM(SeqArrowM[int, int]):
    def __init__(self, n: int) -> None:
        self.n = n

    def apply(self, value: int) -> PSeq[int]:
        return PSeq.mk([value + self.n])


class TestChainArrowM:
    def test_link(self) -> None:
        double = ConcreteSeqArrowM(lambda x: PSeq.mk([x * 2]))
        add_one = ConcreteSeqArrowM(lambda x: PSeq.mk([x + 1]))
        chain = ChainArrowM.link(double, add_one)

        # Test that the chain is created properly
        assert isinstance(chain, ChainArrowM)
        assert len(chain._chain) == 2
        assert chain._chain[0] is double
        assert chain._chain[1] is add_one

    def test_link_multiple(self) -> None:
        a1 = AddArrowM(1)
        a2 = AddArrowM(2)
        a3 = AddArrowM(3)

        chain = a1.and_then(a2).and_then(a3)
        # Test that the chain is created properly
        assert isinstance(chain, ChainArrowM)
        assert len(chain._chain) == 3

    def test_unsafe_bind(self) -> None:
        arrow1 = ConcreteSeqArrowM(lambda x: PSeq.mk([x + 1]))
        arrow2 = ConcreteSeqArrowM(lambda x: PSeq.mk([x + 1]))
        chain = ChainArrowM.link(arrow1, arrow2)

        context = PSeq.mk([10])
        result = chain.unsafe_bind(context, lambda x: PSeq.mk([x * 2]))
        assert list(result) == [20]
