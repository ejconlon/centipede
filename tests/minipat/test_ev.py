from fractions import Fraction

from minipat.ev import Ev
from minipat.time import CycleArc, CycleDelta, CycleSpan, CycleTime


def test_ev_creation() -> None:
    """Test basic Ev creation."""
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    span = CycleSpan(arc, None)
    ev = Ev(span, "test")
    assert ev.span == span
    assert ev.val == "test"


def test_ev_shift() -> None:
    """Test Ev shift operation."""
    arc = CycleArc(CycleTime(Fraction(1)), CycleTime(Fraction(2)))
    span = CycleSpan(arc, None)
    ev = Ev(span, "test")
    shifted = ev.shift(CycleDelta(Fraction(1)))
    assert shifted.span.active.start == Fraction(2)
    assert shifted.span.active.end == Fraction(3)
    assert shifted.val == "test"


def test_ev_scale() -> None:
    """Test Ev scale operation."""
    arc = CycleArc(CycleTime(Fraction(1)), CycleTime(Fraction(2)))
    span = CycleSpan(arc, None)
    ev = Ev(span, "test")
    scaled = ev.scale(Fraction(2))
    assert scaled.span.active.start == Fraction(2)
    assert scaled.span.active.end == Fraction(4)
    assert scaled.val == "test"


def test_ev_clip() -> None:
    """Test Ev clip operation."""
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(4)))
    span = CycleSpan(arc, None)
    ev = Ev(span, "test")
    clipped = ev.clip(Fraction(1, 2))
    assert clipped.span.active.start == Fraction(0)
    assert clipped.span.active.end == Fraction(2)
    assert clipped.val == "test"
