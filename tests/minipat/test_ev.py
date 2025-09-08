from fractions import Fraction

from minipat.arc import CycleArc, Span
from minipat.common import CycleDelta, CycleTime
from minipat.ev import Ev


def test_ev_creation() -> None:
    """Test basic Ev creation."""
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    span = Span(active=arc, whole=None)
    ev = Ev(span, "test")
    assert ev.span == span
    assert ev.val == "test"


def test_ev_shift() -> None:
    """Test Ev shift operation."""
    arc = CycleArc(CycleTime(Fraction(1)), CycleTime(Fraction(2)))
    span = Span(active=arc, whole=None)
    ev = Ev(span, "test")
    shifted = ev.shift(CycleDelta(Fraction(1)))
    assert shifted.span.active.start == Fraction(2)
    assert shifted.span.active.end == Fraction(3)
    assert shifted.val == "test"


def test_ev_scale() -> None:
    """Test Ev scale operation."""
    arc = CycleArc(CycleTime(Fraction(1)), CycleTime(Fraction(2)))
    span = Span(active=arc, whole=None)
    ev = Ev(span, "test")
    scaled = ev.scale(Fraction(2))
    assert scaled.span.active.start == Fraction(2)
    assert scaled.span.active.end == Fraction(4)
    assert scaled.val == "test"


def test_ev_clip() -> None:
    """Test Ev clip operation."""
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(4)))
    span = Span(active=arc, whole=None)
    ev = Ev(span, "test")
    clipped = ev.clip(Fraction(1, 2))
    assert clipped.span.active.start == Fraction(0)
    assert clipped.span.active.end == Fraction(2)
    assert clipped.val == "test"
