from fractions import Fraction

from centipede.minipat.arc import Arc
from centipede.minipat.ev import Ev


def test_ev_creation():
    """Test basic Ev creation."""
    arc = Arc(Fraction(0), Fraction(1))
    ev = Ev(arc, "test")
    assert ev.arc == arc
    assert ev.val == "test"


def test_ev_shift():
    """Test Ev shift operation."""
    arc = Arc(Fraction(1), Fraction(2))
    ev = Ev(arc, "test")
    shifted = ev.shift(Fraction(1))
    assert shifted.arc.start == Fraction(2)
    assert shifted.arc.end == Fraction(3)
    assert shifted.val == "test"


def test_ev_scale():
    """Test Ev scale operation."""
    arc = Arc(Fraction(1), Fraction(2))
    ev = Ev(arc, "test")
    scaled = ev.scale(Fraction(2))
    assert scaled.arc.start == Fraction(2)
    assert scaled.arc.end == Fraction(4)
    assert scaled.val == "test"


def test_ev_clip():
    """Test Ev clip operation."""
    arc = Arc(Fraction(0), Fraction(4))
    ev = Ev(arc, "test")
    clipped = ev.clip(Fraction(1, 2))
    assert clipped.arc.start == Fraction(0)
    assert clipped.arc.end == Fraction(2)
    assert clipped.val == "test"
