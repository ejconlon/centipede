from fractions import Fraction

from minipat.arc import Arc
from minipat.common import CycleDelta, CycleTime


def test_arc_creation():
    """Test basic Arc creation."""
    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    assert arc.start == Fraction(0)
    assert arc.end == Fraction(1)


def test_arc_empty():
    """Test empty Arc creation."""
    empty = Arc.empty()
    assert empty.null()
    assert empty.start == Fraction(0)
    assert empty.end == Fraction(0)


def test_arc_cycle():
    """Test Arc cycle creation."""
    cycle_arc = Arc.cycle(2)
    assert cycle_arc.start == Fraction(2)
    assert cycle_arc.end == Fraction(3)


def test_arc_length():
    """Test Arc length calculation."""
    arc = Arc(CycleTime(Fraction(1)), CycleTime(Fraction(4)))
    assert arc.length() == Fraction(3)


def test_arc_null():
    """Test Arc null detection."""
    # Normal arc should not be null
    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    assert not arc.null()

    # Arc with start >= end should be null
    null_arc = Arc(CycleTime(Fraction(1)), CycleTime(Fraction(1)))
    assert null_arc.null()

    null_arc2 = Arc(CycleTime(Fraction(2)), CycleTime(Fraction(1)))
    assert null_arc2.null()


def test_arc_normalize():
    """Test Arc normalization."""
    # Normal arc should remain unchanged
    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    normalized = arc._normalize()
    assert normalized == arc

    # Invalid arc should become empty
    invalid_arc = Arc(CycleTime(Fraction(2)), CycleTime(Fraction(1)))
    normalized = invalid_arc._normalize()
    assert normalized.null()

    # Special case: (0, 0) should remain as is
    zero_arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(0)))
    normalized = zero_arc._normalize()
    assert normalized == zero_arc


def test_arc_union():
    """Test Arc union operation."""
    arc1 = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(2)))
    arc2 = Arc(CycleTime(Fraction(1)), CycleTime(Fraction(3)))
    union = arc1.union(arc2)
    assert union.start == Fraction(0)
    assert union.end == Fraction(3)

    # Union with empty arc
    empty = Arc.empty()
    union_with_empty = arc1.union(empty)
    assert union_with_empty == arc1

    # Union of non-overlapping arcs
    arc3 = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    arc4 = Arc(CycleTime(Fraction(2)), CycleTime(Fraction(3)))
    union_non_overlap = arc3.union(arc4)
    assert union_non_overlap.start == Fraction(0)
    assert union_non_overlap.end == Fraction(3)


def test_arc_intersect():
    """Test Arc intersection operation."""
    arc1 = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(2)))
    arc2 = Arc(CycleTime(Fraction(1)), CycleTime(Fraction(3)))
    intersection = arc1.intersect(arc2)
    assert intersection.start == Fraction(1)
    assert intersection.end == Fraction(2)

    # Intersection with empty arc
    empty = Arc.empty()
    intersection_with_empty = arc1.intersect(empty)
    assert intersection_with_empty.null()

    # Non-overlapping intersection
    arc3 = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    arc4 = Arc(CycleTime(Fraction(2)), CycleTime(Fraction(3)))
    no_intersection = arc3.intersect(arc4)
    assert no_intersection.null()


def test_arc_shift():
    """Test Arc shift operation."""
    arc = Arc(CycleTime(Fraction(1)), CycleTime(Fraction(3)))
    shifted = arc.shift(CycleDelta(Fraction(2)))
    assert shifted.start == Fraction(3)
    assert shifted.end == Fraction(5)

    # Shift by zero should return normalized arc
    no_shift = arc.shift(CycleDelta(Fraction(0)))
    assert no_shift == arc

    # Shift empty arc
    empty = Arc.empty()
    shifted_empty = empty.shift(CycleDelta(Fraction(1)))
    assert shifted_empty.null()


def test_arc_scale():
    """Test Arc scale operation."""
    arc = Arc(CycleTime(Fraction(1)), CycleTime(Fraction(3)))
    scaled = arc.scale(Fraction(2))
    assert scaled.start == Fraction(2)
    assert scaled.end == Fraction(6)

    # Scale by 1 should return normalized arc
    no_scale = arc.scale(Fraction(1))
    assert no_scale == arc

    # Scale by 0 or negative should return empty
    zero_scale = arc.scale(Fraction(0))
    assert zero_scale.null()

    neg_scale = arc.scale(Fraction(-1))
    assert neg_scale.null()


def test_arc_clip():
    """Test Arc clip operation."""
    arc = Arc(CycleTime(Fraction(0)), CycleTime(Fraction(4)))
    clipped = arc.clip(Fraction(1, 2))  # Clip to half
    assert clipped.start == Fraction(0)
    assert clipped.end == Fraction(2)

    # Clip by 1 should return normalized arc
    no_clip = arc.clip(Fraction(1))
    assert no_clip == arc

    # Clip by 0 or negative should return empty
    zero_clip = arc.clip(Fraction(0))
    assert zero_clip.null()


def test_arc_split_cycles():
    """Test Arc split_cycles operation."""
    arc = Arc(CycleTime(Fraction(1, 2)), CycleTime(Fraction(5, 2)))  # 0.5 to 2.5
    cycles = list(arc.split_cycles())

    # Should split into cycles 0, 1, 2
    assert len(cycles) == 3

    cycle0, arc0 = cycles[0]
    assert cycle0 == 0
    assert arc0.start == Fraction(1, 2)
    assert arc0.end == Fraction(1)

    cycle1, arc1 = cycles[1]
    assert cycle1 == 1
    assert arc1.start == Fraction(1)
    assert arc1.end == Fraction(2)

    cycle2, arc2 = cycles[2]
    assert cycle2 == 2
    assert arc2.start == Fraction(2)
    assert arc2.end == Fraction(5, 2)


def test_arc_split_cycles_with_bounds():
    """Test Arc split_cycles with bounds."""
    arc = Arc(CycleTime(Fraction(1, 2)), CycleTime(Fraction(5, 2)))
    bounds = Arc(CycleTime(Fraction(3, 4)), CycleTime(Fraction(7, 4)))  # 0.75 to 1.75
    cycles = list(arc.split_cycles(bounds))

    # Bounds eliminate cycle 2 compared to unbounded version
    assert len(cycles) == 2

    cycle0, arc0 = cycles[0]
    assert cycle0 == 0
    assert arc0.start == Fraction(1, 2)  # max(0, 1/2) = 1/2
    assert arc0.end == Fraction(1)

    cycle1, arc1 = cycles[1]
    assert cycle1 == 1
    assert arc1.start == Fraction(1)
    assert arc1.end == Fraction(2)  # min(2, 5/2) = 2


def test_arc_union_all():
    """Test Arc.union_all static method."""
    arcs = [
        Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1))),
        Arc(CycleTime(Fraction(2)), CycleTime(Fraction(3))),
        Arc(CycleTime(Fraction(1)), CycleTime(Fraction(2))),
    ]
    union = Arc.union_all(arcs)
    assert union.start == Fraction(0)
    assert union.end == Fraction(3)

    # Empty list should return empty arc
    empty_union = Arc.union_all([])
    assert empty_union.null()


def test_arc_intersect_all():
    """Test Arc.intersect_all static method."""
    arcs = [
        Arc(CycleTime(Fraction(0)), CycleTime(Fraction(3))),
        Arc(CycleTime(Fraction(1)), CycleTime(Fraction(4))),
        Arc(CycleTime(Fraction(2)), CycleTime(Fraction(5))),
    ]
    intersection = Arc.intersect_all(arcs)
    assert intersection.start == Fraction(2)
    assert intersection.end == Fraction(3)

    # Non-overlapping arcs should return empty
    non_overlap = [
        Arc(CycleTime(Fraction(0)), CycleTime(Fraction(1))),
        Arc(CycleTime(Fraction(2)), CycleTime(Fraction(3))),
    ]
    no_intersection = Arc.intersect_all(non_overlap)
    assert no_intersection.null()
