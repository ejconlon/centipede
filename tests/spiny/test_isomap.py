"""Tests for PIsoMap (persistent isomorphic map)."""

import pytest

from spiny.isomap import PIsoMap


class TestPIsoMap:
    """Test cases for PIsoMap functionality."""

    def test_empty_creation(self) -> None:
        """Test creating an empty PIsoMap."""
        revmap: PIsoMap[str, int] = PIsoMap.empty()
        assert revmap.is_empty()
        assert revmap.size() == 0
        assert len(revmap) == 0
        assert not revmap

    def test_mk_creation(self) -> None:
        """Test creating PIsoMap from pairs."""
        pairs = [("a", 1), ("b", 2), ("c", 3)]
        revmap = PIsoMap.mk(pairs)

        assert revmap.size() == 3
        assert len(revmap) == 3
        assert revmap

        # Check forward mappings
        assert revmap.get_fwd("a") == 1
        assert revmap.get_fwd("b") == 2
        assert revmap.get_fwd("c") == 3
        assert revmap.get_fwd("d") is None

        # Check backward mappings
        assert revmap.get_bwd(1) == "a"
        assert revmap.get_bwd(2) == "b"
        assert revmap.get_bwd(3) == "c"
        assert revmap.get_bwd(4) is None

    def test_mk_with_duplicates(self) -> None:
        """Test creating PIsoMap with duplicate keys or values."""
        # Duplicate keys - later value should win
        pairs = [("a", 1), ("a", 2)]
        revmap = PIsoMap.mk(pairs)
        assert revmap.get_fwd("a") == 2
        assert revmap.get_bwd(2) == "a"
        assert revmap.get_bwd(1) is None

        # Duplicate values - later key should win
        pairs = [("a", 1), ("b", 1)]
        revmap = PIsoMap.mk(pairs)
        assert revmap.get_bwd(1) == "b"
        assert revmap.get_fwd("b") == 1
        assert revmap.get_fwd("a") is None

    def test_insert(self) -> None:
        """Test inserting key-value pairs."""
        revmap: PIsoMap[str, int] = PIsoMap.empty()

        # Insert first pair
        revmap1 = revmap.insert("a", 1)
        assert revmap1.get_fwd("a") == 1
        assert revmap1.get_bwd(1) == "a"
        assert revmap1.size() == 1

        # Original map unchanged
        assert revmap.is_empty()

        # Insert second pair
        revmap2 = revmap1.insert("b", 2)
        assert revmap2.get_fwd("a") == 1
        assert revmap2.get_fwd("b") == 2
        assert revmap2.get_bwd(1) == "a"
        assert revmap2.get_bwd(2) == "b"
        assert revmap2.size() == 2

    def test_insert_overwrites(self) -> None:
        """Test that insert removes old mappings to maintain consistency."""
        revmap = PIsoMap.mk([("a", 1), ("b", 2)])

        # Insert with existing key should remove old value mapping
        revmap1 = revmap.insert("a", 3)
        assert revmap1.get_fwd("a") == 3
        assert revmap1.get_bwd(3) == "a"
        assert revmap1.get_bwd(1) is None  # Old value mapping removed
        assert revmap1.size() == 2  # b->2 still exists

        # Insert with existing value should remove old key mapping
        revmap2 = revmap.insert("c", 2)
        assert revmap2.get_fwd("c") == 2
        assert revmap2.get_bwd(2) == "c"
        assert revmap2.get_fwd("b") is None  # Old key mapping removed
        assert revmap2.size() == 2  # a->1 still exists

    def test_remove_key(self) -> None:
        """Test removing mappings by key."""
        revmap = PIsoMap.mk([("a", 1), ("b", 2), ("c", 3)])

        # Remove existing key
        revmap1 = revmap.remove_key("b")
        assert revmap1.get_fwd("b") is None
        assert revmap1.get_bwd(2) is None
        assert revmap1.get_fwd("a") == 1  # Other mappings preserved
        assert revmap1.get_fwd("c") == 3
        assert revmap1.size() == 2

        # Remove non-existing key
        revmap2 = revmap.remove_key("d")
        assert revmap2 == revmap  # No change

    def test_remove_value(self) -> None:
        """Test removing mappings by value."""
        revmap = PIsoMap.mk([("a", 1), ("b", 2), ("c", 3)])

        # Remove existing value
        revmap1 = revmap.remove_value(2)
        assert revmap1.get_bwd(2) is None
        assert revmap1.get_fwd("b") is None
        assert revmap1.get_bwd(1) == "a"  # Other mappings preserved
        assert revmap1.get_bwd(3) == "c"
        assert revmap1.size() == 2

        # Remove non-existing value
        revmap2 = revmap.remove_value(4)
        assert revmap2 == revmap  # No change

    def test_contains(self) -> None:
        """Test containment checks."""
        revmap = PIsoMap.mk([("a", 1), ("b", 2)])

        assert revmap.contains_key("a")
        assert revmap.contains_key("b")
        assert not revmap.contains_key("c")

        assert revmap.contains_value(1)
        assert revmap.contains_value(2)
        assert not revmap.contains_value(3)

        # Test __contains__ (Python 'in' operator)
        assert "a" in revmap
        assert "c" not in revmap

    def test_iteration(self) -> None:
        """Test iteration over keys, values, and items."""
        pairs = [("a", 1), ("b", 2), ("c", 3)]
        revmap = PIsoMap.mk(pairs)

        # Keys
        keys = set(revmap.keys())
        assert keys == {"a", "b", "c"}

        # Values
        values = set(revmap.values())
        assert values == {1, 2, 3}

        # Items
        items = set(revmap.items())
        assert items == {("a", 1), ("b", 2), ("c", 3)}

    def test_getitem(self) -> None:
        """Test bracket operator access."""
        revmap = PIsoMap.mk([("a", 1), ("b", 2)])

        assert revmap["a"] == 1
        assert revmap["b"] == 2

        with pytest.raises(KeyError):
            _ = revmap["c"]

    def test_equality(self) -> None:
        """Test equality comparison."""
        revmap1 = PIsoMap.mk([("a", 1), ("b", 2)])
        revmap2 = PIsoMap.mk([("a", 1), ("b", 2)])
        revmap3 = PIsoMap.mk([("a", 1), ("c", 3)])

        assert revmap1 == revmap2
        assert revmap1 != revmap3
        assert revmap1 != "not a revmap"

    def test_repr(self) -> None:
        """Test string representation."""
        revmap = PIsoMap.mk([("a", 1), ("b", 2)])
        repr_str = repr(revmap)
        assert repr_str.startswith("PIsoMap(")
        assert "('a', 1)" in repr_str
        assert "('b', 2)" in repr_str

    def test_bidirectional_consistency(self) -> None:
        """Test that forward and backward mappings stay consistent."""
        revmap = PIsoMap.mk([("a", 1), ("b", 2), ("c", 3)])

        # Every forward mapping should have a corresponding backward mapping
        for key, value in revmap.items():
            assert revmap.get_bwd(value) == key

        # Every backward mapping should have a corresponding forward mapping
        for value in revmap.values():
            key_opt = revmap.get_bwd(value)
            assert key_opt is not None  # Should always exist
            assert revmap.get_fwd(key_opt) == value
