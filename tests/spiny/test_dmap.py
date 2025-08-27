from spiny.dmap import DKey, DMap


class Domain:
    pass


class NameKey(DKey[Domain, str]):
    pass


class AgeKey(DKey[Domain, int]):
    pass


class ActiveKey(DKey[Domain, bool]):
    pass


def test_empty_dmap():
    """Test creating an empty DMap"""
    dmap = DMap[Domain]()
    assert dmap.null()
    assert dmap.size() == 0


def test_put_and_lookup():
    """Test putting and looking up values with dependent keys"""
    dmap = DMap[Domain]()
    name_key = NameKey()
    age_key = AgeKey()

    # Put values
    dmap = dmap.put(name_key, "Alice")
    dmap = dmap.put(age_key, 30)

    # Lookup values
    assert dmap.lookup(name_key) == "Alice"
    assert dmap.lookup(age_key) == 30

    # Test non-existent key
    active_key = ActiveKey()
    assert dmap.lookup(active_key) is None


def test_contains():
    """Test checking if DMap contains a key"""
    dmap = DMap[Domain]()
    name_key = NameKey()
    age_key = AgeKey()
    active_key = ActiveKey()

    dmap = dmap.put(name_key, "Bob")

    assert dmap.contains(name_key)
    assert not dmap.contains(age_key)
    assert not dmap.contains(active_key)


def test_remove():
    """Test removing values from DMap"""
    dmap = DMap[Domain]()
    name_key = NameKey()
    age_key = AgeKey()

    # Add values
    dmap = dmap.put(name_key, "Charlie")
    dmap = dmap.put(age_key, 25)

    assert dmap.size() == 2
    assert dmap.contains(name_key)
    assert dmap.contains(age_key)

    # Remove one value
    dmap = dmap.remove(name_key)

    assert dmap.size() == 1
    assert not dmap.contains(name_key)
    assert dmap.contains(age_key)
    assert dmap.lookup(name_key) is None
    assert dmap.lookup(age_key) == 25


def test_size_and_null():
    """Test size and null methods"""
    dmap = DMap[Domain]()
    name_key = NameKey()
    age_key = AgeKey()

    # Initially empty
    assert dmap.null()
    assert dmap.size() == 0

    # Add one item
    dmap = dmap.put(name_key, "David")
    assert not dmap.null()
    assert dmap.size() == 1

    # Add another item
    dmap = dmap.put(age_key, 35)
    assert not dmap.null()
    assert dmap.size() == 2

    # Remove items
    dmap = dmap.remove(name_key)
    assert not dmap.null()
    assert dmap.size() == 1

    dmap = dmap.remove(age_key)
    assert dmap.null()
    assert dmap.size() == 0


def test_static_empty():
    """Test creating empty DMap with static method"""
    dmap = DMap[str].empty()
    assert dmap.null()
    assert dmap.size() == 0


def test_dkey_instance_and_key():
    """Test DKey instance creation and key method"""
    name_key = NameKey.instance()
    assert isinstance(name_key, NameKey)
    assert name_key.key() == "NameKey"

    age_key = AgeKey()
    assert age_key.key() == "AgeKey"


def test_multiple_key_types():
    """Test using different key types with the same value type"""

    class FirstNameKey(DKey[Domain, str]):
        pass

    class LastNameKey(DKey[Domain, str]):
        pass

    dmap = DMap[Domain]()
    first_key = FirstNameKey()
    last_key = LastNameKey()

    dmap = dmap.put(first_key, "John")
    dmap = dmap.put(last_key, "Doe")

    assert dmap.lookup(first_key) == "John"
    assert dmap.lookup(last_key) == "Doe"
    assert dmap.size() == 2


def test_underlying_storage():
    """Test that values are properly wrapped internally"""
    dmap = DMap[Domain]()
    name_key = NameKey()

    dmap = dmap.put(name_key, "Test")

    # Check that we can retrieve the value correctly
    assert dmap.lookup(name_key) == "Test"

    # Check that the key exists in underlying storage
    assert dmap.contains(name_key)


def test_singleton():
    """Test creating a DMap with a single key-value pair"""
    name_key = NameKey()
    dmap = DMap.singleton(name_key, "SingleValue")

    assert not dmap.null()
    assert dmap.size() == 1
    assert dmap.get(name_key) == "SingleValue"
    assert dmap.lookup(name_key) == "SingleValue"
    assert dmap.contains(name_key)


def test_get_with_default():
    """Test get method with default values"""
    dmap = DMap[Domain]()
    name_key = NameKey()
    age_key = AgeKey()

    # Add one value
    dmap = dmap.put(name_key, "Alice")

    # Test get with existing key
    assert dmap.get(name_key) == "Alice"

    # Test get with non-existent key and default
    assert dmap.get(age_key, 99) == 99
    assert dmap.get(age_key, default=42) == 42

    # Test get with non-existent key and no default (should raise KeyError)
    try:
        dmap.get(age_key)
        assert False, "Should have raised KeyError"
    except KeyError as e:
        assert str(e) == "'AgeKey'"


def test_get_vs_lookup():
    """Test that get and lookup behave consistently"""
    dmap = DMap[Domain]()
    name_key = NameKey()
    age_key = AgeKey()

    dmap = dmap.put(name_key, "Bob")

    # Both should return the same value for existing keys
    assert dmap.get(name_key) == dmap.lookup(name_key) == "Bob"

    # For non-existent keys:
    # lookup returns None
    assert dmap.lookup(age_key) is None

    # get with default returns default
    assert dmap.get(age_key, 25) == 25

    # get without default raises KeyError
    try:
        dmap.get(age_key)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_merge_empty_maps():
    """Test merging two empty DMaps"""
    dmap1 = DMap[Domain]()
    dmap2 = DMap[Domain]()

    merged = dmap1.merge(dmap2)
    assert merged.null()
    assert merged.size() == 0


def test_merge_empty_with_nonempty():
    """Test merging an empty DMap with a non-empty one"""
    empty_dmap = DMap[Domain]()
    name_key = NameKey()
    age_key = AgeKey()

    # Non-empty DMap
    nonempty_dmap = DMap[Domain]()
    nonempty_dmap = nonempty_dmap.put(name_key, "Alice")
    nonempty_dmap = nonempty_dmap.put(age_key, 30)

    # Merge empty with non-empty
    merged1 = empty_dmap.merge(nonempty_dmap)
    assert merged1.size() == 2
    assert merged1.lookup(name_key) == "Alice"
    assert merged1.lookup(age_key) == 30

    # Merge non-empty with empty
    merged2 = nonempty_dmap.merge(empty_dmap)
    assert merged2.size() == 2
    assert merged2.lookup(name_key) == "Alice"
    assert merged2.lookup(age_key) == 30


def test_merge_disjoint_maps():
    """Test merging DMaps with no overlapping keys"""
    name_key = NameKey()
    age_key = AgeKey()
    active_key = ActiveKey()

    # First DMap
    dmap1 = DMap[Domain]()
    dmap1 = dmap1.put(name_key, "Charlie")
    dmap1 = dmap1.put(age_key, 25)

    # Second DMap with different key
    dmap2 = DMap[Domain]()
    dmap2 = dmap2.put(active_key, True)

    # Merge
    merged = dmap1.merge(dmap2)
    assert merged.size() == 3
    assert merged.lookup(name_key) == "Charlie"
    assert merged.lookup(age_key) == 25
    assert merged.lookup(active_key)


def test_merge_overlapping_maps():
    """Test merging DMaps with overlapping keys - first map takes precedence"""
    name_key = NameKey()
    age_key = AgeKey()
    active_key = ActiveKey()

    # First DMap
    dmap1 = DMap[Domain]()
    dmap1 = dmap1.put(name_key, "Alice")
    dmap1 = dmap1.put(age_key, 30)

    # Second DMap with overlapping keys
    dmap2 = DMap[Domain]()
    dmap2 = dmap2.put(name_key, "Bob")  # Different value for same key
    dmap2 = dmap2.put(active_key, True)

    # Merge - first map takes precedence for conflicts
    merged = dmap1.merge(dmap2)
    assert merged.size() == 3
    assert merged.lookup(name_key) == "Alice"  # From first map
    assert merged.lookup(age_key) == 30  # From first map
    assert merged.lookup(active_key)  # From second map

    # Test reverse merge to verify precedence
    merged_reverse = dmap2.merge(dmap1)
    assert merged_reverse.size() == 3
    assert merged_reverse.lookup(name_key) == "Bob"  # From second map (now first)
    assert merged_reverse.lookup(age_key) == 30  # From first map (now second)
    assert merged_reverse.lookup(active_key)  # From second map (now first)


def test_merge_preserves_original_maps():
    """Test that merge operation doesn't modify the original maps"""
    name_key = NameKey()
    age_key = AgeKey()

    # Original maps
    dmap1 = DMap[Domain]()
    dmap1 = dmap1.put(name_key, "Original1")

    dmap2 = DMap[Domain]()
    dmap2 = dmap2.put(age_key, 42)

    # Merge
    merged = dmap1.merge(dmap2)

    # Original maps should be unchanged
    assert dmap1.size() == 1
    assert dmap1.lookup(name_key) == "Original1"
    assert dmap1.lookup(age_key) is None

    assert dmap2.size() == 1
    assert dmap2.lookup(age_key) == 42
    assert dmap2.lookup(name_key) is None

    # Merged map should have both
    assert merged.size() == 2
    assert merged.lookup(name_key) == "Original1"
    assert merged.lookup(age_key) == 42


def test_merge_with_self():
    """Test merging a DMap with itself"""
    name_key = NameKey()
    age_key = AgeKey()

    dmap = DMap[Domain]()
    dmap = dmap.put(name_key, "SelfMerge")
    dmap = dmap.put(age_key, 99)

    merged = dmap.merge(dmap)

    # Should be identical to original
    assert merged.size() == 2
    assert merged.lookup(name_key) == "SelfMerge"
    assert merged.lookup(age_key) == 99


def test_merge_complex_scenario():
    """Test a complex merge scenario with multiple key types"""

    class EmailKey(DKey[Domain, str]):
        pass

    class ScoreKey(DKey[Domain, float]):
        pass

    name_key = NameKey()
    age_key = AgeKey()
    active_key = ActiveKey()
    email_key = EmailKey()
    score_key = ScoreKey()

    # Create first map
    dmap1 = DMap[Domain]()
    dmap1 = dmap1.put(name_key, "John")
    dmap1 = dmap1.put(age_key, 35)
    dmap1 = dmap1.put(email_key, "john@example.com")

    # Create second map with some overlapping keys
    dmap2 = DMap[Domain]()
    dmap2 = dmap2.put(name_key, "Jane")  # Will be overridden
    dmap2 = dmap2.put(active_key, True)
    dmap2 = dmap2.put(score_key, 95.5)

    # Merge
    merged = dmap1.merge(dmap2)

    # Verify all expected values
    assert merged.size() == 5
    assert merged.lookup(name_key) == "John"  # From first map (precedence)
    assert merged.lookup(age_key) == 35  # From first map only
    assert merged.lookup(email_key) == "john@example.com"  # From first map only
    assert merged.lookup(active_key)  # From second map only
    assert merged.lookup(score_key) == 95.5  # From second map only
