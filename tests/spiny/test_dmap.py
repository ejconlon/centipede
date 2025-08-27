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
