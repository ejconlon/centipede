"""Tests for drum kit functionality."""

from __future__ import annotations

import pytest

from minipat.kit import (
    DEFAULT_KIT,
    DrumSoundElemParser,
    Kit,
    Sound,
    make_drum_sound,
)
from minipat.messages import Note
from spiny.map import PMap


class TestSound:
    """Tests for Sound class."""

    def test_drum_sound_creation(self) -> None:
        """Test basic Sound creation."""
        sound = Sound(Note(36))
        assert sound.note == Note(36)
        assert sound.velocity is None
        assert sound.channel is None

    def test_drum_sound_with_velocity_and_channel(self) -> None:
        """Test Sound creation with velocity and channel."""
        from minipat.messages import Channel, Velocity

        sound = Sound(Note(36), Velocity(100), Channel(9))
        assert sound.note == Note(36)
        assert sound.velocity == Velocity(100)
        assert sound.channel == Channel(9)


class TestMakeSound:
    """Tests for make_drum_sound helper function."""

    def test_make_drum_sound_basic(self) -> None:
        """Test basic drum sound creation."""
        sound = make_drum_sound(36)
        assert sound.note == Note(36)
        assert sound.velocity is None
        assert sound.channel is None

    def test_make_drum_sound_with_velocity_and_channel(self) -> None:
        """Test drum sound creation with velocity and channel."""
        sound = make_drum_sound(36, 100, 9)
        assert sound.note == Note(36)
        from minipat.messages import Channel, Velocity

        assert sound.velocity == Velocity(100)
        assert sound.channel == Channel(9)

    def test_make_drum_sound_validation(self) -> None:
        """Test that make_drum_sound validates input values."""
        # Test invalid note
        with pytest.raises(ValueError):
            make_drum_sound(128)  # Note out of range

        # Test invalid velocity
        with pytest.raises(ValueError):
            make_drum_sound(36, 128)  # Velocity out of range

        # Test invalid channel
        with pytest.raises(ValueError):
            make_drum_sound(36, 100, 16)  # Channel out of range


class TestKit:
    """Tests for Kit class."""

    def test_empty_drum_kit(self) -> None:
        """Test empty drum kit creation."""
        kit: Kit = PMap.empty()
        assert kit.lookup("bd") is None
        assert kit.size() == 0

    def test_drum_kit_with_sounds(self) -> None:
        """Test drum kit with initial sounds."""
        sounds = [("bd", Sound(Note(36))), ("sd", Sound(Note(38)))]
        kit: Kit = PMap.mk(sounds)

        bd_sound = kit.lookup("bd")
        assert bd_sound is not None
        assert bd_sound.note == Note(36)

        sd_sound = kit.lookup("sd")
        assert sd_sound is not None
        assert sd_sound.note == Note(38)

    def test_add_sound(self) -> None:
        """Test adding a sound to the kit."""
        kit: Kit = PMap.empty()
        kit = kit.put("bd", Sound(Note(36)))

        sound = kit.lookup("bd")
        assert sound is not None
        assert sound.note == Note(36)

    def test_remove_sound(self) -> None:
        """Test removing a sound from the kit."""
        kit: Kit = PMap.mk([("bd", Sound(Note(36)))])

        # Sound exists initially
        assert kit.lookup("bd") is not None

        # Remove sound
        has_bd = kit.contains("bd")
        kit = kit.remove("bd")
        assert has_bd is True
        assert kit.lookup("bd") is None

        # Try to remove non-existent sound
        has_nonexistent = kit.contains("nonexistent")
        kit = kit.remove("nonexistent")
        assert has_nonexistent is False

    def test_list_sounds(self) -> None:
        """Test listing all sounds in the kit."""
        sounds = [("bd", Sound(Note(36))), ("sd", Sound(Note(38)))]
        kit: Kit = PMap.mk(sounds)

        listed_sounds = dict(kit.items())
        assert len(listed_sounds) == 2
        assert "bd" in listed_sounds
        assert "sd" in listed_sounds
        assert listed_sounds["bd"].note == Note(36)
        assert listed_sounds["sd"].note == Note(38)

        # Verify it's a copy (modifying returned dict doesn't affect kit)
        listed_sounds["hh"] = Sound(Note(42))
        assert kit.lookup("hh") is None


class TestDefaultKit:
    """Tests for the default drum kit."""

    def test_default_kit_has_common_sounds(self) -> None:
        """Test that the default kit has common drum sounds."""
        # Test some basic sounds
        assert DEFAULT_KIT.lookup("bd") is not None
        assert DEFAULT_KIT.lookup("sd") is not None
        assert DEFAULT_KIT.lookup("hh") is not None
        assert DEFAULT_KIT.lookup("cy") is not None

    def test_default_kit_bass_drum(self) -> None:
        """Test bass drum mapping."""
        bd = DEFAULT_KIT.lookup("bd")
        assert bd is not None
        assert bd.note == Note(36)

    def test_default_kit_snare_drum(self) -> None:
        """Test snare drum mapping."""
        sd = DEFAULT_KIT.lookup("sd")
        assert sd is not None
        assert sd.note == Note(38)

    def test_default_kit_hi_hat(self) -> None:
        """Test hi-hat mapping."""
        hh = DEFAULT_KIT.lookup("hh")
        assert hh is not None
        assert hh.note == Note(42)


class TestDrumSoundElemParser:
    """Tests for DrumSoundElemParser."""

    def test_parse_basic_drum_sounds(self) -> None:
        """Test parsing basic drum sound identifiers."""
        kit = DEFAULT_KIT
        parser = DrumSoundElemParser(kit)

        # Test bass drum
        bd_note = parser.forward("bd")
        assert bd_note == Note(36)

        # Test snare drum
        sd_note = parser.forward("sd")
        assert sd_note == Note(38)

        # Test hi-hat
        hh_note = parser.forward("hh")
        assert hh_note == Note(42)

    def test_parse_unknown_sound(self) -> None:
        """Test parsing unknown drum sound identifier."""
        kit = DEFAULT_KIT
        parser = DrumSoundElemParser(kit)

        with pytest.raises(ValueError, match="Unknown drum sound 'unknownsound'"):
            parser.forward("unknownsound")

    def test_backward_conversion(self) -> None:
        """Test converting notes back to drum sound identifiers."""
        kit = DEFAULT_KIT
        parser = DrumSoundElemParser(kit)

        # Test bass drum
        identifier = parser.backward(Note(36))
        assert identifier == "bd"

        # Test snare drum
        identifier = parser.backward(Note(38))
        assert identifier == "sd"

        # Test unknown note (should return note number)
        identifier = parser.backward(Note(100))
        assert identifier == "100"


class TestNucleusKitManagement:
    """Tests for Nucleus-based kit management."""

    def test_nucleus_drum_kit_property(self) -> None:
        """Test getting and setting the drum_kit property."""
        from minipat.dsl import Nucleus

        n = Nucleus.boot()
        try:
            # Test getting kit
            kit = n.drum_kit
            assert isinstance(kit, PMap)
            assert kit.lookup("bd") is not None

            # Test setting kit
            custom_kit: Kit = PMap.mk([("custom", Sound(Note(100)))])
            n.drum_kit = custom_kit
            assert n.drum_kit.lookup("custom") is not None
            assert n.drum_kit.lookup("custom").note == Note(100)  # type: ignore
        finally:
            n.stop()

    def test_nucleus_add_drum_sound(self) -> None:
        """Test adding drum sounds via Nucleus."""
        from minipat.dsl import Nucleus

        n = Nucleus.boot()
        try:
            n.add_drum_sound("test", 100, 80, 9)

            sound = n.drum_kit.lookup("test")
            assert sound is not None
            assert sound.note == Note(100)
        finally:
            n.stop()

    def test_nucleus_remove_drum_sound(self) -> None:
        """Test removing drum sounds via Nucleus."""
        from minipat.dsl import Nucleus

        n = Nucleus.boot()
        try:
            # Add a sound first
            n.add_drum_sound("test", 100)
            assert n.drum_kit.lookup("test") is not None

            # Remove it
            removed = n.remove_drum_sound("test")
            assert removed is True
            assert n.drum_kit.lookup("test") is None

            # Try to remove non-existent sound
            removed = n.remove_drum_sound("nonexistent")
            assert removed is False
        finally:
            n.stop()

    def test_nucleus_list_drum_sounds(self) -> None:
        """Test listing drum sounds via Nucleus."""
        from minipat.dsl import Nucleus

        n = Nucleus.boot()
        try:
            sounds = n.list_drum_sounds()
            assert isinstance(sounds, PMap)
            assert sounds.contains("bd")
            assert sounds.contains("sd")
            bd_sound = sounds.lookup("bd")
            assert bd_sound is not None
            assert bd_sound.note == Note(36)
        finally:
            n.stop()

    def test_nucleus_reset_kit(self) -> None:
        """Test resetting kit via Nucleus."""
        from minipat.dsl import Nucleus

        n = Nucleus.boot()
        try:
            # Modify kit
            n.add_drum_sound("custom", 100)
            assert n.drum_kit.lookup("custom") is not None

            # Reset kit
            n.reset_kit()
            assert n.drum_kit.lookup("custom") is None
            assert n.drum_kit.lookup("bd") is not None
        finally:
            n.stop()


class TestKitIntegration:
    """Integration tests for kit functionality with pattern system."""

    def test_kit_stream_with_kit(self) -> None:
        """Test creating streams from kit patterns with specific kit."""
        from minipat.combinators import kit_stream_with_kit

        kit = DEFAULT_KIT
        # Create a stream from a drum pattern
        stream = kit_stream_with_kit("bd sd bd sd", kit)

        # Verify the stream contains the correct notes
        # This is a basic test - full integration would require
        # more complex stream evaluation
        assert stream is not None

    def test_nucleus_kit_method(self) -> None:
        """Test the nucleus kit method."""
        from minipat.dsl import Nucleus

        n = Nucleus.boot()
        try:
            # Create a flow from a drum pattern using nucleus
            flow = n.kit("bd sd hh")

            # Verify the flow was created
            assert flow is not None
            assert hasattr(flow, "stream")
        finally:
            n.stop()

    def test_nucleus_kit_with_custom_sounds(self) -> None:
        """Test nucleus kit functionality with custom drum sounds."""
        from minipat.dsl import Nucleus

        n = Nucleus.boot()
        try:
            # Add a custom drum sound
            n.add_drum_sound("custom", 100)

            # Use it in a pattern
            flow = n.kit("bd custom sd")
            assert flow is not None
        finally:
            n.stop()

    def test_error_handling_in_nucleus_patterns(self) -> None:
        """Test error handling when using unknown drum sounds in nucleus patterns."""
        from minipat.dsl import Nucleus

        n = Nucleus.boot()
        try:
            # This should raise an error due to unknown drum sound
            with pytest.raises(ValueError):
                # Attempting to evaluate this pattern should fail
                n.kit("bd unknownsound sd")
                # Force evaluation by trying to access the pattern elements
                # In real usage, this would happen when the pattern is played
        finally:
            n.stop()
