"""Tests for error handling in streams with unknown chords, notes, and sounds."""

import pytest

from minipat.combinators import note_stream, notename_stream, sound_stream
from minipat.dsl import note
from minipat.kit import DEFAULT_KIT


def test_unknown_chord_raises_error() -> None:
    """Test that unknown chord types raise ValueError."""
    # Test with note_stream
    with pytest.raises(ValueError, match="Unknown chord type: unknownchord"):
        note_stream("c4`unknownchord")

    # Test with note function from dsl
    with pytest.raises(ValueError, match="Unknown chord type: unknownchord"):
        note("c4`unknownchord")

    # Test with invalid chord after valid note
    with pytest.raises(ValueError, match="Unknown chord type: badchord"):
        note("f#`badchord")


def test_chord_with_no_name_raises_error() -> None:
    """Test that backtick without chord name raises ValueError."""
    with pytest.raises(ValueError, match="Backtick without chord name"):
        note_stream("c4`")

    with pytest.raises(ValueError, match="Backtick without chord name"):
        note("g3`")


def test_invalid_note_format_raises_error() -> None:
    """Test that invalid note formats raise ValueError."""
    # Invalid note name
    with pytest.raises(ValueError, match="Invalid note name"):
        notename_stream("x4")

    with pytest.raises(ValueError, match="Invalid note name"):
        notename_stream("h#")

    # Empty string causes parse error (different from ValueError)
    with pytest.raises(Exception):  # Will be a parse error, not ValueError
        note_stream("")


def test_old_style_chord_notation_raises_helpful_error() -> None:
    """Test that old-style chord notation without backtick gives helpful error."""
    with pytest.raises(ValueError, match="Use a backtick.*between note and chord"):
        note("c4maj7")  # Missing backtick

    with pytest.raises(ValueError, match="Use a backtick.*between note and chord"):
        note("f#min")  # Missing backtick


def test_unknown_sound_raises_error() -> None:
    """Test that unknown drum sound identifiers raise ValueError."""
    with pytest.raises(ValueError, match="Unknown drum sound 'unknownsound'"):
        sound_stream(DEFAULT_KIT, "unknownsound")

    # Test that error includes available sounds
    try:
        sound_stream(DEFAULT_KIT, "fakesound")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Available sounds:" in str(e)
        # Check some common sounds are listed
        assert "bd" in str(e)
        assert "sd" in str(e)
        assert "hh" in str(e)


def test_valid_inputs_dont_raise_errors() -> None:
    """Test that valid inputs work without errors."""
    # Valid chord types
    note("c4`maj7")
    note("f#`min")
    note("bb3`sus4")

    # Valid notes
    notename_stream("c4 d4 e4")
    notename_stream("f# g# a#")

    # Valid sounds
    sound_stream(DEFAULT_KIT, "bd sd hh")
    sound_stream(DEFAULT_KIT, "cy rd sp")

    # Valid patterns with rests
    note("c4 ~ g4`maj7")
    sound_stream(DEFAULT_KIT, "bd ~ sd ~")


def test_out_of_range_notes_raise_error() -> None:
    """Test that notes outside MIDI range raise ValueError."""
    # Note too high (MIDI only goes to 127)
    # C11 would be MIDI note 132 (11*12 + 0)
    with pytest.raises(ValueError, match="out of MIDI range"):
        note("c11")

    # G#10 would be MIDI note 128 (10*12 + 8)
    with pytest.raises(ValueError, match="out of MIDI range"):
        note("g#10")


def test_chord_producing_no_valid_notes_raises_error() -> None:
    """Test that chords producing no valid MIDI notes raise ValueError."""
    # This would happen with extremely high root notes where all chord
    # intervals would exceed MIDI 127
    # G#10 is MIDI note 128, so ALL notes of any chord would be > 127
    with pytest.raises(ValueError, match="produces no valid MIDI notes"):
        note("g#10`maj7")  # G#10 = 128, all chord notes would be > 127
