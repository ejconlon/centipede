"""Tests for chord voicing modifications (inversions and drop voicings)."""

from fractions import Fraction

from minipat.chords import (
    apply_drop_voicing,
    apply_inversion,
)
from minipat.combinators import ChordElemParser, note_stream
from minipat.dsl import note
from minipat.messages import NoteKey
from minipat.time import CycleArc, CycleTime
from minipat.types import Note
from spiny import PSeq


def test_chord_inversion_functions() -> None:
    """Test the apply_inversion function."""
    # Create a C major triad [C4, E4, G4]
    notes = PSeq.mk([Note(48), Note(52), Note(55)])

    # Root position - no change
    inv0 = apply_inversion(notes, 0)
    assert list(inv0) == [Note(48), Note(52), Note(55)]

    # First inversion - [E4, G4, C5]
    inv1 = apply_inversion(notes, 1)
    assert list(inv1) == [Note(52), Note(55), Note(60)]  # C moved up an octave

    # Second inversion - [G4, C5, E5]
    inv2 = apply_inversion(notes, 2)
    assert list(inv2) == [Note(55), Note(60), Note(64)]  # C and E moved up an octave

    # Third inversion (wraps around) - [C5, E5, G5]
    inv3 = apply_inversion(notes, 3)
    assert list(inv3) == [Note(60), Note(64), Note(67)]  # All notes moved up


def test_chord_drop_voicing_functions() -> None:
    """Test the apply_drop_voicing function."""
    # Create a C major 7 chord [C4, E4, G4, B4]
    notes = PSeq.mk([Note(48), Note(52), Note(55), Note(59)])

    # No drop - unchanged
    drop0 = apply_drop_voicing(notes, 0)
    assert list(drop0) == [Note(48), Note(52), Note(55), Note(59)]

    # Drop 2 - drop the 2nd note from the top (G4 -> G3)
    drop2 = apply_drop_voicing(notes, 2)
    assert list(drop2) == [
        Note(48),
        Note(52),
        Note(43),
        Note(59),
    ]  # G dropped an octave

    # Drop 3 - drop the 3rd note from the top (E4 -> E3)
    drop3 = apply_drop_voicing(notes, 3)
    assert list(drop3) == [
        Note(48),
        Note(40),
        Note(55),
        Note(59),
    ]  # E dropped an octave

    # Drop 4 - drop the 4th note from the top (C4 -> C3)
    drop4 = apply_drop_voicing(notes, 4)
    assert list(drop4) == [
        Note(36),
        Note(52),
        Note(55),
        Note(59),
    ]  # C dropped an octave


def test_chord_parser_with_inversions() -> None:
    """Test ChordElemParser with inversion modifiers."""
    parser = ChordElemParser()

    # C major 7, root position
    root = parser.apply("c4`maj7")
    assert list(root) == [Note(48), Note(52), Note(55), Note(59)]

    # C major 7, first inversion
    inv1 = parser.apply("c4`maj7`inv1")
    assert list(inv1) == [Note(52), Note(55), Note(59), Note(60)]

    # C major 7, second inversion
    inv2 = parser.apply("c4`maj7`inv2")
    assert list(inv2) == [Note(55), Note(59), Note(60), Note(64)]


def test_chord_parser_with_drop_voicings() -> None:
    """Test ChordElemParser with drop voicing modifiers."""
    parser = ChordElemParser()

    # C major 7, drop 2
    drop2 = parser.apply("c4`maj7`drop2")
    assert list(drop2) == [Note(48), Note(52), Note(43), Note(59)]

    # C major 7, drop 3
    drop3 = parser.apply("c4`maj7`drop3")
    assert list(drop3) == [Note(48), Note(40), Note(55), Note(59)]


def test_chord_parser_combined_modifiers() -> None:
    """Test ChordElemParser with combined inversion and drop modifiers."""
    parser = ChordElemParser()

    # C major 7, first inversion then drop 2
    combined = parser.apply("c4`maj7`inv1`drop2")
    # First inversion: [E4, G4, B4, C5] = [52, 55, 59, 60]
    # Then drop 2 (2nd from top = B4): [52, 55, 47, 60]
    assert list(combined) == [Note(52), Note(55), Note(47), Note(60)]


def test_chord_voicing_in_streams() -> None:
    """Test voicing modifiers in note streams."""
    # Create a pattern with inversions
    stream = note_stream("c4`maj7 c4`maj7`inv1 c4`maj7`inv2")

    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = list(stream.unstream(arc))

    # Each chord produces 4 notes (maj7), 3 chords = 12 events
    assert len(events) == 12

    # Collect notes by time position (each chord occupies 1/3 of the cycle)
    first_chord_notes = []
    second_chord_notes = []
    third_chord_notes = []

    for span, event in events:
        note_val = event.val.lookup(NoteKey())
        if note_val is not None:
            midi_note = int(note_val)
            # Check time position to group notes
            if span.active.start < Fraction(1, 3):
                first_chord_notes.append(midi_note)
            elif span.active.start < Fraction(2, 3):
                second_chord_notes.append(midi_note)
            else:
                third_chord_notes.append(midi_note)

    # First chord - root position [48, 52, 55, 59]
    assert sorted(first_chord_notes) == [48, 52, 55, 59]

    # Second chord - first inversion [52, 55, 59, 60]
    assert sorted(second_chord_notes) == [52, 55, 59, 60]

    # Third chord - second inversion [55, 59, 60, 64]
    assert sorted(third_chord_notes) == [55, 59, 60, 64]


def test_chord_voicing_in_dsl() -> None:
    """Test voicing modifiers using the DSL note function."""
    # Create a flow with chord voicings
    flow = note("c4`maj7 g4`dom7`inv1 f4`min7`drop2")

    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = list(flow.stream.unstream(arc))

    # Each chord has 4 notes (maj7, dom7, min7), 3 chords = 12 events
    assert len(events) == 12


def test_invalid_voicing_modifiers() -> None:
    """Test error handling for invalid voicing modifiers."""
    import pytest

    parser = ChordElemParser()

    # Invalid inversion format
    with pytest.raises(ValueError, match="Invalid inversion modifier"):
        parser.apply("c4`maj7`invX")

    # Invalid drop format
    with pytest.raises(ValueError, match="Invalid drop voicing modifier"):
        parser.apply("c4`maj7`dropY")

    # Unknown modifier
    with pytest.raises(ValueError, match="Unknown voicing modifier"):
        parser.apply("c4`maj7`unknown")


def test_voicing_edge_cases() -> None:
    """Test edge cases for voicing modifications."""
    parser = ChordElemParser()

    # High note where inversion might exceed MIDI range
    high_notes = parser.apply("g9`maj7`inv1")
    # All notes should be filtered to valid MIDI range
    for chord_note in high_notes:
        assert 0 <= int(chord_note) <= 127

    # Low note where drop voicing might go below MIDI range
    low_notes = parser.apply("c1`maj7`drop4")
    # All notes should be filtered to valid MIDI range
    for chord_note in low_notes:
        assert 0 <= int(chord_note) <= 127


def test_multiple_voicing_modifiers() -> None:
    """Test multiple voicing modifiers in sequence."""
    parser = ChordElemParser()

    # Test order matters: inv1 then drop2 vs drop2 then inv1
    inv_then_drop = parser.apply("c4`maj7`inv1`drop2")
    drop_then_inv = parser.apply("c4`maj7`drop2`inv1")

    # These should produce different results
    assert list(inv_then_drop) != list(drop_then_inv)


def test_triads_with_voicings() -> None:
    """Test voicing modifications work with triads."""
    parser = ChordElemParser()

    # C major triad inversions
    root = parser.apply("c4`maj")
    assert list(root) == [Note(48), Note(52), Note(55)]

    inv1 = parser.apply("c4`maj`inv1")
    assert list(inv1) == [Note(52), Note(55), Note(60)]

    inv2 = parser.apply("c4`maj`inv2")
    assert list(inv2) == [Note(55), Note(60), Note(64)]

    # Drop voicings on triads
    drop2 = parser.apply("c4`maj`drop2")
    assert list(drop2) == [Note(48), Note(40), Note(55)]  # E dropped
