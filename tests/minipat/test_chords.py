"""Tests for chord functionality."""

from fractions import Fraction

from minipat.chords import (
    Chord,
    chord_to_notes,
    get_chord_intervals,
    parse_chord_name,
)
from minipat.combinators import ChordElemParser
from minipat.dsl import note
from minipat.messages import NoteKey
from minipat.time import CycleArc, CycleTime


def test_chord_name_parsing() -> None:
    """Test parsing various chord name strings."""
    # Test basic chord names
    assert parse_chord_name("maj") == Chord("maj")
    assert parse_chord_name("min") == Chord("min")
    assert parse_chord_name("dim") == Chord("dim")
    assert parse_chord_name("aug") == Chord("aug")

    # Test aliases
    assert parse_chord_name("major") == Chord("maj")
    assert parse_chord_name("M") == Chord("maj")
    assert parse_chord_name("minor") == Chord("min")
    assert parse_chord_name("m") == Chord("min")

    # Test seventh chords
    assert parse_chord_name("maj7") == Chord("maj7")
    assert parse_chord_name("M7") == Chord("maj7")
    assert parse_chord_name("min7") == Chord("min7")
    assert parse_chord_name("m7") == Chord("min7")
    assert parse_chord_name("dom7") == Chord("dom7")

    # Test extensions
    assert parse_chord_name("maj9") == Chord("maj9")
    assert parse_chord_name("add9") == Chord("add9")
    assert parse_chord_name("sus2") == Chord("sus2")
    assert parse_chord_name("sus4") == Chord("sus4")

    # Test unknown chord
    assert parse_chord_name("unknown") is None

    # Test case insensitivity
    assert parse_chord_name("MAJ") == Chord("maj")
    assert parse_chord_name("Min") == Chord("min")
    assert parse_chord_name("DOM7") == Chord("dom7")
    assert parse_chord_name("mAJ7") == Chord("maj7")


def test_chord_intervals() -> None:
    """Test chord interval mappings."""
    # Test basic triads
    assert list(get_chord_intervals(Chord("maj"))) == [0, 4, 7]
    assert list(get_chord_intervals(Chord("min"))) == [0, 3, 7]
    assert list(get_chord_intervals(Chord("dim"))) == [0, 3, 6]
    assert list(get_chord_intervals(Chord("aug"))) == [0, 4, 8]

    # Test seventh chords
    assert list(get_chord_intervals(Chord("maj7"))) == [0, 4, 7, 11]
    assert list(get_chord_intervals(Chord("min7"))) == [0, 3, 7, 10]
    assert list(get_chord_intervals(Chord("dom7"))) == [0, 4, 7, 10]

    # Test suspended chords
    assert list(get_chord_intervals(Chord("sus2"))) == [0, 2, 7]
    assert list(get_chord_intervals(Chord("sus4"))) == [0, 5, 7]


def test_chord_to_notes() -> None:
    """Test converting chords to Note objects."""
    from minipat.messages import Note

    # C major chord (C4 = 48)
    c_major = chord_to_notes(Note(48), Chord("maj"))
    assert [int(n) for n in c_major] == [48, 52, 55]  # C, E, G

    # F# minor chord (F#4 = 54)
    fs_minor = chord_to_notes(Note(54), Chord("min"))
    assert [int(n) for n in fs_minor] == [54, 57, 61]  # F#, A, C#

    # G dominant 7th (G4 = 55)
    g_dom7 = chord_to_notes(Note(55), Chord("dom7"))
    assert [int(n) for n in g_dom7] == [55, 59, 62, 65]  # G, B, D, F


def test_chord_elem_parser() -> None:
    """Test the ChordElemParser class."""
    parser = ChordElemParser()

    # Test basic major chord
    c_maj = parser.apply("c4`maj")
    assert len(c_maj) == 3
    note_values = [int(note) for note in c_maj]
    assert note_values == [48, 52, 55]  # C4, E4, G4

    # Test minor chord with sharp
    fs_min = parser.apply("f#4`min")
    assert len(fs_min) == 3
    note_values = [int(note) for note in fs_min]
    assert note_values == [54, 57, 61]  # F#4, A4, C#5

    # Test seventh chord with flat
    bb_dom7 = parser.apply("bb4`dom7")
    assert len(bb_dom7) == 4
    note_values = [int(note) for note in bb_dom7]
    assert note_values == [58, 62, 65, 68]  # Bb4, D5, F5, Ab5

    # Test chord without explicit octave (uses default octave 4)
    c_maj_default = parser.apply("c`maj")
    assert len(c_maj_default) == 3
    note_values = [int(note) for note in c_maj_default]
    assert note_values == [48, 52, 55]  # C4, E4, G4

    # Test suspended chord
    g_sus4 = parser.apply("g4`sus4")
    assert len(g_sus4) == 3
    note_values = [int(note) for note in g_sus4]
    assert note_values == [55, 60, 62]  # G4, C5, D5

    # Test case-insensitive chord names
    c_maj_upper = parser.apply("C4`MAJ")
    assert len(c_maj_upper) == 3
    note_values = [int(note) for note in c_maj_upper]
    assert note_values == [48, 52, 55]  # C4, E4, G4

    f_min_mixed = parser.apply("F#4`Min")
    assert len(f_min_mixed) == 3
    note_values = [int(note) for note in f_min_mixed]
    assert note_values == [54, 57, 61]  # F#4, A4, C#5

    # Test bare notes (without chord)
    c_note = parser.apply("c4")
    assert len(c_note) == 1
    assert int(c_note[0]) == 48  # Just C4

    fs_note = parser.apply("f#")
    assert len(fs_note) == 1
    assert int(fs_note[0]) == 54  # F# with default octave

    bb_note = parser.apply("bb3")
    assert len(bb_note) == 1
    assert int(bb_note[0]) == 46  # Bb3


def test_chord_elem_parser_errors() -> None:
    """Test ChordElemParser error handling."""
    parser = ChordElemParser()

    # Test backtick without chord name
    try:
        parser.apply("c4`")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Backtick without chord name" in str(e)

    # Test invalid note name
    try:
        parser.apply("x4`maj")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid note name" in str(e)

    # Test unknown chord type
    try:
        parser.apply("c4`unknown")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown chord type" in str(e)

    # Test old-style chord notation (helpful error)
    try:
        parser.apply("c4maj")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Use a backtick" in str(e)
        assert "c4`maj" in str(e)  # Should suggest the correct format


def test_chord_dsl_function() -> None:
    """Test the chord DSL function."""
    # Test basic chord progression
    progression = note("c4`maj f4`min g4`maj")

    # Query events from the flow
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = progression.stream.unstream(arc)
    event_list = list(events)

    # Should have events for all three chords
    assert len(event_list) >= 9  # At least 3 notes per chord

    # Group events by their timing to identify chords
    chord_groups: dict[Fraction, list[int]] = {}
    for span, event in event_list:
        start_time = span.active.start
        if start_time not in chord_groups:
            chord_groups[start_time] = []
        note_val = event.val.lookup(NoteKey())
        if note_val is not None:
            chord_groups[start_time].append(int(note_val))

    # Sort chord groups by time
    sorted_times = sorted(chord_groups.keys())
    assert len(sorted_times) == 3  # Three chords

    # Check first chord (C major)
    first_chord = sorted(chord_groups[sorted_times[0]])
    assert first_chord == [48, 52, 55]  # C4, E4, G4

    # Check second chord (F minor)
    second_chord = sorted(chord_groups[sorted_times[1]])
    assert second_chord == [53, 56, 60]  # F4, Ab4, C5

    # Check third chord (G major)
    third_chord = sorted(chord_groups[sorted_times[2]])
    assert third_chord == [55, 59, 62]  # G4, B4, D5


def test_chord_with_rests() -> None:
    """Test chord patterns with rests."""
    progression = note("c4`maj ~ f4`min")

    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = progression.stream.unstream(arc)
    event_list = list(events)

    # Group events by timing
    chord_groups: dict[Fraction, list[int]] = {}
    for span, event in event_list:
        start_time = span.active.start
        if start_time not in chord_groups:
            chord_groups[start_time] = []
        note_val = event.val.lookup(NoteKey())
        if note_val is not None:
            chord_groups[start_time].append(int(note_val))

    # Should have two chord groups (no events during rest)
    sorted_times = sorted(chord_groups.keys())
    assert len(sorted_times) == 2

    # First chord should be C major
    first_chord = sorted(chord_groups[sorted_times[0]])
    assert first_chord == [48, 52, 55]

    # Second chord should be F minor
    second_chord = sorted(chord_groups[sorted_times[1]])
    assert second_chord == [53, 56, 60]


def test_chord_with_bare_notes() -> None:
    """Test chord patterns that include bare notes."""
    # Mix of chords and single notes
    progression = note("c4 e4 g4`maj")

    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))
    events = progression.stream.unstream(arc)
    event_list = list(events)

    # Group events by timing
    note_groups: dict[Fraction, list[int]] = {}
    for span, event in event_list:
        start_time = span.active.start
        if start_time not in note_groups:
            note_groups[start_time] = []
        note_val = event.val.lookup(NoteKey())
        if note_val is not None:
            note_groups[start_time].append(int(note_val))

    sorted_times = sorted(note_groups.keys())
    assert len(sorted_times) == 3

    # First should be single C4
    assert note_groups[sorted_times[0]] == [48]

    # Second should be single E4
    assert note_groups[sorted_times[1]] == [52]

    # Third should be G major chord
    third_notes = sorted(note_groups[sorted_times[2]])
    assert third_notes == [55, 59, 62]  # G4, B4, D5


def test_complex_chord_names() -> None:
    """Test complex chord names and variations."""
    parser = ChordElemParser()

    # Test major 7th
    cmaj7 = parser.apply("c4`maj7")
    note_values = [int(note) for note in cmaj7]
    assert note_values == [48, 52, 55, 59]  # C, E, G, B

    # Test minor major 7th
    cminmaj7 = parser.apply("c4`mmaj7")
    note_values = [int(note) for note in cminmaj7]
    assert note_values == [48, 51, 55, 59]  # C, Eb, G, B

    # Test dominant 9th
    c9 = parser.apply("c4`9")
    note_values = [int(note) for note in c9]
    assert note_values == [48, 52, 55, 58, 62]  # C, E, G, Bb, D

    # Test diminished 7th
    cdim7 = parser.apply("c4`dim7")
    note_values = [int(note) for note in cdim7]
    assert note_values == [48, 51, 54, 57]  # C, Eb, Gb, A
