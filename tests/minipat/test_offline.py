"""Tests for minipat.offline module."""

from __future__ import annotations

from fractions import Fraction

from minipat.combinators import tab_data_stream
from minipat.messages import MidiAttrs, NoteKey, TabFretKey, TabStringKey
from minipat.offline import render_lilypond
from minipat.time import CycleArc, CycleTime


def test_arpeggio_pattern_note_events() -> None:
    """Test that the arpeggio pattern from lilypond_tab.py emits expected note events.

    The pattern '[6#3 5#2 4#0 3#0 2#0 1#3 #320003 _]/2' over arc (0, 2) should produce:
    - Individual arpeggio notes stretched across 2 cycles: G(43), B(47), D(50), G(55), B(59), G(67)
    - Full chord at 1.5 with all 6 notes simultaneously
    - Pattern is slowed down by factor of 2, taking 2 cycles to complete
    """
    # Create the arpeggio pattern from the lilypond_tab.py example
    stream = tab_data_stream("[6#3 5#2 4#0 3#0 2#0 1#3 #320003 _]/2")
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(2)))

    # Process the stream using the same logic as render_lilypond
    from minipat.combinators import TabBinder
    from minipat.ev import EvHeap, ev_heap_empty
    from minipat.stream import Stream
    from minipat.time import CycleSpan

    tab_binder = TabBinder()
    all_events: EvHeap[MidiAttrs] = ev_heap_empty()

    # Convert TabData to MidiAttrs events
    for span, tab_ev in stream.unstream(arc):
        midi_attrs_pat = tab_binder.apply(tab_ev.val)
        unit_cycle = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))

        for inner_span, inner_ev in Stream.pat(midi_attrs_pat).unstream(unit_cycle):
            # Apply timing adjustment as in render_lilypond
            span_duration = span.active.end - span.active.start
            adjusted_start = span.active.start + (
                inner_span.active.start * span_duration
            )
            adjusted_end = span.active.start + (inner_span.active.end * span_duration)

            adjusted_span = CycleSpan(
                inner_span.whole,
                CycleArc(CycleTime(adjusted_start), CycleTime(adjusted_end)),
            )
            all_events = all_events.insert(adjusted_span, inner_ev)

    # Extract note events for verification
    note_key = NoteKey()
    string_key = TabStringKey()
    fret_key = TabFretKey()

    note_events = []
    for span, ev in all_events:
        note = ev.val.get(note_key)
        string_num = ev.val.get(string_key)
        fret = ev.val.get(fret_key)
        start_time = float(span.active.start)

        note_events.append(
            {
                "note": note,
                "string": string_num,
                "fret": fret,
                "start": start_time,
            }
        )

    # Sort by start time for easier verification
    note_events.sort(key=lambda x: (x["start"], x["note"]))

    # Expected notes based on guitar tuning and fret positions:
    # String 6 (E) + 3 frets = G (MIDI 43)
    # String 5 (A) + 2 frets = B (MIDI 47)
    # String 4 (D) + 0 frets = D (MIDI 50)
    # String 3 (G) + 0 frets = G (MIDI 55)
    # String 2 (B) + 0 frets = B (MIDI 59)
    # String 1 (E) + 3 frets = G (MIDI 67)

    expected_notes = [
        # Pattern stretched over 2 cycles with /2 operator
        {"note": 43, "string": 6, "fret": 3, "start": 0.0},  # G at 0.0-0.25
        {"note": 47, "string": 5, "fret": 2, "start": 0.25},  # B at 0.25-0.5
        {"note": 50, "string": 4, "fret": 0, "start": 0.5},  # D at 0.5-0.75
        {"note": 55, "string": 3, "fret": 0, "start": 0.75},  # G at 0.75-1.0
        {"note": 59, "string": 2, "fret": 0, "start": 1.0},  # B at 1.0-1.25
        {"note": 67, "string": 1, "fret": 3, "start": 1.25},  # G at 1.25-1.5
        # Full chord at 1.5 (all notes simultaneously)
        {"note": 43, "string": 6, "fret": 3, "start": 1.5},  # G
        {"note": 47, "string": 5, "fret": 2, "start": 1.5},  # B
        {"note": 50, "string": 4, "fret": 0, "start": 1.5},  # D
        {"note": 55, "string": 3, "fret": 0, "start": 1.5},  # G
        {"note": 59, "string": 2, "fret": 0, "start": 1.5},  # B
        {"note": 67, "string": 1, "fret": 3, "start": 1.5},  # G
    ]

    # Verify we have the expected number of notes
    assert len(note_events) == len(expected_notes), (
        f"Expected {len(expected_notes)} notes, got {len(note_events)}"
    )

    # Verify each note matches expectations
    for i, (actual, expected) in enumerate(zip(note_events, expected_notes)):
        assert actual["note"] == expected["note"], (
            f"Note {i}: expected MIDI {expected['note']}, got {actual['note']}"
        )
        assert actual["string"] == expected["string"], (
            f"Note {i}: expected string {expected['string']}, got {actual['string']}"
        )
        assert actual["fret"] == expected["fret"], (
            f"Note {i}: expected fret {expected['fret']}, got {actual['fret']}"
        )
        assert abs(actual["start"] - expected["start"]) < 0.001, (
            f"Note {i}: expected start {expected['start']}, got {actual['start']}"
        )


def test_render_lilypond_arpeggio_integration() -> None:
    """Integration test that render_lilypond processes the arpeggio pattern correctly."""
    import tempfile
    from pathlib import Path

    stream = tab_data_stream("[6#3 5#2 4#0 3#0 2#0 1#3 #320003 _]/2")
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(2)))

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Generate only LilyPond source
        result = render_lilypond(
            arc=arc,
            tab_stream=stream,
            name="test_arpeggio",
            directory=temp_path,
            pdf=False,
            svg=False,
        )

        # Verify the result contains the expected LilyPond source file
        assert "ly" in result
        assert result["ly"].exists()

        # Read the LilyPond content
        ly_content = result["ly"].read_text()

        # Verify it contains the expected notes (basic smoke test)
        assert "g,4\\6" in ly_content  # Low G on 6th string
        assert "b,4\\5" in ly_content  # B on 5th string
        assert "d4\\4" in ly_content  # D on 4th string
        assert "g4\\3" in ly_content  # G on 3rd string
        assert "b4\\2" in ly_content  # B on 2nd string
        assert "g'4\\1" in ly_content  # High G on 1st string

        # Verify it contains the chord notation
        assert "<" in ly_content and ">2" in ly_content  # Chord brackets with half note

        # Verify the pattern appears once (stretched across 2 cycles, not repeated)
        g_count = ly_content.count("g,4\\6")
        assert g_count == 1, f"Expected 1 occurrence of 'g,4\\6', found {g_count}"
