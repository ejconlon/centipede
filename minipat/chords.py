"""Chord definitions and utilities for minipat."""

from __future__ import annotations

from typing import NewType, Optional, cast

from minipat.types import ChordData, Note
from spiny import PSeq

Chord = NewType("Chord", str)


_CHORD_MAP = cast(
    dict[str, Chord],
    {
        # Major variants
        "major": "maj",
        "maj": "maj",
        "M": "maj",
        # Augmented
        "aug": "aug",
        "plus": "aug",
        "sharp5": "aug",
        # Sixth chords
        "six": "6",
        "6": "6",
        "69": "69",
        "sixNine": "69",
        "six9": "69",
        "sixby9": "69",
        "6by9": "69",
        # Major seventh and extensions
        "major7": "maj7",
        "maj7": "maj7",
        "M7": "maj7",
        "major9": "maj9",
        "maj9": "maj9",
        "M9": "maj9",
        "add9": "add9",
        "major11": "maj11",
        "maj11": "maj11",
        "M11": "maj11",
        "add11": "add11",
        "major13": "maj13",
        "maj13": "maj13",
        "M13": "maj13",
        "add13": "add13",
        # Dominant chords
        "7": "dom7",
        "dom7": "dom7",
        "dom9": "dom9",
        "dom11": "dom11",
        "dom13": "dom13",
        "sevenSharp11": "7s11",
        "7s11": "7s11",
        "sevenFlat5": "7f5",
        "7f5": "7f5",
        "sevenSharp5": "7s5",
        "7s5": "7s5",
        "sevenFlat9": "7f9",
        "7f9": "7f9",
        "sevenSharp9": "7s9",
        "7s9": "7s9",
        "nine": "9",
        "9": "9",
        "eleven": "11",
        "11": "11",
        "thirteen": "13",
        "13": "13",
        # Minor chords
        "minor": "min",
        "min": "min",
        "m": "min",
        "diminished": "dim",
        "dim": "dim",
        "minorSharp5": "mins5",
        "mins5": "mins5",
        "msharp5": "mins5",
        "mS5": "mins5",
        "minor6": "min6",
        "min6": "min6",
        "m6": "min6",
        "minorSixNine": "min69",
        "minor69": "min69",
        "min69": "min69",
        "minSixNine": "min69",
        "m69": "min69",
        "mSixNine": "min69",
        "m6by9": "min69",
        "minor7flat5": "min7f5",
        "minor7f5": "min7f5",
        "min7flat5": "min7f5",
        "min7f5": "min7f5",
        "m7flat5": "min7f5",
        "m7f5": "min7f5",
        "minor7": "min7",
        "min7": "min7",
        "m7": "min7",
        "minor7sharp5": "min7s5",
        "minor7s5": "min7s5",
        "min7sharp5": "min7s5",
        "min7s5": "min7s5",
        "m7sharp5": "min7s5",
        "m7s5": "min7s5",
        "minor7flat9": "min7f9",
        "minor7f9": "min7f9",
        "min7flat9": "min7f9",
        "min7f9": "min7f9",
        "m7flat9": "min7f9",
        "m7f9": "min7f9",
        "minor7sharp9": "min7s9",
        "minor7s9": "min7s9",
        "min7sharp9": "min7s9",
        "min7s9": "min7s9",
        "m7sharp9": "min7s9",
        "m7s9": "min7s9",
        "diminished7": "dim7",
        "dim7": "dim7",
        "minor9": "min9",
        "min9": "min9",
        "m9": "min9",
        "minor11": "min11",
        "min11": "min11",
        "m11": "min11",
        "minor13": "min13",
        "min13": "min13",
        "m13": "min13",
        "minorMajor7": "mmaj7",
        "minMaj7": "mmaj7",
        "mmaj7": "mmaj7",
        # Other chords
        "one": "1",
        "1": "1",
        "five": "5",
        "5": "5",
        "sus2": "sus2",
        "sus4": "sus4",
        "sevenSus2": "7sus2",
        "7sus2": "7sus2",
        "sevenSus4": "7sus4",
        "7sus4": "7sus4",
        "nineSus4": "9sus4",
        "ninesus4": "9sus4",
        "9sus4": "9sus4",
    },
)


# Chord note intervals (semitones from root)
_CHORD_INTERVALS = cast(
    dict[Chord, list[int]],
    {
        # Major chords
        "maj": [0, 4, 7],
        "aug": [0, 4, 8],
        "6": [0, 4, 7, 9],
        "69": [0, 4, 7, 9, 14],
        "maj7": [0, 4, 7, 11],
        "maj9": [0, 4, 7, 11, 14],
        "add9": [0, 4, 7, 14],
        "maj11": [0, 4, 7, 11, 14, 17],
        "add11": [0, 4, 7, 17],
        "maj13": [0, 4, 7, 11, 14, 21],
        "add13": [0, 4, 7, 21],
        # Dominant chords
        "dom7": [0, 4, 7, 10],
        "dom9": [0, 4, 7, 14],
        "dom11": [0, 4, 7, 17],
        "dom13": [0, 4, 7, 21],
        "7s11": [0, 4, 6, 10],
        "7f5": [0, 4, 6, 10],
        "7s5": [0, 4, 8, 10],
        "7f9": [0, 4, 7, 10, 13],
        "7s9": [0, 4, 7, 10, 15],
        "9": [0, 4, 7, 10, 14],
        "11": [0, 4, 7, 10, 14, 17],
        "13": [0, 4, 7, 10, 14, 17, 21],
        # Minor chords
        "min": [0, 3, 7],
        "dim": [0, 3, 6],
        "mins5": [0, 3, 8],
        "min6": [0, 3, 7, 9],
        "min69": [0, 3, 7, 9, 14],
        "min7f5": [0, 3, 6, 10],
        "min7": [0, 3, 7, 10],
        "min7s5": [0, 3, 8, 10],
        "min7f9": [0, 3, 7, 10, 13],
        "min7s9": [0, 3, 7, 10, 15],
        "dim7": [0, 3, 6, 9],
        "min9": [0, 3, 7, 10, 14],
        "min11": [0, 3, 7, 10, 14, 17],
        "min13": [0, 3, 7, 10, 14, 17, 21],
        "mmaj7": [0, 3, 7, 11],
        # Other chords
        "1": [0],
        "5": [0, 7],
        "sus2": [0, 2, 7],
        "sus4": [0, 5, 7],
        "7sus2": [0, 2, 7, 10],
        "7sus4": [0, 5, 7, 10],
        "9sus4": [0, 5, 7, 10, 14],
    },
)


def parse_chord_name(name: str) -> Optional[Chord]:
    """Parse a chord name string into a canonical chord name.

    Args:
        name: The chord name string to parse (case-insensitive)

    Returns:
        Canonical chord name if found, None otherwise
    """
    # Try exact match first
    if name in _CHORD_MAP:
        return _CHORD_MAP[name]

    # Then try lowercase match
    return _CHORD_MAP.get(name.lower())


def get_chord_intervals(chord: Chord) -> PSeq[int]:
    """Get the semitone intervals for a chord.

    Args:
        chord: The chord name

    Returns:
        PSeq of semitone intervals from the root
    """
    return PSeq.mk(_CHORD_INTERVALS[chord])


def chord_to_notes(root: Note, chord: Chord) -> PSeq[Note]:
    """Convert a chord to Note objects.

    Only includes notes within valid MIDI range (0-127).

    Args:
        root: Note object for the root
        chord: The chord type

    Returns:
        PSeq of Note objects for the chord (filtered to valid MIDI range)
    """
    intervals = _CHORD_INTERVALS[chord]
    notes = []
    for interval in intervals:
        midi_value = root + interval
        if 0 <= midi_value <= 127:
            notes.append(Note(midi_value))
    return PSeq.mk(notes)


def apply_inversion(notes: PSeq[Note], inversion: int) -> PSeq[Note]:
    """Apply inversion to chord notes.

    An inversion rotates the notes of a chord upward by octaves.
    For example:
    - inv0 (root position): [C, E, G]
    - inv1 (first inversion): [E, G, C'] where C' is C up an octave
    - inv2 (second inversion): [G, C', E'] where both C and E are up an octave

    Args:
        notes: The chord notes to invert
        inversion: The inversion number (0 = root position, 1 = first, 2 = second, etc.)

    Returns:
        PSeq of inverted Note objects (filtered to valid MIDI range)
    """
    if inversion == 0 or len(notes) == 0:
        return notes

    # Clamp inversion to valid range
    actual_inversion = min(inversion, len(notes))

    # Split at the inversion point
    to_rotate, rest = notes.split_at(actual_inversion)

    # Move rotated notes up an octave (if within MIDI range)
    rotated_notes: PSeq[Note] = PSeq.empty()
    for note in to_rotate:
        octave_up = note + 12
        if octave_up <= 127:
            rotated_notes = rotated_notes.snoc(Note(octave_up))

    # Combine: rest first, then octave-shifted rotated notes
    return rest.concat(rotated_notes)


def apply_drop_voicing(notes: PSeq[Note], drop: int) -> PSeq[Note]:
    """Apply drop voicing to chord notes.

    Drop voicing takes specific notes from the top of a close voicing
    and drops them down an octave. For example:
    - drop2: Drop the second note from the top down an octave
    - drop3: Drop the third note from the top down an octave
    - drop2+4: Drop both the second and fourth notes from top

    Args:
        notes: The chord notes to apply drop voicing to
        drop: The drop number (2 = drop2, 3 = drop3, etc.)

    Returns:
        PSeq of notes with drop voicing applied (filtered to valid MIDI range)
    """
    if drop <= 0 or len(notes) <= 1:
        return notes

    notes_list = list(notes)
    dropped_notes = notes_list.copy()

    # Find the position to drop (counting from the top, which is the end)
    # drop2 means drop the 2nd note from the top
    position_from_end = drop
    if position_from_end <= len(notes_list):
        # The actual index (0-based from the start)
        idx = len(notes_list) - position_from_end

        # Drop this note down an octave if possible
        midi_value = notes_list[idx] - 12
        if midi_value >= 0:
            dropped_notes[idx] = Note(midi_value)

    return PSeq.mk(dropped_notes)


def get_all_chords() -> set[Chord]:
    """Get all available chord types.

    Returns:
        Set of all Chord values that can be used with chord_to_notes
    """
    return set(Chord(chord) for chord in _CHORD_INTERVALS.keys())


def chord_data_to_notes(chord_data: ChordData) -> PSeq[Note]:
    """Convert ChordData to a sequence of Note objects.

    Applies the chord type and any voicing modifiers to generate
    the actual notes of the chord.

    Args:
        chord_data: The ChordData to render

    Returns:
        PSeq of Note objects for the chord (filtered to valid MIDI range)
    """
    # Get base chord notes
    notes = chord_to_notes(chord_data.root_note, Chord(chord_data.chord_name))

    # Apply voicing modifiers in order
    for modifier_type, modifier_value in chord_data.modifiers:
        if modifier_type == "inv":
            notes = apply_inversion(notes, modifier_value)
        elif modifier_type == "drop":
            notes = apply_drop_voicing(notes, modifier_value)

    return notes
