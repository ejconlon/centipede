"""Musical scale definitions and note classification for PushPluck.

This module provides classes and constants for working with musical scales,
note names, and scale classification. It includes a comprehensive set of
scales from various musical traditions and utilities for note manipulation.
"""

from dataclasses import dataclass
from enum import Enum, unique
from typing import Dict, List, Set, Tuple


@unique
class NoteName(Enum):
    """Enumeration of the twelve chromatic note names.

    Values correspond to semitone offsets from C within an octave.
    Uses flat notation for accidentals (Db, Eb, Gb, Ab, Bb).
    """

    C = 0  # C natural
    Db = 1  # D flat
    D = 2  # D natural
    Eb = 3  # E flat
    E = 4  # E natural
    F = 5  # F natural
    Gb = 6  # G flat
    G = 7  # G natural
    Ab = 8  # A flat
    A = 9  # A natural
    Bb = 10  # B flat
    B = 11  # B natural

    def add_steps(self, steps: int) -> "NoteName":
        """Add semitone steps to this note name.

        Args:
            steps: Number of semitones to add (can be negative).

        Returns:
            The resulting note name after adding the steps.
        """
        v = self.value + steps
        while v < 0:
            v += MAX_NOTES
        while v >= MAX_NOTES:
            v -= MAX_NOTES
        return NOTE_LOOKUP[v]


MAX_NOTES = 12
"""Number of distinct note names in the chromatic scale."""

CIRCLE_OF_FIFTHS = [
    NoteName[x]
    for x in ["C", "G", "D", "A", "E", "B", "Gb", "Db", "Ab", "Eb", "Bb", "F"]
]
"""The circle of fifths as a list of note names.

Starting from C and progressing by perfect fifths, wrapping around
the chromatic circle. Useful for key signature calculations.
"""


def _build_note_lookup() -> Dict[int, NoteName]:
    """Build a lookup table from semitone values to note names.

    Returns:
        Dictionary mapping integers (0-11) to NoteName enum values.
    """
    d: Dict[int, NoteName] = {}
    for n in NoteName:
        d[n.value] = n
    assert len(d) == MAX_NOTES
    return d


NOTE_LOOKUP = _build_note_lookup()
"""Lookup table from semitone offset (0-11) to NoteName."""


def name_and_octave_from_note(note: int) -> Tuple[NoteName, int]:
    """Extract note name and octave from a MIDI note number.

    Args:
        note: MIDI note number (0-127).

    Returns:
        Tuple of (note_name, octave) where octave follows scientific
        pitch notation (middle C is C4, octave 4).
    """
    offset = note % 12
    name = NOTE_LOOKUP[offset]
    octave = note // 12 - 2
    return name, octave


class ScaleClassifier:
    """Classifies notes relative to a specific scale and root.

    This class provides methods to determine whether a note is the
    root of the scale or a member of the scale, useful for coloring
    and visualization in the fretboard interface.
    """

    def __init__(self, root: NoteName, members: Set[NoteName]) -> None:
        """Initialize the classifier with a root and scale members.

        Args:
            root: The root note of the scale.
            members: Set of all notes that are members of this scale.
        """
        self._root = root
        self._members = members

    def is_root(self, name: NoteName) -> bool:
        """Check if a note name is the root of this scale.

        Args:
            name: The note name to check.

        Returns:
            True if this is the root note of the scale.
        """
        return self._root == name

    def is_member(self, name: NoteName) -> bool:
        """Check if a note name is a member of this scale.

        Args:
            name: The note name to check.

        Returns:
            True if this note is a member of the scale.
        """
        return name in self._members


@dataclass(frozen=True)
class Scale:
    """Represents a musical scale with its name and interval pattern.

    A scale is defined by its name and a list of semitone intervals
    from the root note. The intervals list always starts with 0 (the root)
    and contains the semitone offsets for all notes in the scale.
    """

    name: str  # Human-readable name of the scale
    """The human-readable name of this musical scale."""
    intervals: List[int]  # Semitone intervals from root (always starts with 0)
    """List of semitone intervals from the root note.

    Always starts with 0 (the root) and contains ascending intervals
    for all notes in the scale. For example, major scale: [0, 2, 4, 5, 7, 9, 11].
    """

    def to_classifier(self, root: NoteName) -> ScaleClassifier:
        """Create a scale classifier for this scale with the given root.

        Args:
            root: The root note for this scale instance.

        Returns:
            A ScaleClassifier that can determine note relationships
            to this scale.

        Raises:
            AssertionError: If the scale definition is invalid (intervals
                           not sorted, out of range, or duplicate notes).
        """
        members: Set[NoteName] = set()
        assert self.intervals[0] == 0
        last_steps = -1
        for steps in self.intervals:
            assert steps >= 0 and steps < MAX_NOTES
            assert steps > last_steps
            last_steps = steps
            note = root.add_steps(steps)
            assert note not in members
            members.add(note)
        return ScaleClassifier(root, members)


SCALES: List[Scale] = [
    Scale("Major", [0, 2, 4, 5, 7, 9, 11]),
    Scale("Minor", [0, 2, 3, 5, 7, 8, 10]),
    Scale("Dorian", [0, 2, 3, 5, 7, 9, 10]),
    Scale("Mixolydian", [0, 2, 4, 5, 7, 9, 10]),
    Scale("Lydian", [0, 2, 4, 6, 7, 9, 11]),
    Scale("Phrygian", [0, 1, 3, 5, 7, 8, 10]),
    Scale("Locrian", [0, 1, 3, 4, 7, 8, 10]),
    Scale("Diminished", [0, 1, 3, 4, 6, 7, 9, 10]),
    Scale("Whole-half", [0, 2, 3, 5, 6, 8, 9, 11]),
    Scale("Whole Tone", [0, 2, 4, 6, 8, 10]),
    Scale("Minor Blues", [0, 3, 5, 6, 7, 10]),
    Scale("Minor Pentatonic", [0, 3, 5, 7, 10]),
    Scale("Major Pentatonic", [0, 2, 4, 7, 9]),
    Scale("Harmonic Minor", [0, 2, 3, 5, 7, 8, 11]),
    Scale("Melodic Minor", [0, 2, 3, 5, 7, 9, 11]),
    Scale("Super Locrian", [0, 1, 3, 4, 6, 8, 10]),
    Scale("Bhairav", [0, 1, 4, 5, 7, 8, 11]),
    Scale("Hungarian Minor", [0, 2, 3, 6, 7, 8, 11]),
    Scale("Minor Gypsy", [0, 1, 4, 5, 7, 8, 10]),
    Scale("Hirojoshi", [0, 2, 3, 7, 8]),
    Scale("In-Sen", [0, 1, 5, 7, 10]),
    Scale("Iwato", [0, 1, 5, 6, 10]),
    Scale("Kumoi", [0, 2, 3, 7, 9]),
    Scale("Pelog", [0, 1, 3, 4, 7, 8]),
    Scale("Spanish", [0, 1, 3, 4, 5, 6, 8, 10]),
    Scale("Chromatic", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
]
"""Comprehensive list of musical scales from various traditions.

Includes Western classical modes, blues scales, pentatonic scales,
harmonic and melodic minor variants, diminished scales, world music
scales, and the complete chromatic scale.
"""

SCALE_LOOKUP: Dict[str, Scale] = {s.name: s for s in SCALES}
"""Dictionary lookup from scale name to Scale object.

Provides efficient access to scales by name for configuration
and menu systems.
"""
