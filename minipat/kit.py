"""Kit functionality for the minipat pattern system.

This module provides hit pattern mapping from string identifiers to MIDI parameters.
It supports kit notation like "bd" for bass drum, "sd" for snare drum, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, override

from minipat.messages import (
    ChannelField,
    NoteField,
    VelocityField,
)
from minipat.types import (
    Channel,
    Note,
    Velocity,
)
from spiny.arrow import BiArrow
from spiny.map import PMap

Kit = PMap[str, "Sound"]


@dataclass(frozen=True)
class Sound:
    """A sound with MIDI parameters.

    Args:
        note: MIDI note number for the sound
        velocity: Optional default velocity for the sound
        channel: Optional default channel for the sound
    """

    note: Note
    velocity: Optional[Velocity] = None
    channel: Optional[Channel] = None


# Default drum kit with common drum sounds
# Based on General MIDI drum map (Channel 10, notes 35-81)
DEFAULT_KIT: Kit = PMap.mk(
    [
        # Kick drums
        ("bd", Sound(Note(36))),  # Bass Drum 1
        ("bd2", Sound(Note(35))),  # Bass Drum 2
        # Snare drums
        ("sd", Sound(Note(38))),  # Snare Drum 1
        ("sd2", Sound(Note(40))),  # Snare Drum 2
        # Hi-hats
        ("hh", Sound(Note(42))),  # Closed Hi-hat
        ("hho", Sound(Note(46))),  # Open Hi-hat
        ("hhp", Sound(Note(44))),  # Pedal Hi-hat
        # Cymbals
        ("cy", Sound(Note(49))),  # Crash Cymbal 1
        ("cy2", Sound(Note(57))),  # Crash Cymbal 2
        ("rd", Sound(Note(51))),  # Ride Cymbal 1
        ("rd2", Sound(Note(59))),  # Ride Cymbal 2
        ("sp", Sound(Note(55))),  # Splash Cymbal
        ("cb", Sound(Note(56))),  # Cowbell
        # Toms
        ("lt", Sound(Note(43))),  # Low Tom
        ("mt", Sound(Note(47))),  # Mid Tom
        ("ht", Sound(Note(50))),  # High Tom
        ("ft", Sound(Note(41))),  # Floor Tom
        # Percussion
        ("cl", Sound(Note(39))),  # Hand Clap
        ("ta", Sound(Note(54))),  # Tambourine
        ("sh", Sound(Note(69))),  # Shaker
        ("ma", Sound(Note(70))),  # Maracas
        ("ws", Sound(Note(76))),  # Woodblock
        # Latin percussion
        ("co", Sound(Note(62))),  # Conga
        ("bo", Sound(Note(61))),  # Bongo
        ("ti", Sound(Note(65))),  # Timbale
        ("ag", Sound(Note(67))),  # Agogo
        # Electronic sounds (using higher notes)
        ("noi", Sound(Note(31))),  # Noise (using stick)
    ]
)


class DrumSoundElemParser(BiArrow[str, Note]):
    """Parser for drum sound identifiers to MIDI notes.

    Converts drum sound string identifiers (like "bd", "sd", "hh") to
    MIDI note numbers using a provided drum kit.

    Examples:
        "bd"  -> Note(36)  # Bass drum
        "sd"  -> Note(38)  # Snare drum
        "hh"  -> Note(42)  # Hi-hat
        "cy"  -> Note(49)  # Crash cymbal

    Raises:
        ValueError: If the drum sound identifier is not found in the kit
    """

    def __init__(self, kit: Kit) -> None:
        """Initialize the parser with a kit.

        Args:
            kit: The kit to use for parsing
        """
        self._kit = kit

    @override
    def apply(self, value: str) -> Note:
        """Parse a drum sound identifier to a MIDI note.

        Args:
            value: String identifier for the drum sound

        Returns:
            MIDI note number for the drum sound

        Raises:
            ValueError: If the identifier is not found in the kit
        """
        sound = self._kit.lookup(value)
        if sound is None:
            available_sounds = ", ".join(sorted(self._kit.keys()))
            raise ValueError(
                f"Unknown drum sound '{value}'. Available sounds: {available_sounds}"
            )
        return sound.note

    @override
    def rev_apply(self, value: Note) -> str:
        """Convert a MIDI note back to a drum sound identifier.

        Args:
            value: MIDI note number

        Returns:
            String identifier for the drum sound, or the note number as string
            if no matching identifier is found
        """
        for identifier, sound in self._kit.items():
            if sound.note == value:
                return identifier
        # Fallback to note number if no identifier found
        return str(int(value))


def add_hit(
    note: int, velocity: Optional[int] = None, channel: Optional[int] = None
) -> Sound:
    """Create a Sound object with validation.

    Args:
        note: MIDI note number (0-127)
        velocity: Optional default velocity (0-127)
        channel: Optional default channel (0-15)

    Returns:
        A Sound object

    Raises:
        ValueError: If note, velocity, or channel values are out of range

    Example:
        hit = add_hit(49, 100, 9)  # Create a crash cymbal hit
        n.kit = n.kit.put("crash2", hit)
    """
    midi_note = NoteField.mk(note)
    midi_velocity = None if velocity is None else VelocityField.mk(velocity)
    midi_channel = None if channel is None else ChannelField.mk(channel)
    return Sound(midi_note, midi_velocity, midi_channel)
