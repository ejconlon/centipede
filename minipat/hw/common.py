"""Common utilities for hardware modules.

This module provides shared functionality for working with preset databases
and MIDI message generation across different hardware modules like fluid and sc88.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, TypeVar

from minipat.messages import (
    Channel,
    ControlMessage,
    ControlNum,
    ControlVal,
    MidiBundle,
    MidiMessage,
    Program,
    ProgramMessage,
    midi_bundle_concat,
)
from spiny.seq import PSeq

# Generic type for preset tuple formats
T = TypeVar("T")


# =============================================================================
# Preset/Instrument Abstraction
# =============================================================================


@dataclass(frozen=True)
class Inst:
    """Represents a hardware instrument with its preset.

    Attributes:
        ix: Index in the instrument list
        preset: Preset object containing bank, program, and name
    """

    ix: int
    preset: Preset


@dataclass(frozen=True)
class Preset:
    """Represents a hardware preset (bank + program + name combination).

    Attributes:
        bank: MIDI bank number
        program: MIDI program number
        name: Display name of the preset
    """

    bank: ControlVal
    program: Program
    name: str


class Hardware:
    """Hardware abstraction for working with preset databases.

    This class provides a unified interface for working with hardware modules
    that have preset databases, supporting both simple preset lookups and
    more complex instrument abstractions.
    """

    def __init__(self, presets: Iterable[Preset]):
        """Initialize hardware with preset database.

        Args:
            presets: Sequence of (bank, program, name) tuples defining presets
        """
        self._anno_presets = [
            (_process_name_for_search(preset.name), preset) for preset in presets
        ]

    @staticmethod
    def mk(*tups: tuple[int, int, str]) -> Hardware:
        return Hardware(Preset(ControlVal(b), Program(p), n) for (b, p, n) in tups)

    def get_preset_by_ids(self, bank: ControlVal, program: Program) -> Optional[Preset]:
        """Get preset name by bank and program number.

        Args:
            bank: Bank number
            program: Program number

        Returns:
            Preset if found, None otherwise
        """
        for _, preset in self._anno_presets:
            if preset.bank == bank and preset.program == program:
                return preset
        return None

    def find_preset_by_name(self, name: str) -> Optional[Preset]:
        """Find bank and program number by name.

        Args:
            name: Name of the preset

        Returns:
            Preset if found, None otherwise
        """
        for _, preset in self._anno_presets:
            if preset.name == name:
                return preset
        return None

    def list_presets_by_bank(self, bank: Optional[ControlVal] = None) -> list[Preset]:
        """Get list of presets, optionally filtered by bank.

        Args:
            bank: Optional bank number to filter by

        Returns:
            List of Pesets
        """
        if bank is None:
            return [preset for _, preset in self._anno_presets]
        return [preset for _, preset in self._anno_presets if preset.bank == bank]

    def search_presets_by_name(
        self, query: str, bank: Optional[ControlVal] = None
    ) -> list[Preset]:
        """Search for presets containing the query string.

        Args:
            query: String to search for (case-insensitive)
            bank: Optional bank number to filter by

        Returns:
            List of Presets matching the query
        """
        search_query = _process_name_for_search(query)
        results = []
        for search_name, preset in self._anno_presets:
            if bank is not None and preset.bank != bank:
                continue
            if search_query in search_name:
                results.append(preset)
        return results

    def get_available_banks(self) -> list[ControlVal]:
        """Get list of available bank numbers.

        Returns:
            Sorted list of unique bank numbers
        """
        banks = set(preset.bank for _, preset in self._anno_presets)
        return sorted(banks)

    def find_instrument(self, fragment: str) -> Optional[Inst]:
        """Find an instrument by name fragment.

        Searches through the preset list for the first preset
        whose name contains the given fragment (case-insensitive).

        Args:
            fragment: Text fragment to search for in preset names

        Returns:
            The first matching instrument, or None if not found
        """
        search_fragment = _process_name_for_search(fragment)
        for i, (search_name, preset) in enumerate(self._anno_presets):
            if search_fragment in search_name:
                return Inst(i, preset)
        return None

    def first_instrument(self) -> Optional[Inst]:
        """Get the first instrument in the list.

        Returns:
            First instrument, or None if no presets available
        """
        if not self._anno_presets:
            return None
        return Inst(0, self._anno_presets[0][1])

    def next_instrument(self, inst: Inst) -> Optional[Inst]:
        """Get the next instrument in the list, wrapping to the beginning.

        Args:
            inst: Current instrument

        Returns:
            Next instrument, or None if no presets available
        """
        if not self._anno_presets or inst.ix >= len(self._anno_presets) - 1:
            return self.first_instrument()
        else:
            j = inst.ix + 1
            return Inst(j, self._anno_presets[j][1])


def _process_name_for_search(name: str) -> str:
    """Process name for searching by normalizing case and punctuation.

    Converts to lowercase and replaces hyphens/underscores with spaces.

    Args:
        name: Name to process

    Returns:
        Processed name
    """
    return name.lower().replace("-", " ").replace("_", " ")


# =============================================================================
# MIDI Message Generation
# =============================================================================


def set_sound(chan: Channel, bank: ControlVal, program: Program) -> MidiBundle:
    """Set a complete sound on the specified channel.

    This function sends the sequence of messages needed to properly
    set an instrument on MIDI devices that support bank selection:
    1. Bank Select MSB (CC 0) with bank
    2. Bank Select LSB (CC 32) with 0
    3. Program Change with program

    Args:
        chan: MIDI channel (0-15)
        bank: MIDI bank number (0-127)
        program: MIDI program number (0-127)

    Returns:
        Bundle containing the control changes and program change
    """
    return PSeq.mk(
        [
            ControlMessage(chan, ControlNum(0), bank),  # Bank Select MSB
            ControlMessage(chan, ControlNum(32), ControlVal(0)),  # Bank Select LSB
            ProgramMessage(chan, program),  # Program Change
        ]
    )


def set_level(chan: Channel, level: ControlVal) -> MidiMessage:
    """Set the volume level for a channel.

    Args:
        chan: MIDI channel (0-15)
        level: Volume level (0-127)

    Returns:
        A volume control change message
    """
    return ControlMessage(chan, ControlNum(7), level)


def set_pan(chan: Channel, pan: ControlVal) -> MidiMessage:
    """Set the pan position for a channel.

    Args:
        chan: MIDI channel (0-15)
        pan: Pan position (0=left, 64=center, 127=right)

    Returns:
        A pan control change message
    """
    return ControlMessage(chan, ControlNum(10), pan)


def set_reverb(chan: Channel, reverb: ControlVal) -> MidiMessage:
    """Set the reverb level for a channel.

    Args:
        chan: MIDI channel (0-15)
        reverb: Reverb level (0-127)

    Returns:
        A reverb control change message
    """
    return ControlMessage(chan, ControlNum(91), reverb)


def set_chorus(chan: Channel, chorus: ControlVal) -> MidiMessage:
    """Set the chorus level for a channel.

    Args:
        chan: MIDI channel (0-15)
        chorus: Chorus level (0-127)

    Returns:
        A chorus control change message
    """
    return ControlMessage(chan, ControlNum(93), chorus)


def all_sounds_off(chan: Channel) -> MidiMessage:
    """Turn off all sounds on a channel immediately.

    Args:
        chan: MIDI channel (0-15)

    Returns:
        An all sounds off message
    """
    return ControlMessage(chan, ControlNum(120), ControlVal(0))


def all_notes_off(chan: Channel) -> MidiMessage:
    """Turn off all non-sustained notes on a channel.

    This respects sustain pedal settings - sustained notes will
    continue until the sustain pedal is released.

    Args:
        chan: MIDI channel (0-15)

    Returns:
        An all notes off message
    """
    return ControlMessage(chan, ControlNum(123), ControlVal(0))


def reinit(chan: Channel) -> MidiBundle:
    """Reinitialize a channel to default settings.

    Sets the channel to:
    - Sound: Piano (program 0, variation 0)
    - Level: 100
    - Pan: Center (64)
    - Reverb: 40
    - Chorus: 0

    Args:
        chan: MIDI channel (0-15)

    Returns:
        Bundle containing all initialization messages
    """
    left = set_sound(chan, ControlVal(0), Program(0))  # Piano (bank=0, program=0)
    right = PSeq.mk(
        [
            set_level(chan, ControlVal(100)),  # Full volume
            set_pan(chan, ControlVal(64)),  # Center pan
            set_reverb(chan, ControlVal(40)),  # Some reverb
            set_chorus(chan, ControlVal(0)),  # No chorus
        ]
    )
    return midi_bundle_concat(left, right)
