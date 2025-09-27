"""Tablature parsing and utilities for minipat."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from minipat.types import Note, TabData
from spiny import PSeq


class TabInst(Enum):
    """Tablature instrument types."""

    StandardGuitar = auto()
    DropDGuitar = auto()
    OpenGGuitar = auto()
    OpenDGuitar = auto()
    DadgadGuitar = auto()
    StandardBass = auto()
    FiveStringBass = auto()
    Ukulele = auto()
    Mandolin = auto()
    Banjo = auto()


class StringOrder(Enum):
    """String numbering/ordering conventions."""

    LowToHigh = auto()  # 1=lowest pitch string
    HighToLow = auto()  # 1=highest pitch string (standard for guitar)


@dataclass(frozen=True)
class TabConfig:
    """Configuration for a tablature instrument."""

    tuning: PSeq[Note]  # MIDI note numbers for each string
    order: StringOrder  # String numbering convention


# Mapping from TabInst to TabConfig
TAB_CONFIGS: dict[TabInst, TabConfig] = {
    TabInst.StandardGuitar: TabConfig(
        tuning=PSeq.mk(
            [Note(40), Note(45), Note(50), Note(55), Note(59), Note(64)]
        ),  # E2 A2 D3 G3 B3 E4
        order=StringOrder.HighToLow,
    ),
    TabInst.DropDGuitar: TabConfig(
        tuning=PSeq.mk(
            [Note(38), Note(45), Note(50), Note(55), Note(59), Note(64)]
        ),  # D2 A2 D3 G3 B3 E4
        order=StringOrder.HighToLow,
    ),
    TabInst.OpenGGuitar: TabConfig(
        tuning=PSeq.mk(
            [Note(38), Note(43), Note(50), Note(55), Note(59), Note(62)]
        ),  # D2 G2 D3 G3 B3 D4
        order=StringOrder.HighToLow,
    ),
    TabInst.OpenDGuitar: TabConfig(
        tuning=PSeq.mk(
            [Note(38), Note(45), Note(50), Note(54), Note(57), Note(62)]
        ),  # D2 A2 D3 F#3 A3 D4
        order=StringOrder.HighToLow,
    ),
    TabInst.DadgadGuitar: TabConfig(
        tuning=PSeq.mk(
            [Note(38), Note(45), Note(50), Note(55), Note(57), Note(62)]
        ),  # D2 A2 D3 G3 A3 D4
        order=StringOrder.HighToLow,
    ),
    TabInst.StandardBass: TabConfig(
        tuning=PSeq.mk([Note(28), Note(33), Note(38), Note(43)]),  # E1 A1 D2 G2
        order=StringOrder.HighToLow,
    ),
    TabInst.FiveStringBass: TabConfig(
        tuning=PSeq.mk(
            [Note(23), Note(28), Note(33), Note(38), Note(43)]
        ),  # B0 E1 A1 D2 G2
        order=StringOrder.HighToLow,
    ),
    TabInst.Ukulele: TabConfig(
        tuning=PSeq.mk(
            [Note(67), Note(60), Note(64), Note(69)]
        ),  # G4 C4 E4 A4 (reentrant)
        order=StringOrder.HighToLow,
    ),
    TabInst.Mandolin: TabConfig(
        tuning=PSeq.mk([Note(55), Note(62), Note(69), Note(76)]),  # G3 D4 A4 E5
        order=StringOrder.HighToLow,
    ),
    TabInst.Banjo: TabConfig(
        tuning=PSeq.mk(
            [Note(67), Note(50), Note(55), Note(59), Note(62)]
        ),  # G4 D3 G3 B3 D4
        order=StringOrder.HighToLow,
    ),
}


@dataclass(frozen=True)
class TabNote:
    """A note with tablature information."""

    note: Note
    instrument: TabInst
    string_num: int  # 1-based string number
    fret: int  # Fret number (0 = open string)


def interpret_tab_data(
    tab_data: TabData,
    instrument: TabInst,
) -> PSeq[TabNote]:
    """Interpret TabData into notes based on instrument tuning.

    Args:
        tab_data: Parsed tab data
        instrument: Instrument type (determines tuning and string order)

    Returns:
        PSeq of TabNote objects

    Raises:
        ValueError: If tab data is invalid for the instrument
    """
    # Get configuration for the instrument
    config = TAB_CONFIGS[instrument]
    open_strings = list(config.tuning)
    num_strings = len(open_strings)

    # Determine starting string
    if tab_data.start_string is not None:
        start_string = tab_data.start_string
        if start_string < 1 or start_string > num_strings:
            raise ValueError(f"Invalid string number {start_string} for {instrument}")
    else:
        # Default to highest numbered string
        start_string = num_strings

    # Get fret positions
    fret_values = list(tab_data.frets)
    if not fret_values:
        raise ValueError("No fret positions specified")

    notes = PSeq.empty(TabNote)

    for i, fret in enumerate(fret_values):
        # Calculate which string we're on
        if config.order == StringOrder.LowToHigh:
            # String 1 = lowest pitch (index 0)
            string_idx = (start_string - 1) + i
            display_string = start_string + i
        else:
            # String 1 = highest pitch (last index)
            # String numbering goes from high to low
            string_idx = (num_strings - start_string) + i
            display_string = start_string - i

        if string_idx < 0 or string_idx >= num_strings:
            # Out of range, skip
            continue

        if display_string < 1 or display_string > num_strings:
            continue

        # Skip muted strings (None values)
        if fret is None:
            continue

        if fret < 0:
            raise ValueError(f"Invalid fret number: {fret}")

        # Calculate MIDI note
        open_note = open_strings[string_idx]
        midi_note = open_note + fret

        if 0 <= midi_note <= 127:
            notes = notes.snoc(
                TabNote(
                    note=Note(midi_note),
                    instrument=instrument,
                    string_num=display_string,
                    fret=fret,
                )
            )

    return notes
