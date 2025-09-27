"""Core types for minipat."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType, Optional

from spiny.seq import PSeq

# =============================================================================
# MIDI Value Types
# =============================================================================

Note = NewType("Note", int)
"""MIDI note number (0-127)"""

Velocity = NewType("Velocity", int)
"""MIDI velocity (0-127)"""

Channel = NewType("Channel", int)
"""MIDI channel (0-15)"""

Program = NewType("Program", int)
"""MIDI program number (0-127)"""

ControlNum = NewType("ControlNum", int)
"""MIDI control number (0-127)"""

ControlVal = NewType("ControlVal", int)
"""MIDI control value (0-127)"""


# =============================================================================
# Tablature Types
# =============================================================================


@dataclass(frozen=True)
class TabData:
    """Structured representation of parsed tablature notation."""

    start_string: Optional[int]  # Starting string number (1-based), None for default
    frets: PSeq[Optional[int]]  # Fret positions for each string (None for muted)
