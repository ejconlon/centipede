"""Minipat pattern language package for rhythmic and melodic pattern programming."""

from minipat.dsl import (
    Nucleus,
    channel,
    control,
    midinote,
    note,
    program,
    value,
    vel,
)
from minipat.kit import (
    DEFAULT_KIT,
    Kit,
    Sound,
    make_drum_sound,
)

__all__ = [
    "note",
    "midinote",
    "vel",
    "program",
    "control",
    "value",
    "channel",
    "Nucleus",
    "DEFAULT_KIT",
    "Kit",
    "Sound",
    "make_drum_sound",
]
