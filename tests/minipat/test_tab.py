"""Tests for tablature parsing."""

from __future__ import annotations

import pytest

from minipat.parser import parse_tab
from minipat.tab import TabInst, TabNote, interpret_tab_data
from spiny import PSeq


def _parse_interp(
    tab_str: str, inst: TabInst = TabInst.StandardGuitar
) -> PSeq[TabNote]:
    tab = parse_tab(tab_str)
    return interpret_tab_data(tab, inst)


class TestParseTab:
    """Tests for parse_tab function."""

    def test_c_major_chord(self) -> None:
        """Test parsing C major chord (x32010)."""
        notes = _parse_interp("#x32010")
        assert len(notes) == 5

        # Extract MIDI values and tab info
        midi_values = [int(tn.note) for tn in notes]
        string_nums = [tn.string_num for tn in notes]
        frets = [tn.fret for tn in notes]
        instruments = [tn.instrument for tn in notes]

        # Check MIDI notes match expected C chord
        assert midi_values == [48, 52, 55, 60, 64]
        # Check string numbers (6, 5, 4, 3, 2, 1 -> skip muted 6)
        assert string_nums == [5, 4, 3, 2, 1]
        # Check frets
        assert frets == [3, 2, 0, 1, 0]
        # Check instrument
        assert all(inst == TabInst.StandardGuitar for inst in instruments)

    def test_g_major_chord(self) -> None:
        """Test parsing G major chord (320003)."""
        notes = _parse_interp("#320003")
        assert len(notes) == 6

        midi_values = [int(tn.note) for tn in notes]
        string_nums = [tn.string_num for tn in notes]
        frets = [tn.fret for tn in notes]

        assert midi_values == [43, 47, 50, 55, 59, 67]
        assert string_nums == [6, 5, 4, 3, 2, 1]
        assert frets == [3, 2, 0, 0, 0, 3]

    def test_explicit_string_number(self) -> None:
        """Test parsing with explicit starting string."""
        notes = _parse_interp("4#221")
        assert len(notes) == 3

        string_nums = [tn.string_num for tn in notes]
        frets = [tn.fret for tn in notes]

        # Starting from string 4, going down
        assert string_nums == [4, 3, 2]
        assert frets == [2, 2, 1]

    def test_muted_strings(self) -> None:
        """Test that muted strings are skipped."""
        notes = _parse_interp("#xx00")
        assert len(notes) == 2

        string_nums = [tn.string_num for tn in notes]
        frets = [tn.fret for tn in notes]

        # Only strings 4 and 3 (open) - skipping muted 6 and 5
        assert string_nums == [4, 3]
        assert frets == [0, 0]

    def test_different_instrument(self) -> None:
        """Test parsing with different instrument."""
        notes = _parse_interp("#0000", TabInst.StandardBass)
        assert len(notes) == 4

        instruments = [tn.instrument for tn in notes]
        assert all(inst == TabInst.StandardBass for inst in instruments)

        # Bass strings should be different pitches (4-string bass, high to low)
        midi_values = [int(tn.note) for tn in notes]
        assert midi_values == [28, 33, 38, 43]  # E1, A1, D2, G2

    def test_invalid_format(self) -> None:
        """Test error handling for invalid formats."""
        # Missing # separator
        with pytest.raises(ValueError, match="missing #"):
            _parse_interp("320003")

        # Invalid string number
        with pytest.raises(ValueError, match="Invalid string number"):
            _parse_interp("7#0000")

        # Invalid fret token (only single digits supported)
        with pytest.raises(ValueError, match="Invalid fret token"):
            _parse_interp("#a")

        # Empty string
        with pytest.raises(ValueError, match="missing #"):
            _parse_interp("")

        # No fret positions
        with pytest.raises(ValueError, match="No fret positions"):
            _parse_interp("#")
