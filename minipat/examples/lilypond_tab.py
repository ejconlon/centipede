"""Example demonstrating LilyPond tablature engraving with minipat.

This example creates guitar tablature patterns and renders them to both
LilyPond source files and PDF output using the lilypond binary.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from minipat.combinators import chord_data_stream, tab_data_stream
from minipat.offline import render_lilypond
from minipat.stream import Stream
from minipat.time import Bpc, Cps, CycleArc, CycleTime
from minipat.types import ChordData


def main() -> None:
    """Generate LilyPond tablature examples and compile to PDF."""
    # Ensure output directory exists
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    # Example 1: Simple C major chord
    print("Generating C major chord tablature...")
    generate_chord_example(out_dir)

    # Example 2: Arpeggio pattern
    print("Generating arpeggio pattern...")
    generate_arpeggio_example(out_dir)

    # Example 3: Chord data pattern
    print("Generating chord data progression...")
    generate_chord_data_example(out_dir)

    print(f"✓ Examples generated in {out_dir}/")
    print("LilyPond source files (.ly), PDFs, and SVGs available for inspection.")


def generate_chord_example(out_dir: Path) -> None:
    """Generate a simple C major chord example."""
    # Create C major chord pattern: x32010
    stream = tab_data_stream("#x32010")

    # Create arc for one cycle
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))

    # Render to LilyPond and compile to PDF and SVG
    output_files = render_lilypond(
        arc=arc,
        tab_stream=stream,
        name="c_major_chord",
        directory=out_dir,
        cps=Cps(Fraction(1, 2)),  # 2 seconds per cycle
        bpc=Bpc(4),  # 4/4 time signature
        pdf=True,
        svg=True,
    )

    print(f"  ✓ Generated: {', '.join(str(path) for path in output_files.values())}")


def generate_arpeggio_example(out_dir: Path) -> None:
    """Generate an arpeggio pattern example."""
    # Create G major arpeggio followed by G major chord
    # Individual notes: 6#3 (G), 5#2 (B), 4#0 (D), 3#0 (G), 2#0 (B), 1#3 (G)
    # Then chord: #320003 (G major), then silence
    stream = tab_data_stream("[6#3 5#2 4#0 3#0 2#0 1#3 #320003 _]/2")

    # Create arc for two cycles to show the pattern
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(2)))

    # Render to LilyPond and compile to PDF and SVG
    output_files = render_lilypond(
        arc=arc,
        tab_stream=stream,
        name="arpeggio_pattern",
        directory=out_dir,
        cps=Cps(Fraction(1, 2)),  # 2 seconds per cycle
        bpc=Bpc(4),  # 4/4 time signature
        pdf=True,
        svg=True,
    )

    print(f"  ✓ Generated: {', '.join(str(path) for path in output_files.values())}")


def generate_chord_data_example(out_dir: Path) -> None:
    """Generate a chord progression example using both tab stream and chord data."""
    # Create a tab pattern for the notes (strumming pattern)
    tabs = tab_data_stream("#x32010 #x02210 #xx3210 #320003")  # C, Am, F, G chords

    # Create chord names to display above the staff using chord_data_stream
    chords: Stream[ChordData] = chord_data_stream("c4`maj a4`min f4`maj g4`maj")

    # Create arc for one cycle
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(1)))

    # Render to LilyPond and compile to PDF and SVG with both streams
    output_files = render_lilypond(
        arc=arc,
        tab_stream=tabs,
        chord_stream=chords,
        name="chord_progression",
        directory=out_dir,
        cps=Cps(Fraction(1, 2)),  # 2 seconds per cycle
        bpc=Bpc(4),  # 4/4 time signature
        pdf=True,
        svg=True,
    )

    print(f"  ✓ Generated: {', '.join(str(path) for path in output_files.values())}")


if __name__ == "__main__":
    main()
