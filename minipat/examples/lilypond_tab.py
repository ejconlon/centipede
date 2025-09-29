"""Example demonstrating LilyPond tablature engraving with minipat.

This example creates guitar tablature patterns and renders them to both
LilyPond source files and PDF output using the lilypond binary.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from minipat.combinators import tab_stream
from minipat.ev import EvHeap, ev_heap_empty
from minipat.messages import MidiAttrs
from minipat.offline import render_lilypond
from minipat.time import Bpc, Cps, CycleArc, CycleTime


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

    print(f"✓ Examples generated in {out_dir}/")
    print("LilyPond source files (.ly) and PDFs available for inspection.")


def generate_chord_example(out_dir: Path) -> None:
    """Generate a simple C major chord example."""
    # Create C major chord pattern: x32010
    stream = tab_stream("#x32010")

    # Create event heap for one cycle
    start_time = CycleTime(Fraction(0))
    end_time = CycleTime(Fraction(1))
    arc = CycleArc(start_time, end_time)

    events: EvHeap[MidiAttrs] = ev_heap_empty()
    for span, ev in stream.unstream(arc):
        events = events.insert(span, ev)

    # Render to LilyPond and compile to PDF
    pdf_path = render_lilypond(
        start=start_time,
        events=events,
        name="c_major_chord",
        directory=out_dir,
        cps=Cps(Fraction(1, 2)),  # 2 seconds per cycle
        bpc=Bpc(4),  # 4/4 time signature
    )

    print(f"  ✓ Generated: {pdf_path}")


def generate_arpeggio_example(out_dir: Path) -> None:
    """Generate an arpeggio pattern example."""
    # Create a simple G major chord pattern for arpeggio
    stream = tab_stream("#320003")

    # Create event heap for two cycles to show the pattern
    start_time = CycleTime(Fraction(0))
    end_time = CycleTime(Fraction(2))
    arc = CycleArc(start_time, end_time)

    events: EvHeap[MidiAttrs] = ev_heap_empty()
    for span, ev in stream.unstream(arc):
        events = events.insert(span, ev)

    # Render to LilyPond and compile to PDF
    pdf_path = render_lilypond(
        start=start_time,
        events=events,
        name="arpeggio_pattern",
        directory=out_dir,
        cps=Cps(Fraction(1, 2)),  # 2 seconds per cycle
        bpc=Bpc(4),  # 4/4 time signature
    )

    print(f"  ✓ Generated: {pdf_path}")


if __name__ == "__main__":
    main()
