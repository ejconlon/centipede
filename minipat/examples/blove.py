from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from minipat.combinators import chord_data_stream, tab_data_stream
from minipat.offline import render_lilypond
from minipat.stream import Stream
from minipat.time import CycleArc, CycleTime
from minipat.types import ChordData


def seq(*xs: str) -> str:
    # Build a sequential pattern that stretches to fit one cycle per item
    # This ensures each chord lasts one full measure/cycle
    pattern = " ".join(xs)
    return f"[{pattern}]/{len(xs)}"


def generate(out_dir: Path) -> None:
    # Create a tab pattern for the notes (strumming pattern)
    tabs = tab_data_stream(
        seq(
            "~",  # Lead in
            "~",  # Start
            "~",
            "~",
            "~",
            "~",
            "~",
            "~",
            "~",
            "~",
            "~",
            "~",
            "~",
            "~",
            "~",  # 1st ending
            "~",
            "~",
            "~",  # To start
            "~",  # 2nd ending
            "~",
            "~",
            "~",  # End
        )
    )

    # Create chord names to display above the staff using chord_data_stream
    chords: Stream[ChordData] = chord_data_stream(
        seq(
            "~",  # Lead in
            "e`m7f5",  # Start
            "a`7s5",
            "d`min",
            "~",
            "g`min7",
            "c`7",
            "f`maj7",
            "[e`m7f5 a`7]",
            "d`min",
            "g`min7",
            "bb`7s11",
            "a`7",
            "d`min",  # 1st ending
            "g`7s11",
            "e`min7f5",
            "a`7",  # To start
            "[d`min b`7s9]",  # 2nd ending
            "[bb`7 a`7]",
            "d`min",
            "~",  # End
        )
    )

    # Create arc for full song
    arc = CycleArc(CycleTime(Fraction(0)), CycleTime(Fraction(22)))

    # Render to LilyPond and compile to PDF and SVG with both streams
    output_files = render_lilypond(
        arc=arc,
        tab_stream=tabs,
        chord_stream=chords,
        name="beautiful_love",
        directory=out_dir,
        pdf=True,
        svg=True,
        key="f",
    )

    print(f"Generated: {', '.join(str(path) for path in output_files.values())}")


def main() -> None:
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    generate(out_dir)


if __name__ == "__main__":
    main()
