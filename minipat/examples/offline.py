from minipat.dsl import note
from minipat.offline import play_midi, render_midi


def main() -> None:
    mid = render_midi(
        [
            (1, note("[~ g3 a3 b3, c2 g2 a2 b2]")),
            (1, note("[d4'min7 g4'dom7 c4'maj7 _, d2 g2 c2 c2]")),
        ]
    )
    play_midi(mid)


if __name__ == "__main__":
    main()
