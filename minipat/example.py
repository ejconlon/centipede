"""Example usage and testing for the live pattern system."""

import logging
import time
from fractions import Fraction

from centipede.actor import new_system
from minipat.live import Backend, LiveEnv, LiveSystem, LogBackend, Orbit
from minipat.pat import Pat
from minipat.stream import pat_stream


def main():
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)

    # Create pattern system
    system = new_system("live_test")
    backend: Backend[str] = LogBackend()
    live: LiveSystem[str] = LiveSystem.start(system, backend, LiveEnv(debug=True))

    try:
        # Create some simple patterns and convert to streams
        pattern1 = Pat.pure("kick")
        pattern2 = Pat.seq([Pat.pure("snare"), Pat.silence()])
        stream1 = pat_stream(pattern1)
        stream2 = pat_stream(pattern2)

        # Set streams on orbits
        live.set_orbit(Orbit(0), stream1)
        live.set_orbit(Orbit(1), stream2)

        # Set tempo and start playback
        live.set_cps(Fraction(2))  # 2 cycles per second
        live.play()

        # Let it play for a few seconds
        time.sleep(3)

        # Test muting
        live.mute(Orbit(1))
        time.sleep(2)

        # Test soloing
        live.solo(Orbit(0))
        time.sleep(2)

        # Panic stop
        live.panic()
        time.sleep(1)

    finally:
        # Clean shutdown
        system.stop()
        system.wait()


if __name__ == "__main__":
    main()
