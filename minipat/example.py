"""Example usage and testing for the live pattern system."""

import logging
import time
from fractions import Fraction

from bad_actor import Actor, ActorEnv, new_system
from minipat.live import (
    BackendEvents,
    BackendMessage,
    BackendPlay,
    LiveSystem,
    LogProcessor,
    Orbit,
    Processor,
)
from minipat.pat import Pat
from minipat.stream import pat_stream


class LogBackendActor[T](Actor[BackendMessage[T]]):
    """Backend actor that handles processed events by logging them."""

    def __init__(self) -> None:
        self._logger = logging.getLogger("log_backend")
        self._playing = False

    def on_start(self, env: ActorEnv) -> None:
        env.logger.debug("Log backend actor started")

    def on_message(self, env: ActorEnv, value: BackendMessage[T]) -> None:
        match value:
            case BackendPlay(playing):
                self._playing = playing
                if playing:
                    self._logger.info("PLAY: Processing messages")
                else:
                    self._logger.info("PAUSE: Ignoring messages")
            case BackendEvents(events):
                self._logger.debug(f"Received {len(events)} processed events")
                for event in events:
                    self._logger.info(f"Backend Event: {event}")


def main() -> None:
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)

    # Create pattern system
    system = new_system("live_test")

    # Create processor and backend actor
    processor: Processor[str, str] = LogProcessor[str]()
    backend_actor = LogBackendActor[str]()
    backend_sender = system.spawn_actor("log_backend", backend_actor)

    live: LiveSystem[str, str] = LiveSystem.start(system, processor, backend_sender)

    try:
        # Create some simple patterns and convert to streams
        pattern1 = Pat.pure("kick")
        pattern2 = Pat.seq([Pat.pure("snare"), Pat.silent()])
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
