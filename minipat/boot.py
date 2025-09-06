import code

from minipat.dsl import (
    Flow,
    Nucleus,
    channel,
    control,
    midinote,
    note,
    program,
    value,
    vel,
)

__all__ = [
    "Flow",
    "note",
    "midinote",
    "vel",
    "channel",
    "program",
    "control",
    "value",
    "Nucleus",
]


_NUCLEUS_NAME = "n"


def _init() -> None:
    n = Nucleus.boot()
    globals()[_NUCLEUS_NAME] = n


def _cleanup() -> None:
    n = globals().get(_NUCLEUS_NAME)
    if n is not None:
        n.stop()


if __name__ == "__main__":
    _init()
    try:
        code.interact(local=locals())
    finally:
        _cleanup()
