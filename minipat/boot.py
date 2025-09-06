import os

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
    "_var_name",
    "_init",
    "_cleanup",
]


_DEFAULT_VAR_NAME = "n"


def _var_name() -> str:
    return os.environ.get("MINIPAT_VAR", _DEFAULT_VAR_NAME)


def _init() -> None:
    name = _var_name()
    n = Nucleus.boot()
    globals()[name] = n
    n.play()


def _cleanup() -> None:
    name = _var_name()
    n = globals().get(name)
    if n is not None:
        n.stop()


if __name__ == "__main__":
    _init()
