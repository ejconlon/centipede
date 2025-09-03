from abc import ABCMeta, abstractmethod
from typing import NewType, override

from minipat.pat import Pat, Selected
from spiny.dmap import DKey, DMap


class MidiDom:
    pass


type MidiAttrs = DMap[MidiDom]


class MidiKey[V](DKey[MidiDom, V]):
    pass


Note = NewType("Note", int)


class NoteKey(MidiKey[Note]):
    pass


Vel = NewType("Vel", int)


class VelKey(DKey[MidiDom, Vel]):
    pass


class Selector[V](metaclass=ABCMeta):
    @abstractmethod
    @classmethod
    def parse(cls, sel: Selected[str]) -> V:
        raise NotImplementedError

    @abstractmethod
    @classmethod
    def render(cls, value: V) -> Selected[str]:
        raise NotImplementedError


class NoteNumSelector(Selector[Note]):
    # TODO parse/render string and for each str assert integer representation (0-127) with empty selected
    # Attrs are singletons with NoteKey key

    @override
    @classmethod
    def parse(cls, sel: Selected[str]) -> Note:
        raise Exception("TODO")

    @override
    @classmethod
    def render(cls, value: Note) -> Selected[str]:
        raise Exception("TODO")


class NoteNameSelector(Selector[Note]):
    # TODO parse/render string and for each str assert string note representation (like c4) with empty selected
    # Attrs are singletons with NoteKey key

    @override
    @classmethod
    def parse(cls, sel: Selected[str]) -> Note:
        raise Exception("TODO")

    @override
    @classmethod
    def render(cls, value: Note) -> Selected[str]:
        raise Exception("TODO")


class VelSelector(Selector[Vel]):
    # TODO parse/render string and for each str assert integer representation (0-127) with empty selected

    @override
    @classmethod
    def parse(cls, sel: Selected[str]) -> Vel:
        raise Exception("TODO")

    @override
    @classmethod
    def render(cls, value: Vel) -> Selected[str]:
        raise Exception("TODO")


def midinote(s: str) -> Pat[MidiAttrs]:
    # TODO parse pat and use NoteNumSelector
    # Attrs are singletons with NoteKey key
    raise Exception("TODO")


def note(s: str) -> Pat[MidiAttrs]:
    # TODO parse pat and use NoteNameSelector
    # Attrs are singletons with NoteKey key
    raise Exception("TODO")


def vel(s: str) -> Pat[MidiAttrs]:
    # TODO parse pat and use VelSelector
    # Attrs are singletons with VelKey key
    raise Exception("TODO")
