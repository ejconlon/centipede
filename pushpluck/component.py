from abc import ABCMeta, abstractmethod
from typing import Generic, Optional, Type, TypeVar, override

C = TypeVar("C")
X = TypeVar("X", bound="MappedComponentConfig")
R = TypeVar("R")
K = TypeVar("K", bound="Component")


class MappedComponentConfig(Generic[C], metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def extract(cls: Type[X], root_config: C) -> X:
        raise NotImplementedError()


class Component(Generic[C, R], metaclass=ABCMeta):
    @abstractmethod
    def handle_config(self, config: C, reset: bool) -> Optional[R]:
        raise NotImplementedError()


class MappedComponent(Generic[C, X, R], Component[C, R]):
    @classmethod
    @abstractmethod
    def extract_config(cls: Type[K], root_config: C) -> X:
        raise NotImplementedError()

    def __init__(self, config: X) -> None:
        self._config = config

    @abstractmethod
    def handle_mapped_config(self, config: X) -> R:
        raise NotImplementedError()

    @override
    def handle_config(self, config: C, reset: bool) -> Optional[R]:
        sub_config = type(self).extract_config(config)
        if sub_config != self._config or reset:
            return self.handle_mapped_config(sub_config)
        else:
            return None
