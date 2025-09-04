"""Component architecture for the PushPluck application.

This module provides abstract base classes for implementing components that
respond to configuration changes in a type-safe manner.
"""

from abc import ABCMeta, abstractmethod
from typing import Any, Generic, Optional, Type, TypeVar, override

C = TypeVar("C")
"""Type variable for the root configuration type."""
X = TypeVar("X", bound="MappedComponentConfig[Any]")
"""Type variable for mapped component configuration types."""
R = TypeVar("R")
"""Type variable for component result types."""
K = TypeVar("K", bound="Component[Any, Any]")
"""Type variable for component types."""


class MappedComponentConfig(Generic[C], metaclass=ABCMeta):
    """Abstract base for configuration objects that can be extracted from a root config."""

    @classmethod
    @abstractmethod
    def extract(cls: Type[X], root_config: C) -> X:
        """Extract this configuration type from a root configuration.

        Args:
            root_config: The root configuration object to extract from.

        Returns:
            The extracted configuration of this type.
        """
        raise NotImplementedError()


class Component(Generic[C, R], metaclass=ABCMeta):
    """Abstract base class for components that handle configuration changes."""

    @abstractmethod
    def handle_config(self, config: C, reset: bool) -> Optional[R]:
        """Handle a configuration update.

        Args:
            config: The new configuration to apply.
            reset: Whether this is a reset operation.

        Returns:
            Optional result from handling the configuration change.
        """
        raise NotImplementedError()


class MappedComponent(Generic[C, X, R], Component[C, R]):
    """Component that extracts its specific config from a root configuration.

    This class handles the extraction of component-specific configuration from
    a root configuration object and only processes changes when the relevant
    configuration actually changes.
    """

    @classmethod
    @abstractmethod
    def extract_config(cls: Type[K], root_config: C) -> X:
        """Extract this component's configuration from the root config.

        Args:
            root_config: The root configuration object.

        Returns:
            The component-specific configuration.
        """
        raise NotImplementedError()

    def __init__(self, config: X) -> None:
        """Initialize the component with its configuration.

        Args:
            config: The initial component configuration.
        """
        self._config = config

    @abstractmethod
    def handle_mapped_config(self, config: X) -> R:
        """Handle a change in the component's mapped configuration.

        Args:
            config: The new component-specific configuration.

        Returns:
            Result from handling the configuration change.
        """
        raise NotImplementedError()

    @override
    def handle_config(self, config: C, reset: bool) -> Optional[R]:
        """Handle a root configuration update by extracting relevant changes.

        This method extracts the component-specific configuration and only
        processes it if it has changed or a reset is requested.

        Args:
            config: The new root configuration.
            reset: Whether to force processing even if config hasn't changed.

        Returns:
            Optional result from handling the configuration change.
        """
        sub_config = type(self).extract_config(config)
        if sub_config != self._config or reset:
            return self.handle_mapped_config(sub_config)
        else:
            return None
