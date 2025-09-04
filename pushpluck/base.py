"""Base classes and utilities for the PushPluck application.

This module provides fundamental abstract base classes, utility types, and
exceptions used throughout the PushPluck application.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any


class Closeable(metaclass=ABCMeta):
    """Abstract base class for objects that need explicit resource cleanup."""

    @abstractmethod
    def close(self) -> None:
        """Close this to free resources and deny further use."""
        raise NotImplementedError()


class Resettable(metaclass=ABCMeta):
    """Abstract base class for objects that can be reset to their initial state."""

    @abstractmethod
    def reset(self) -> None:
        """Reset this to a known good state for further use."""
        raise NotImplementedError()


@dataclass(frozen=True)
class Unit:
    """A singleton type with exactly one value.

    A simple type with one inhabitant (according to eq and hash).
    This is useful for representing the presence of something without
    additional information, similar to () in other languages.
    """

    @staticmethod
    def instance() -> Unit:
        """Get the singleton Unit instance.

        Returns:
            The singleton Unit instance.
        """
        return _UNIT_SINGLETON


_UNIT_SINGLETON = Unit()


class MatchException(Exception):
    """Exception raised when pattern matching fails."""

    def __init__(self, value: Any) -> None:
        """Initialize a MatchException with the unmatched value.

        Args:
            value: The value that failed to match any pattern.
        """
        super().__init__(f"Failed to match value: {value}")
