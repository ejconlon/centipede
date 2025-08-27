"""Common types and constants for minipat pattern system."""

from fractions import Fraction
from typing import NewType

CycleTime = NewType("CycleTime", Fraction)
"""Time measured in elapsed cycles as fractions."""

PosixTime = NewType("PosixTime", float)
"""Time measured as POSIX timestamp (seconds since epoch)."""

type Delta = Fraction
"""Type alias for time deltas represented as fractions."""

type Factor = Fraction
"""Type alias for scaling factors represented as fractions."""

ZERO = Fraction(0)
"""The constant 0 as a fraction."""

ONE = Fraction(1)
"""The constant 1 as a fraction."""

ONE_HALF = Fraction(1, 2)
"""The constant 1/2 as a fraction."""


def format_fraction(frac: Fraction) -> str:
    """Format a fraction according to the printing rules.

    Rules:
    - Always print fractional representations in parentheses
    - Handle integers appropriately (no parentheses needed)

    Args:
        frac: The fraction to format

    Returns:
        String representation of the fraction
    """
    if frac.denominator == 1:
        # It's a whole number
        return str(frac.numerator)
    else:
        # Always use parenthesized fraction format
        return f"({frac.numerator}/{frac.denominator})"
