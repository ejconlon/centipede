"""Common types and constants for minipat pattern system."""

from fractions import Fraction

type Time = Fraction
"""Type alias for time values represented as fractions."""

type Delta = Fraction
"""Type alias for time deltas represented as fractions."""

type Factor = Fraction
"""Type alias for scaling factors represented as fractions."""

ZERO = Fraction(0)
"""The constant zero as a fraction."""

ONE = Fraction(1)
"""The constant one as a fraction."""
