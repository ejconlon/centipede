"""Printer for Pat patterns back to mini notation string format."""

from __future__ import annotations

from fractions import Fraction

from minipat.common import format_fraction
from minipat.pat import (
    Pat,
    PatAlternating,
    PatChoice,
    PatElongation,
    PatEuclidean,
    PatPar,
    PatPolymetric,
    PatProbability,
    PatPure,
    PatRepetition,
    PatReplicate,
    PatSelect,
    PatSeq,
    PatSilence,
)


def print_pattern(pat: Pat[str]) -> str:
    """Print a Pat pattern back to mini notation string format.

    Args:
        pat: The Pat pattern to print

    Returns:
        String representation in mini notation format

    Raises:
        NotPrintableError: If the pattern cannot be printed
    """
    match pat.unwrap:
        case PatSilence():
            return "~"

        case PatPure(val):
            return str(val)

        case PatSeq(children):
            if len(children) == 1:
                return print_pattern(children[0])

            parts = []
            for child in children:
                # If the child is a sequence with multiple elements, bracket it
                # Groups already have their own brackets
                if isinstance(child.unwrap, PatSeq) and len(child.unwrap.patterns) > 1:
                    parts.append(f"[{print_pattern(child)}]")
                else:
                    parts.append(print_pattern(child))
            return " ".join(parts)

        case PatPar(children):
            # Parallel patterns use [a,b,c] notation
            pattern_strs = [print_pattern(pattern) for pattern in children]
            return f"[{', '.join(pattern_strs)}]"

        case PatChoice(choices):
            choice_strs = [print_pattern(choice) for choice in choices]
            return f"[{' | '.join(choice_strs)}]"

        case PatEuclidean(atom, hits, steps, rotation):
            atom_str = print_pattern(atom)
            if rotation == 0:
                return f"{atom_str}({hits},{steps})"
            else:
                return f"{atom_str}({hits},{steps},{rotation})"

        case PatPolymetric(patterns, None):
            pattern_strs = [print_pattern(pattern) for pattern in patterns]
            return f"{{{', '.join(pattern_strs)}}}"

        case PatRepetition(pattern, operator, count):
            # If the pattern being repeated is a multi-element sequence, it needs brackets
            # Bracketed sequences already handled above
            if isinstance(pattern.unwrap, PatSeq) and len(pattern.unwrap.patterns) > 1:
                pattern_str = f"[{print_pattern(pattern)}]"
            else:
                pattern_str = print_pattern(pattern)
            op_str = operator.value
            # Format fractions with % instead of /, but only for non-integer fractions
            if hasattr(count, "numerator") and hasattr(count, "denominator"):
                if count.denominator == 1:
                    # Integer, just show the numerator
                    count_str = str(count.numerator)
                else:
                    # Non-integer fraction, use % notation
                    count_str = f"{count.numerator}%{count.denominator}"
            else:
                count_str = str(count)
            return f"{pattern_str}{op_str}{count_str}"

        case PatElongation(pattern, count):
            pattern_str = print_pattern(pattern)
            # Use underscore for elongation (count is the actual number of underscores)
            return f"{pattern_str}{'_' * count}"

        case PatProbability(pattern, probability):
            pattern_str = print_pattern(pattern)
            if probability == Fraction(1, 2):
                return f"{pattern_str}?"
            else:
                return f"{pattern_str}?{format_fraction(probability)}"

        case PatSelect(pattern, selector):
            pattern_str = print_pattern(pattern)
            return f"{pattern_str}:{selector}"

        case PatAlternating(patterns):
            pattern_strs = [print_pattern(pattern) for pattern in patterns]
            return f"<{' '.join(pattern_strs)}>"

        case PatReplicate(pattern, count):
            # If the pattern being replicated is a multi-element sequence, it needs brackets
            if isinstance(pattern.unwrap, PatSeq) and len(pattern.unwrap.patterns) > 1:
                pattern_str = f"[{print_pattern(pattern)}]"
            else:
                pattern_str = print_pattern(pattern)
            return f"{pattern_str}!{count}"

        case PatPolymetric(patterns, subdivision):
            pattern_strs = [print_pattern(pattern) for pattern in patterns]
            if subdivision is None:
                return f"{{{', '.join(pattern_strs)}}}"
            else:
                return f"{{{', '.join(pattern_strs)}}}%{subdivision}"

        case _:
            # This should never happen if all pattern types are handled above
            raise Exception(f"Unhandled pattern type: {type(pat.unwrap).__name__}")


def print_pattern_grouped(pat: Pat[str]) -> str:
    """Print a pattern with grouping brackets if it's a sequence.

    This is useful for printing sub-patterns that might need bracketing
    in certain contexts.
    """
    if isinstance(pat.unwrap, PatSeq) and len(pat.unwrap.patterns) > 1:
        return f"[{print_pattern(pat)}]"
    else:
        return print_pattern(pat)
