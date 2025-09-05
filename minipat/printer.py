"""Printer for Pat patterns back to mini notation string format."""

from __future__ import annotations

from fractions import Fraction
from typing import cast

from minipat.common import format_fraction
from minipat.pat import (
    Pat,
    PatAlt,
    PatEuc,
    PatPar,
    PatPoly,
    PatProb,
    PatPure,
    PatRand,
    PatRepeat,
    PatSeq,
    PatSilent,
    PatSpeed,
    PatStretch,
)


def print_pattern(pat: Pat[str]) -> str:
    """Print a Pat pattern back to mini notation string format.

    Args:
        pat: The Pat pattern to print

    Returns:
        String representation in mini notation format
    """
    match pat.unwrap:
        case PatSilent():
            return "~"

        case PatPure(value):
            return cast(str, value)

        case PatSeq(children):
            if len(children) == 1:
                return print_pattern(children[0])

            parts = []
            for child in children:
                # If the child is a sequence with multiple elements, bracket it
                # Groups already have their own brackets
                if isinstance(child.unwrap, PatSeq) and len(child.unwrap.pats) > 1:
                    parts.append(f"[{print_pattern(child)}]")
                else:
                    parts.append(print_pattern(child))
            return " ".join(parts)

        case PatPar(children):
            # Parallel patterns use [a,b,c] notation
            pattern_strs = [print_pattern(pattern) for pattern in children]
            return f"[{', '.join(pattern_strs)}]"

        case PatRand(pats):
            choice_strs = [print_pattern(choice) for choice in pats]
            return f"[{' | '.join(choice_strs)}]"

        case PatEuc(pat, hits, steps, rotation):
            atom_str = print_pattern(pat)
            if rotation == 0:
                return f"{atom_str}({hits},{steps})"
            else:
                return f"{atom_str}({hits},{steps},{rotation})"

        case PatPoly(pats, None):
            pattern_strs = [print_pattern(pattern) for pattern in pats]
            return f"{{{', '.join(pattern_strs)}}}"

        case PatSpeed(pat, op, factor):
            # If the pattern being repeated is a multi-element sequence, it needs brackets
            # Bracketed sequences already handled above
            if isinstance(pat.unwrap, PatSeq) and len(pat.unwrap.pats) > 1:
                pattern_str = f"[{print_pattern(pat)}]"
            else:
                pattern_str = print_pattern(pat)
            op_str = op.value
            # Format fractions with % instead of /, but only for non-integer fractions
            if hasattr(factor, "numerator") and hasattr(factor, "denominator"):
                if factor.denominator == 1:
                    # Integer, just show the numerator
                    count_str = str(factor.numerator)
                else:
                    # Non-integer fraction, use % notation
                    count_str = f"{factor.numerator}%{factor.denominator}"
            else:
                count_str = str(factor)
            return f"{pattern_str}{op_str}{count_str}"

        case PatStretch(pat, count):
            pattern_str = print_pattern(pat)
            # Always use @notation for stretch patterns
            return f"{pattern_str}@{count}"

        case PatProb(pat, chance):
            pattern_str = print_pattern(pat)
            if chance == Fraction(1, 2):
                return f"{pattern_str}?"
            else:
                return f"{pattern_str}?{format_fraction(chance)}"

        case PatAlt(pats):
            pattern_strs = [print_pattern(pattern) for pattern in pats]
            return f"<{' '.join(pattern_strs)}>"

        case PatRepeat(pat, count):
            # If the pattern being replicated is a multi-element sequence, it needs brackets
            if isinstance(pat.unwrap, PatSeq) and len(pat.unwrap.pats) > 1:
                pattern_str = f"[{print_pattern(pat)}]"
            else:
                pattern_str = print_pattern(pat)
            return f"{pattern_str}!{format_fraction(count)}"

        case PatPoly(pats, subdiv):
            pattern_strs = [print_pattern(pattern) for pattern in pats]
            if subdiv is None:
                return f"{{{', '.join(pattern_strs)}}}"
            else:
                return f"{{{', '.join(pattern_strs)}}}%{subdiv}"

        case _:
            # This should never happen if all pattern types are handled above
            raise Exception(f"Unhandled pattern type: {type(pat.unwrap).__name__}")


def print_pattern_grouped(pat: Pat[str]) -> str:
    """Print a pattern with grouping brackets if it's a sequence.

    This is useful for printing sub-patterns that might need bracketing
    in certain contexts.
    """
    if isinstance(pat.unwrap, PatSeq) and len(pat.unwrap.pats) > 1:
        return f"[{print_pattern(pat)}]"
    else:
        return print_pattern(pat)
