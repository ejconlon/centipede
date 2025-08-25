"""Printer for Pat patterns back to mini notation string format."""

from __future__ import annotations

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
    PatSelect,
    PatSeq,
    PatSilence,
)


class NotImplementedError(Exception):
    """Raised when a Pat constructor cannot be printed.

    This exception is raised when attempting to print a pattern
    that doesn't have a corresponding representation in mini notation.
    """

    pass


def print_pattern(pat: Pat[str], *, _top_level: bool = True) -> str:
    """Print a Pat pattern back to mini notation string format.

    Args:
        pat: The Pat pattern to print

    Returns:
        String representation in mini notation format

    Raises:
        NotImplementedError: If the pattern cannot be printed
    """
    match pat.unwrap:
        case PatSilence():
            return "~"

        case PatPure(val):
            return str(val)

        case PatSeq(children):
            if len(children) == 1:
                return print_pattern(children[0], _top_level=False)

            parts = []
            for child in children:
                # If the child is a sequence with multiple elements, bracket it
                # Groups already have their own brackets
                if isinstance(child.unwrap, PatSeq) and len(child.unwrap.children) > 1:
                    parts.append(f"[{print_pattern(child, _top_level=False)}]")
                else:
                    parts.append(print_pattern(child, _top_level=False))
            return " ".join(parts)

        case PatPar(children):
            # Parallel patterns use [a,b,c] notation
            pattern_strs = [
                print_pattern(pattern, _top_level=False) for pattern in children
            ]
            return f"[{', '.join(pattern_strs)}]"

        case PatChoice(choices):
            choice_strs = [
                print_pattern(choice, _top_level=False) for choice in choices
            ]
            return f"[{' | '.join(choice_strs)}]"

        case PatEuclidean(atom, hits, steps, rotation):
            atom_str = print_pattern(atom, _top_level=False)
            if rotation == 0:
                return f"{atom_str}({hits},{steps})"
            else:
                return f"{atom_str}({hits},{steps},{rotation})"

        case PatPolymetric(patterns):
            pattern_strs = [
                print_pattern(pattern, _top_level=False) for pattern in patterns
            ]
            return f"{{{', '.join(pattern_strs)}}}"

        case PatRepetition(pattern, operator, count):
            # If the pattern being repeated is a multi-element sequence, it needs brackets
            # Bracketed sequences already handled above
            if isinstance(pattern.unwrap, PatSeq) and len(pattern.unwrap.children) > 1:
                pattern_str = f"[{print_pattern(pattern, _top_level=False)}]"
            else:
                pattern_str = print_pattern(pattern, _top_level=False)
            op_str = operator.value
            return f"{pattern_str}{op_str}{count}"

        case PatElongation(pattern, count):
            pattern_str = print_pattern(pattern, _top_level=False)
            # Use underscore for elongation (count is the actual number of underscores)
            return f"{pattern_str}{'_' * count}"

        case PatProbability(pattern, probability):
            pattern_str = print_pattern(pattern, _top_level=False)
            if probability == 0.5:
                return f"{pattern_str}?"
            else:
                raise NotImplementedError(
                    f"PatProbability with custom probability {probability} cannot be printed - "
                    "only default 0.5 probability (?) is supported"
                )

        case PatSelect(pattern, selector):
            pattern_str = print_pattern(pattern, _top_level=False)
            return f"{pattern_str}:{selector}"

        case PatAlternating(patterns):
            pattern_strs = [
                print_pattern(pattern, _top_level=False) for pattern in patterns
            ]
            return f"<{' '.join(pattern_strs)}>"

        case _:
            pattern_type = type(pat.unwrap).__name__
            raise NotImplementedError(
                f"{pattern_type} cannot be printed to mini notation format"
            )


def print_pattern_grouped(pat: Pat[str]) -> str:
    """Print a pattern with grouping brackets if it's a sequence.

    This is useful for printing sub-patterns that might need bracketing
    in certain contexts.
    """
    if isinstance(pat.unwrap, PatSeq) and len(pat.unwrap.children) > 1:
        return f"[{print_pattern(pat, _top_level=False)}]"
    else:
        return print_pattern(pat, _top_level=False)
