"""Printer for Pat patterns back to mini notation string format."""

from __future__ import annotations

from centipede.minipat.pat import (
    Pat,
    PatAlternating,
    PatChoice,
    PatElongation,
    PatEuclidean,
    PatGroup,
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
    """Raised when a Pat constructor cannot be printed."""

    pass


def print_pattern(pat: Pat[str]) -> str:
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
                return print_pattern(children[0])
            parts = []
            for child in children:
                # If the child is a sequence with multiple elements, bracket it
                if isinstance(child.unwrap, PatSeq) and len(child.unwrap.children) > 1:
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

        case PatPolymetric(patterns):
            pattern_strs = [print_pattern(pattern) for pattern in patterns]
            return f"{{{', '.join(pattern_strs)}}}"

        case PatRepetition(pattern, operator, count):
            pattern_str = print_pattern(pattern)
            op_str = operator.value
            return f"{pattern_str}{op_str}{count}"

        case PatElongation(pattern, count):
            pattern_str = print_pattern(pattern)
            # Use underscore for elongation (count is the actual number of underscores)
            return f"{pattern_str}{'_' * count}"

        case PatProbability(pattern, probability):
            pattern_str = print_pattern(pattern)
            if probability == 0.5:
                return f"{pattern_str}?"
            else:
                raise NotImplementedError(
                    f"PatProbability with custom probability {probability} cannot be printed - "
                    "only default 0.5 probability (?) is supported"
                )

        case PatSelect(pattern, selector):
            pattern_str = print_pattern(pattern)
            return f"{pattern_str}:{selector}"

        case PatGroup(pattern):
            pattern_str = print_pattern(pattern)
            return f"[{pattern_str}]"

        case PatAlternating(patterns):
            pattern_strs = [print_pattern(pattern) for pattern in patterns]
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
        return f"[{print_pattern(pat)}]"
    else:
        return print_pattern(pat)
