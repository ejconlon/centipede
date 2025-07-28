from __future__ import annotations

from lark import Lark, Transformer

from centipede.minipat.pat import Pat

# Grammar for pattern parsing
PATTERN_GRAMMAR = """
start: pattern
pattern: element+
element: SYMBOL | silence
silence: "~"

SYMBOL: /[a-zA-Z0-9]+/

%import common.WS
%ignore WS
"""


class PatternTransformer(Transformer):
    """Transform parsed pattern into Pat objects."""

    def start(self, items):
        """Transform the root pattern."""
        return items[0]

    def pattern(self, items):
        """Transform a pattern into a sequence of events."""
        return Pat.seq(items)

    def element(self, items):
        """Transform an element into a pure pattern."""
        return items[0]

    def silence(self, items):
        """Transform silence into empty pattern."""
        return Pat.silence()

    def SYMBOL(self, token):
        """Transform a symbol token into a pure pattern."""
        return Pat.pure(str(token))


def parse_pattern(pattern_str: str) -> Pat[str]:
    """Parse a pattern string like 'bd sd sd' into a Pat object.

    Args:
        pattern_str: A string representing a pattern, e.g., "bd sd sd"

    Returns:
        A Pat object representing the parsed pattern

    Examples:
        >>> parse_pattern("bd sd sd")
        # Returns a Pat.seq containing Pat.pure("bd"), Pat.pure("sd"), Pat.pure("sd")

        >>> parse_pattern("bd ~ sd")
        # Returns a Pat.seq with "bd", silence, "sd"
    """
    parser = Lark(PATTERN_GRAMMAR)
    transformer = PatternTransformer()
    tree = parser.parse(pattern_str)
    return transformer.transform(tree)
