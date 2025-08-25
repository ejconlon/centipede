"""Parser for minipat pattern language using Lark."""

from __future__ import annotations

from fractions import Fraction

from lark import Lark, Transformer

from minipat.common import format_fraction
from minipat.pat import Pat, PatSeq, RepetitionOp

# Lark grammar for parsing minipat pattern notation.
# This grammar defines the syntax for the minipat pattern language, including
# sequences, choices, parallel patterns, euclidean rhythms, and more.
PATTERN_GRAMMAR = """
start: pattern

// Main pattern can be a sequence, dot grouping, or elements
pattern: dot_group | element+

// Elements can be various types
element: elongation | repetition | replicate | ratio | probability | base_element
base_element: atom | seq | choice | parallel | alternating | euclidean | polymetric | polymetric_sub

// Basic atoms
atom: select | symbol | silence
symbol: SYMBOL
silence: "~"
select: SYMBOL ":" (numeric_value | SYMBOL)

// Grouping structures
seq: "[" pattern "]"
dot_group: element+ DOT element+
choice: "[" choice_list "]"
choice_list: pattern ("|" pattern)+
parallel: "[" parallel_list "]"
parallel_list: pattern ("," pattern)+
alternating: "<" pattern+ ">"

// Euclidean rhythms: symbol(hits,steps) or symbol(hits,steps,rotation)
euclidean: atom "(" numeric_value "," numeric_value ("," numeric_value)? ")"

// Polymetric sequences
polymetric: "{" pattern ("," pattern)+ "}"
polymetric_sub: "{" pattern ("," pattern)+ "}" PERCENT numeric_value

// Repetition and speed modifiers
repetition: (base_element | probability) MULTIPLY numeric_value | (base_element | probability) DIVIDE numeric_value | repetition DIVIDE numeric_value | repetition MULTIPLY numeric_value
replicate: (base_element | probability) EXCLAMATION numeric_value | replicate EXCLAMATION numeric_value
ratio: (base_element | probability) MULTIPLY numeric_value PERCENT numeric_value | replicate PERCENT numeric_value
elongation: (base_element | probability) UNDERSCORE+ | (base_element | probability) AT+ | repetition UNDERSCORE+ | repetition AT+

// Operator tokens
MULTIPLY: "*"
DIVIDE: "/"
UNDERSCORE: "_"
AT: "@"
EXCLAMATION: "!"
PERCENT: "%"
DOT: "."


// Probability
probability: atom "?" probability_value?
probability_value: numeric_value

// Numeric values - supports integers, decimals, and parenthesized fractions
numeric_value: NUMBER | DECIMAL | "(" fraction ")"
fraction: NUMBER "/" NUMBER

// Tokens
SYMBOL: /[a-zA-Z0-9]([a-zA-Z0-9_-]*[a-zA-Z0-9])?/
NUMBER: /\\d+/
DECIMAL: /\\d*\\.\\d+/

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
        if len(items) == 1:
            return items[0]
        return Pat.seq(items)

    def element(self, items):
        """Transform an element into a pattern."""
        return items[0]

    def base_element(self, items):
        """Transform a base element into a pattern."""
        return items[0]

    # Basic atoms
    def atom(self, items):
        """Transform an atom element."""
        return items[0]

    def symbol(self, items):
        """Transform a symbol."""
        return items[0]

    def silence(self, _items):
        """Transform silence into empty pattern."""
        return Pat.silence()

    def select(self, items):
        """Transform sample selection like 'bd:2' or 'bd:bar'."""
        symbol_token, selector_token = items
        # Create a pattern from the symbol token and use the selector as string
        # Extract the actual symbol string if it's already transformed
        if hasattr(symbol_token, "unwrap") and hasattr(symbol_token.unwrap, "val"):
            symbol_str = symbol_token.unwrap.val
        else:
            symbol_str = str(symbol_token)
        symbol_pat = Pat.pure(symbol_str)

        # Extract selector value - could be Fraction, string, or raw value
        if isinstance(selector_token, Fraction):
            selector = format_fraction(selector_token)
        elif hasattr(selector_token, "unwrap") and hasattr(
            selector_token.unwrap, "val"
        ):
            selector = selector_token.unwrap.val
        else:
            selector = str(selector_token)

        return Pat.select(symbol_pat, selector)

    def seq(self, items):
        """Transform grouping [...] or .pattern."""
        pattern = items[0]
        # If the pattern is already a sequence, return it as-is
        # Otherwise, create a sequence with the single pattern
        if isinstance(pattern.unwrap, PatSeq):
            return pattern
        else:
            return Pat.seq([pattern])

    def choice(self, items):
        """Transform choice patterns [a|b|c]."""
        choices = items[0]
        return Pat.choice(choices)

    def choice_list(self, items):
        """Transform choice list a|b|c."""
        return items

    def parallel(self, items):
        """Transform parallel patterns [a,b,c]."""
        patterns = items[0]
        return Pat.par(patterns)

    def parallel_list(self, items):
        """Transform parallel list a,b,c."""
        return items

    def alternating(self, items):
        """Transform alternating patterns <a b c>."""
        # If we get a single item that's a sequence, extract its children
        if len(items) == 1 and isinstance(items[0].unwrap, PatSeq):
            patterns = list(items[0].unwrap.children)
            return Pat.alternating(patterns)
        else:
            return Pat.alternating(items)

    # Euclidean rhythms
    def euclidean(self, items):
        """Transform Euclidean rhythm pattern like bd(3,8)."""
        atom = items[0]
        # Convert Fraction to int (should be whole numbers for euclidean)
        hits = int(items[1])
        steps = int(items[2])
        rotation = int(items[3]) if len(items) > 3 else 0
        return Pat.euclidean(atom, hits, steps, rotation)

    # Polymetric sequences
    def polymetric(self, items):
        """Transform polymetric patterns {a,b,c}."""
        return Pat.polymetric(items)

    # Repetition and speed modifiers
    def repetition(self, items):
        """Transform repetition patterns like bd*2 or bd/2."""
        element = items[0]
        op_str = str(items[1])
        num = int(items[2])  # Convert Fraction to int for repetition count

        # Convert string operator to enum
        if op_str == "*":
            op = RepetitionOp.FAST
        elif op_str == "/":
            op = RepetitionOp.SLOW
        else:
            raise ValueError(f"Unknown repetition operator: {op_str}")

        return Pat.repetition(element, op, num)

    def elongation(self, items):
        """Transform elongation patterns like bd_ or bd@."""
        element = items[0]
        elongation_symbols = items[1:]
        elongation_count = len(elongation_symbols)
        return Pat.elongation(element, elongation_count)

    def replicate(self, items):
        """Transform replicate patterns like bd!3."""
        element = items[0]
        # items[1] is the EXCLAMATION token, items[2] is the count
        count = int(items[2])
        return Pat.replicate(element, count)

    def ratio(self, items):
        """Transform ratio patterns like bd*3%2."""
        element = items[0]
        # items[1]=MULTIPLY, items[2]=numerator, items[3]=PERCENT, items[4]=denominator
        numerator = int(items[2])
        denominator = int(items[4])
        return Pat.ratio(element, numerator, denominator)

    def polymetric_sub(self, items):
        """Transform polymetric patterns with subdivision like {a,b}%4."""
        # items structure: {, pattern, comma, pattern, ..., }, PERCENT, numeric_value
        # Find the last numeric value (subdivision)
        subdivision = int(items[-1])
        # Everything else except the PERCENT and subdivision are pattern-related
        patterns = []
        for item in items[:-2]:  # Skip PERCENT and subdivision
            if hasattr(item, "unwrap"):  # It's a pattern
                patterns.append(item)
        return Pat.polymetric_sub(patterns, subdivision)

    def dot_group(self, items):
        """Transform dot grouping patterns like bd sd . hh cp."""
        # Find the DOT token and split the elements
        dot_index = -1
        for i, item in enumerate(items):
            if hasattr(item, "type") and item.type == "DOT":
                dot_index = i
                break

        if dot_index == -1:
            raise ValueError("DOT not found in dot_group")

        left_elements = items[:dot_index]
        right_elements = items[dot_index + 1 :]

        # Create sequences from left and right parts
        left_seq = (
            Pat.seq(left_elements) if len(left_elements) > 1 else left_elements[0]
        )
        right_seq = (
            Pat.seq(right_elements) if len(right_elements) > 1 else right_elements[0]
        )

        # Return a sequence of the two parts
        return Pat.seq([left_seq, right_seq])

    # Probability
    def probability(self, items):
        """Transform probability patterns like bd?, bd?0.3, bd?(1/2)."""
        element = items[0]

        # Handle different probability formats
        if len(items) == 1:
            # Simple "bd?" case - default 0.5 probability
            return Pat.probability(element)
        elif len(items) == 2:
            # "bd?VALUE" case - get probability from probability_value
            prob_value = items[1]
            return Pat.probability(element, prob_value)
        else:
            raise ValueError(f"Invalid probability pattern with {len(items)} items")

    def probability_value(self, items):
        """Transform probability value - numeric value."""
        return items[0]

    def numeric_value(self, items):
        """Transform numeric value - integer, decimal, or fraction."""
        return items[0]

    def fraction(self, items):
        """Transform fraction like 1/2 into Fraction."""
        numerator = int(items[0])
        denominator = int(items[1])
        return Fraction(numerator, denominator)

    def SYMBOL(self, token):
        """Transform a symbol token into a pure pattern."""
        return Pat.pure(str(token))

    def NUMBER(self, token):
        """Transform a number token."""
        return Fraction(int(str(token)))

    def DECIMAL(self, token):
        """Transform a decimal token."""
        return Fraction(str(token))


def parse_pattern(pattern_str: str) -> Pat[str]:
    """Parse a pattern string into a Pat object.

    Args:
        pattern_str: A string representing a pattern

    Returns:
        A Pat object representing the parsed pattern

    Examples:
        >>> parse_pattern("bd sd sd")
        # Returns a Pat.seq containing Pat.pure("bd"), Pat.pure("sd"), Pat.pure("sd")

        >>> parse_pattern("bd ~ sd")
        # Returns a Pat.seq with "bd", silence, "sd"

        >>> parse_pattern("bd*2 sd")
        # Returns a Pat.seq with repeated "bd", then "sd"

        >>> parse_pattern("[bd sd] cp")
        # Returns a Pat.seq with grouped "bd sd", then "cp"

        >>> parse_pattern("bd(3,8)")
        # Returns a euclidean rhythm pattern

        >>> parse_pattern("{bd, sd}")
        # Returns parallel patterns

        >>> parse_pattern("[bd|sd|cp]")
        # Returns choice pattern
    """
    parser = Lark(PATTERN_GRAMMAR)
    transformer = PatternTransformer()
    tree = parser.parse(pattern_str)
    return transformer.transform(tree)
