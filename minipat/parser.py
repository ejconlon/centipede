"""Parser for minipat pattern language using Lark."""

from __future__ import annotations

from fractions import Fraction

from lark import Lark, Transformer

from minipat.pat import Pat, PatElongation, PatSeq, RepetitionOp

# Lark grammar for parsing minipat pattern notation.
# This grammar defines the syntax for the minipat pattern language, including
# sequences, choices, parallel patterns, euclidean rhythms, and more.
PATTERN_GRAMMAR = """
%import common.WS
%ignore WS

// Tokens
SYMBOL: /[a-zA-Z0-9]([a-zA-Z0-9_-]*[a-zA-Z0-9])?/
NUMBER: /\\d+/
DECIMAL: /\\d*\\.\\d+/

// Operator tokens
MULTIPLY: "*"
DIVIDE: "/"
UNDERSCORE: "_"
AT: "@"
EXCLAMATION: "!"
PERCENT: "%"
DOT: "."
COLON: ":"

// Numeric values - supports integers, decimals, and fractions
numeric_value: NUMBER | DECIMAL | fraction | "(" fraction ")"
fraction: NUMBER "%" NUMBER

start: pattern

// Main pattern can be a sequence, dot grouping, or elements
pattern: dot_group | element_sequence
element_sequence: element (UNDERSCORE+ | element)*

// Elements can be various types
element: elongation | repetition | replicate | probability | atom | seq | choice | parallel | alternating | euclidean | polymetric

// Basic atoms
atom: symbol | silence
symbol: symbol_with_selector | SYMBOL
symbol_with_selector: SYMBOL COLON SYMBOL
silence: "~"

// Grouping structures
seq: "[" pattern "]"
dot_group: element_sequence (DOT element_sequence)+
choice: "[" choice_list "]"
choice_list: pattern ("|" pattern)+
parallel: "[" parallel_list "]"
parallel_list: pattern ("," pattern)+
alternating: "<" pattern+ ">"

// Euclidean rhythms: symbol(hits,steps) or symbol(hits,steps,rotation)
euclidean: atom "(" numeric_value "," numeric_value ("," numeric_value)? ")"

// Polymetric sequences
pattern_list: pattern ("," pattern)+
polymetric: "{" pattern_list "}" (PERCENT numeric_value)?

// Repetition and speed modifiers
repetition: element (MULTIPLY | DIVIDE) numeric_value
replicate: element EXCLAMATION numeric_value
elongation: element (UNDERSCORE+ | AT numeric_value)

// Probability
probability: atom "?" probability_value?
probability_value: numeric_value
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

    def element_sequence(self, items):
        """Transform element sequence, handling trailing underscores."""
        if len(items) == 1:
            return items[0]

        result = []
        current_element = items[0]

        for i in range(1, len(items)):
            item = items[i]
            if isinstance(item, str) and item == "_":
                # This is an underscore token - elongate the current element
                if isinstance(current_element.unwrap, PatElongation):
                    # Already elongated, add to count
                    base_element = current_element.unwrap.pattern
                    total_count = current_element.unwrap.count + 1
                    current_element = Pat.elongation(base_element, total_count)
                else:
                    # Create new elongation
                    current_element = Pat.elongation(current_element, 1)
            else:
                # This is another element
                result.append(current_element)
                current_element = item

        result.append(current_element)

        if len(result) == 1:
            return result[0]
        return Pat.seq(result)

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

    def symbol_with_selector(self, items):
        """Transform a symbol with selector (e.g., 'bd:kick')."""
        # items[0] and items[2] are already transformed Pat[str] objects from SYMBOL
        symbol_pat = items[0]
        selector_pat = items[2]

        # Extract the actual string values
        symbol_str = symbol_pat.unwrap.value
        selector_str = selector_pat.unwrap.value

        # Combine symbol and selector with colon
        combined_value = f"{symbol_str}:{selector_str}"
        return Pat.pure(combined_value)

    def silence(self, _items):
        """Transform silence into empty pattern."""
        return Pat.silence()

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
            patterns = list(items[0].unwrap.patterns)
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
    def pattern_list(self, items):
        """Transform pattern list a,b,c."""
        return items

    def polymetric(self, items):
        """Transform polymetric patterns {a,b,c} or {a,b,c}%4."""
        patterns = items[0]  # The pattern_list

        if len(items) > 1:
            # Has subdivision: {a,b,c}%4
            # items[1] is PERCENT token, items[2] is subdivision value
            factor = int(items[2])
            return Pat.polymetric(patterns, factor)
        else:
            # No subdivision: {a,b,c}
            return Pat.polymetric(patterns)

    # Repetition and speed modifiers
    def repetition(self, items):
        """Transform repetition patterns like bd*2 or bd/2."""
        element = items[0]
        op_str = str(items[1])
        num = items[2]  # Keep as is - can be int, float, or Fraction

        # Convert string operator to enum
        if op_str == "*":
            op = RepetitionOp.Fast
        elif op_str == "/":
            op = RepetitionOp.Slow
        else:
            raise ValueError(f"Unknown repetition operator: {op_str}")

        return Pat.repetition(element, op, num)

    def elongation(self, items):
        """Transform elongation patterns like bd_ or bd@2."""
        element = items[0]

        # Check if we have @ followed by a number
        if len(items) >= 3 and str(items[1]) == "@":
            # Case: bd@N (@ followed by numeric value)
            n = int(items[2])
            current_count = max(0, n - 1)  # @N means N-1 underscores
        else:
            # Case: bd_ (underscores)
            elongation_symbols = items[1:]
            current_count = len(elongation_symbols)

        # Check if the element is already an elongation and collapse them
        if isinstance(element.unwrap, PatElongation):
            # Nested elongation: combine the counts
            base_element = element.unwrap.pattern
            total_count = element.unwrap.count + current_count
            return Pat.elongation(base_element, total_count)
        else:
            # Regular elongation
            return Pat.elongation(element, current_count)

    def replicate(self, items):
        """Transform replicate patterns like bd!3."""
        element = items[0]
        # items[1] is the EXCLAMATION token, items[2] is the count
        count = int(items[2])
        return Pat.replicate(element, count)

    def dot_group(self, items):
        """Transform dot grouping patterns like bd sd _ . hh cp _ . oh."""
        # items alternate: element_sequence, DOT, element_sequence, DOT, ...
        # Extract just the element_sequences (skip DOT tokens)
        sequences = []
        for i, item in enumerate(items):
            if i % 2 == 0:  # Even indices are element_sequences
                sequences.append(item)

        return Pat.seq(sequences)

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
        """Transform fraction like 1%2 into Fraction."""
        numerator = int(items[0])
        denominator = int(items[1])
        return Fraction(numerator, denominator)

    def SYMBOL(self, token):
        """Transform a symbol token into a pure pattern with string value."""
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
