from __future__ import annotations

from lark import Lark, Transformer

from centipede.minipat.pat import Pat, PatSeq, RepetitionOp

# Grammar for pattern parsing
PATTERN_GRAMMAR = """
start: pattern

// Main pattern can be a sequence or a single element
pattern: element+

// Elements can be various types
element: probability | elongation | repetition | base_element
base_element: atom | group | choice | parallel | alternating | euclidean | polymetric

// Basic atoms
atom: sample_selection | symbol | silence
symbol: SYMBOL
silence: "~"
sample_selection: SYMBOL ":" (NUMBER | SYMBOL)

// Grouping structures
group: "[" pattern "]" | "." pattern "."
choice: "[" choice_list "]"
choice_list: pattern ("|" pattern)+
parallel: "[" parallel_list "]"
parallel_list: pattern ("," pattern)+
alternating: "<" pattern+ ">"

// Euclidean rhythms: symbol(hits,steps) or symbol(hits,steps,rotation)
euclidean: atom "(" NUMBER "," NUMBER ("," NUMBER)? ")"

// Polymetric sequences
polymetric: "{" pattern ("," pattern)+ "}"

// Repetition and speed modifiers
repetition: base_element MULTIPLY NUMBER | base_element DIVIDE NUMBER | repetition DIVIDE NUMBER | repetition MULTIPLY NUMBER
elongation: base_element UNDERSCORE+ | base_element AT+ | repetition UNDERSCORE+ | repetition AT+

// Operator tokens
MULTIPLY: "*"
DIVIDE: "/"
UNDERSCORE: "_"
AT: "@"

// Probability
probability: atom "?"

// Tokens
SYMBOL: /[a-zA-Z0-9]([a-zA-Z0-9_-]*[a-zA-Z0-9])?/
NUMBER: /\\d+/

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

    def silence(self, items):
        """Transform silence into empty pattern."""
        return Pat.silence()

    def sample_selection(self, items):
        """Transform sample selection like 'bd:2' or 'bd:bar'."""
        symbol_token, selector_token = items
        # Create a pattern from the symbol token and use the selector as string
        # Extract the actual symbol string if it's already transformed
        if hasattr(symbol_token, "unwrap") and hasattr(symbol_token.unwrap, "val"):
            symbol_str = symbol_token.unwrap.val
        else:
            symbol_str = str(symbol_token)
        symbol_pat = Pat.pure(symbol_str)

        # Extract selector value - could be number or string
        if hasattr(selector_token, "unwrap") and hasattr(selector_token.unwrap, "val"):
            selector = selector_token.unwrap.val
        else:
            selector = str(selector_token)

        return Pat.select(symbol_pat, selector)

    # Grouping structures
    def group(self, items):
        """Transform grouping [...] or .pattern."""
        return Pat.group(items[0])

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
        num = int(items[2])

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

    # Probability
    def probability(self, items):
        """Transform probability patterns like bd?."""
        element = items[0]
        return Pat.probability(element)

    def SYMBOL(self, token):
        """Transform a symbol token into a pure pattern."""
        return Pat.pure(str(token))

    def NUMBER(self, token):
        """Transform a number token."""
        return int(str(token))


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
