"""Round-trip tests for parsing and printing mini notation patterns.

Based on test patterns from https://raw.githubusercontent.com/ejconlon/minipat/refs/heads/master/minipat/test/Main.hs
"""

import pytest

from centipede.minipat.parser import parse_pattern
from centipede.minipat.printer import NotImplementedError, print_pattern


def round_trip_test(pattern_str: str, expected_str: str | None = None) -> None:
    """Test that a pattern can be parsed and printed back.

    Args:
        pattern_str: The original pattern string
        expected_str: Expected printed result (defaults to pattern_str)
    """
    expected = expected_str if expected_str is not None else pattern_str

    # Parse the pattern
    parsed = parse_pattern(pattern_str)

    # Print it back
    printed = print_pattern(parsed)

    # Should round-trip correctly
    assert printed == expected, (
        f"Round-trip failed: {pattern_str} -> {printed} (expected {expected})"
    )

    # Parse the printed version to ensure it's equivalent
    reparsed = parse_pattern(printed)

    # The parsed structures should be equivalent (this is a basic check)
    # More sophisticated equality checking could be implemented later
    assert type(parsed.unwrap) is type(reparsed.unwrap)


def test_basic_patterns():
    """Test basic pattern elements."""
    round_trip_test("x")
    round_trip_test("~")
    round_trip_test("bd")
    round_trip_test("hh")


def test_sequence_patterns():
    """Test sequence patterns."""
    round_trip_test("x y")
    round_trip_test("bd sd hh")
    round_trip_test("bd ~ sd")
    round_trip_test("a b c d")


def test_sample_selection():
    """Test sample selection patterns."""
    round_trip_test("bd:0")
    round_trip_test("sd:1")
    round_trip_test("hh:2")
    round_trip_test("bd:0 sd:1")


def test_repetition_patterns():
    """Test repetition and speed patterns."""
    round_trip_test("x*9")
    round_trip_test("bd*3")
    round_trip_test("x/2")
    round_trip_test("bd/4")
    round_trip_test("hh*2/3")


def test_elongation_patterns():
    """Test elongation patterns."""
    round_trip_test("x_")
    round_trip_test("bd__")
    round_trip_test("hh___")


def test_probability_patterns():
    """Test probability patterns."""
    round_trip_test("x?")
    round_trip_test("bd?")
    round_trip_test("hh? sd?")


def test_choice_patterns():
    """Test choice patterns."""
    round_trip_test("[x | y]", "[x | y]")
    round_trip_test("[bd | sd | cp]", "[bd | sd | cp]")
    round_trip_test("[bd | sd]", "[bd | sd]")


def test_euclidean_patterns():
    """Test Euclidean rhythm patterns."""
    round_trip_test("x(1,2)")
    round_trip_test("bd(3,8)")
    round_trip_test("cp(5,8,2)")
    round_trip_test("hh(7,16)")


def test_polymetric_patterns():
    """Test polymetric patterns."""
    round_trip_test("{x, y}", "{x, y}")
    round_trip_test("{bd, sd}", "{bd, sd}")
    round_trip_test("{bd sd, hh*8}", "{bd sd, hh*8}")
    round_trip_test("{a, b, c}", "{a, b, c}")


def test_complex_patterns():
    """Test complex nested patterns."""
    round_trip_test("bd*3 sd")
    round_trip_test("bd:0*2 sd:1/2")
    round_trip_test("bd(3,8) ~ hh?")
    round_trip_test("{bd*2, sd_} cp")
    round_trip_test("bd? [sd | cp] hh*4")


def test_grouped_sequences():
    """Test that grouped sequences are handled correctly."""
    # Grouped sequences should preserve their brackets
    round_trip_test("[bd sd]")  # Preserve explicit grouping
    round_trip_test("[bd sd] cp")  # Preserve nested grouping
    round_trip_test("bd [sd cp] hh")  # Preserve nested grouping


def test_whitespace_normalization():
    """Test that whitespace is normalized correctly."""
    # These should normalize to standard spacing
    patterns_with_expected = [
        ("bd   sd", "bd sd"),
        ("  bd sd  ", "bd sd"),
        ("bd\tsd", "bd sd"),
        ("bd  ~  sd", "bd ~ sd"),
    ]

    for original, expected in patterns_with_expected:
        round_trip_test(original, expected)


class TestNonPrintablePatterns:
    """Test patterns that cannot be printed."""

    def test_patpar_not_printable(self):
        """Test that PatPar patterns raise NotImplementedError."""
        # Create a PatPar pattern directly (not through parsing)
        from centipede.minipat.pat import Pat

        pat = Pat.par([Pat.pure("bd"), Pat.pure("sd")])

        with pytest.raises(NotImplementedError, match="PatPar cannot be printed"):
            print_pattern(pat)

    def test_custom_probability_not_printable(self):
        """Test that custom probability values raise NotImplementedError."""
        from centipede.minipat.pat import Pat

        pat = Pat.probability(Pat.pure("bd"), 0.75)

        with pytest.raises(NotImplementedError, match="custom probability"):
            print_pattern(pat)


class TestParsingEdgeCases:
    """Test edge cases in parsing that might affect round-tripping."""

    def test_empty_pattern_fails(self):
        """Test that empty patterns fail to parse."""
        with pytest.raises(Exception):
            parse_pattern("")

    def test_invalid_syntax_fails(self):
        """Test that invalid syntax fails to parse."""
        invalid_patterns = [
            "bd(",
            "bd)",
            "[bd",
            "bd]",
            "{bd",
            "bd}",
            "bd|",
            "|bd",
            "bd:",
            ":bd",
            "bd*",
            "*bd",
            "bd/",
            "/bd",
        ]

        for pattern in invalid_patterns:
            with pytest.raises(Exception):
                parse_pattern(pattern)


class TestPatternEquivalence:
    """Test that semantically equivalent patterns round-trip correctly."""

    def test_single_element_sequences(self):
        """Test that single-element sequences are simplified."""
        # A sequence with one element should print as just that element
        parsed = parse_pattern("bd")
        printed = print_pattern(parsed)
        assert printed == "bd"

    def test_nested_sequences_preserve_grouping(self):
        """Test behavior with nested sequences."""
        # Sequences should preserve explicit grouping
        round_trip_test("bd sd")
        round_trip_test("[bd sd] cp")  # Preserve grouping


# Integration tests with real-world patterns
class TestRealWorldPatterns:
    """Test patterns that might be used in actual compositions."""

    def test_drum_patterns(self):
        """Test typical drum patterns."""
        drum_patterns = [
            ("bd ~ sd ~", None),
            ("bd bd ~ sd", None),
            ("[bd bd] sd [bd sd]", None),  # Preserve grouping
            ("bd*2 sd cp*3", None),
            ("{bd, sd*2, hh*8}", None),
            ("bd(3,8) sd(5,8)", None),
        ]

        for pattern, expected in drum_patterns:
            round_trip_test(pattern, expected)

    def test_melodic_patterns(self):
        """Test patterns that might represent melodies."""
        melodic_patterns = [
            ("c4 e4 g4 c5", None),
            ("c:0 e:1 g:2", None),
            ("[c e g] [d f a]", None),  # Preserve grouping
            ("c*2 e g/2", None),
            ("c? e g? c", None),
        ]

        for pattern, expected in melodic_patterns:
            round_trip_test(pattern, expected)

    def test_complex_compositions(self):
        """Test complex compositional patterns."""
        complex_patterns = [
            ("bd*2 [sd cp] ~ {hh*4, oh}", None),  # Preserve grouping
            ("{bd(3,8), sd?, hh*8/2}", None),
            ("bd:0*2 [sd | cp] hh:1", None),  # Choice patterns keep their brackets
            ("bass:0 bass:1? ~ [bass:2 bass:3]/2", None),  # Preserve grouping
        ]

        for pattern, expected in complex_patterns:
            round_trip_test(pattern, expected)
