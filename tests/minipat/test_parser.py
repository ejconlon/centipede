from fractions import Fraction

import pytest

from minipat.parser import parse_pattern
from minipat.pat import (
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
    SpeedOp,
)


def assert_string_value(value: str, expected_value: str) -> None:
    """Helper function to assert string values."""
    assert value == expected_value


def test_parse_basic_symbol() -> None:
    """Test parsing a single symbol."""
    result = parse_pattern("bd")
    assert isinstance(result.unwrap, PatPure)
    assert_string_value(result.unwrap.value, "bd")


def test_parse_simple_sequence() -> None:
    """Test parsing a simple sequence of symbols."""
    result = parse_pattern("bd sd hh")
    assert isinstance(result.unwrap, PatSeq)

    children = list(result.unwrap.pats)
    assert len(children) == 3
    assert all(isinstance(child.unwrap, PatPure) for child in children)
    assert_string_value(children[0].unwrap.value, "bd")
    assert_string_value(children[1].unwrap.value, "sd")
    assert_string_value(children[2].unwrap.value, "hh")


def test_parse_silence() -> None:
    """Test parsing silence (~)."""
    result = parse_pattern("bd ~ sd")
    assert isinstance(result.unwrap, PatSeq)

    children = list(result.unwrap.pats)
    assert len(children) == 3
    assert isinstance(children[0].unwrap, PatPure)
    assert isinstance(children[1].unwrap, PatSilent)
    assert isinstance(children[2].unwrap, PatPure)
    assert_string_value(children[0].unwrap.value, "bd")
    assert_string_value(children[2].unwrap.value, "sd")


def test_parse_sample_selection() -> None:
    """Test parsing sample selection (symbol:number)."""
    result = parse_pattern("bd:2 sd:0")
    assert isinstance(result.unwrap, PatSeq)

    children = list(result.unwrap.pats)
    assert len(children) == 2
    assert isinstance(children[0].unwrap, PatPure)
    selected_0 = children[0].unwrap.value
    assert selected_0 == "bd:2"
    assert isinstance(children[1].unwrap, PatPure)
    selected_1 = children[1].unwrap.value
    assert selected_1 == "sd:0"


def test_parse_sample_selection_strings() -> None:
    """Test parsing sample selection with string selectors."""
    result = parse_pattern("bd:kick sd:snare")
    assert isinstance(result.unwrap, PatSeq)

    children = list(result.unwrap.pats)
    assert len(children) == 2
    assert isinstance(children[0].unwrap, PatPure)
    selected_0 = children[0].unwrap.value
    assert selected_0 == "bd:kick"
    assert isinstance(children[1].unwrap, PatPure)
    selected_1 = children[1].unwrap.value
    assert selected_1 == "sd:snare"


def test_parse_seq_brackets() -> None:
    """Test parsing seqed patterns with brackets."""
    result = parse_pattern("[bd sd] cp")
    assert isinstance(result.unwrap, PatSeq)
    # TODO assert whatever


# def test_parse_seq_dots():
#     """Test parsing seqed patterns with dots."""
#     # TODO: Dots are not currently supported in the grammar
#     result = parse_pattern(".bd sd. cp")
#     assert isinstance(result.unwrap, PatSeq)
#     # TODO assert whatever


def test_parse_choice_pattern() -> None:
    """Test parsing choice patterns [a|b|c]."""
    result = parse_pattern("[bd|sd|cp]")
    # Should return choice pattern
    assert isinstance(result.unwrap, PatRand)
    choices = list(result.unwrap.pats)
    assert len(choices) == 3
    assert_string_value(choices[0].unwrap.value, "bd")
    assert_string_value(choices[1].unwrap.value, "sd")
    assert_string_value(choices[2].unwrap.value, "cp")


def test_parse_parallel_pattern() -> None:
    """Test parsing parallel patterns [a,b,c]."""
    result = parse_pattern("[bd,sd,cp]")
    # Should return parallel pattern
    assert isinstance(result.unwrap, PatPar)
    patterns = list(result.unwrap.pats)
    assert len(patterns) == 3
    assert_string_value(patterns[0].unwrap.value, "bd")
    assert_string_value(patterns[1].unwrap.value, "sd")
    assert_string_value(patterns[2].unwrap.value, "cp")


def test_parse_alternating_pattern() -> None:
    """Test parsing alternating patterns <a b c>."""
    result = parse_pattern("<bd sd cp>")
    # Should return alternating pattern
    assert isinstance(result.unwrap, PatAlt)
    patterns = list(result.unwrap.pats)
    assert len(patterns) == 3
    assert_string_value(patterns[0].unwrap.value, "bd")
    assert_string_value(patterns[1].unwrap.value, "sd")
    assert_string_value(patterns[2].unwrap.value, "cp")


def test_parse_repetition_multiply() -> None:
    """Test parsing repetition with * operator."""
    result = parse_pattern("bd*2")
    assert isinstance(result.unwrap, PatSpeed)
    assert result.unwrap.op == SpeedOp.Fast
    assert result.unwrap.factor == 2
    assert_string_value(result.unwrap.pat.unwrap.value, "bd")


def test_parse_repetition_divide() -> None:
    """Test parsing repetition with / operator (slowdown)."""
    result = parse_pattern("bd/2")
    assert isinstance(result.unwrap, PatSpeed)
    assert result.unwrap.op == SpeedOp.Slow
    assert result.unwrap.factor == 2
    assert_string_value(result.unwrap.pat.unwrap.value, "bd")


def test_parse_multiple_repetition() -> None:
    """Test parsing multiple repetition operators."""
    result = parse_pattern("bd*2/4")
    # Should be a repetition with SLOW operator
    assert isinstance(result.unwrap, PatSpeed)
    assert result.unwrap.op == SpeedOp.Slow
    assert result.unwrap.factor == 4

    # The child should be the multiplication repetition
    child = result.unwrap.pat
    assert isinstance(child.unwrap, PatSpeed)
    assert child.unwrap.op == SpeedOp.Fast
    assert child.unwrap.factor == 2


def test_parse_elongation() -> None:
    """Test parsing stretch with _ or @."""
    result = parse_pattern("bd_")
    assert isinstance(result.unwrap, PatStretch)
    assert result.unwrap.count == 2  # bd_ = stretch by 2
    assert_string_value(result.unwrap.pat.unwrap.value, "bd")

    result2 = parse_pattern("bd__")
    assert isinstance(result2.unwrap, PatStretch)
    assert result2.unwrap.count == 3  # bd__ = stretch by 3
    assert_string_value(result2.unwrap.pat.unwrap.value, "bd")


def test_stretch_equivalence() -> None:
    """Test that different stretch notations are equivalent."""
    # All these forms should create stretch with count=3
    patterns = ["x@3", "x__", "x _ _", "x_ _"]

    for pattern_str in patterns:
        result = parse_pattern(pattern_str)
        # Handle both direct stretch and sequence with single stretched element
        if isinstance(result.unwrap, PatStretch):
            assert result.unwrap.count == 3, (
                f"Pattern '{pattern_str}' should have count=3, got {result.unwrap.count}"
            )
        elif isinstance(result.unwrap, PatSeq) and len(list(result.unwrap.pats)) == 1:
            child = next(iter(result.unwrap.pats))
            assert isinstance(child.unwrap, PatStretch)
            assert child.unwrap.count == 3, (
                f"Pattern '{pattern_str}' should have count=3, got {child.unwrap.count}"
            )
        else:
            assert False, (
                f"Pattern '{pattern_str}' has unexpected structure: {result.unwrap}"
            )


def test_symbol_with_colon() -> None:
    """Test that a:b is a valid identifier."""
    result = parse_pattern("a:b")
    assert isinstance(result.unwrap, PatPure)
    assert result.unwrap.value == "a:b"

    # Test in sequences too
    result2 = parse_pattern("a:b c:d")
    assert isinstance(result2.unwrap, PatSeq)
    children = list(result2.unwrap.pats)
    assert len(children) == 2
    assert children[0].unwrap.value == "a:b"
    assert children[1].unwrap.value == "c:d"


def test_rests_without_spacing() -> None:
    """Test that rests (~) work without spacing."""
    test_patterns = [
        ("a~b", ["a", "~", "b"]),
        ("~a~", ["~", "a", "~"]),
        ("a~b~c", ["a", "~", "b", "~", "c"]),
    ]

    for pattern_str, expected_values in test_patterns:
        result = parse_pattern(pattern_str)
        assert isinstance(result.unwrap, PatSeq), (
            f"Pattern '{pattern_str}' should be a sequence"
        )

        children = list(result.unwrap.pats)
        assert len(children) == len(expected_values), (
            f"Pattern '{pattern_str}' should have {len(expected_values)} elements"
        )

        for i, (child, expected) in enumerate(zip(children, expected_values)):
            if expected == "~":
                assert isinstance(child.unwrap, PatSilent), (
                    f"Element {i} in '{pattern_str}' should be silence"
                )
            else:
                assert isinstance(child.unwrap, PatPure), (
                    f"Element {i} in '{pattern_str}' should be pure value"
                )
                assert child.unwrap.value == expected, (
                    f"Element {i} in '{pattern_str}' should be '{expected}'"
                )


def test_parse_probability() -> None:
    """Test parsing probability with ?."""
    result = parse_pattern("bd?")
    assert isinstance(result.unwrap, PatProb)
    assert_string_value(result.unwrap.pat.unwrap.value, "bd")
    assert result.unwrap.chance == 0.5


def test_parse_euclidean_rhythm() -> None:
    """Test parsing Euclidean rhythms."""
    result = parse_pattern("bd(3,8)")
    assert isinstance(result.unwrap, PatEuc)
    assert_string_value(result.unwrap.pat.unwrap.value, "bd")
    assert result.unwrap.hits == 3
    assert result.unwrap.steps == 8
    assert result.unwrap.rotation == 0


def test_parse_euclidean_with_rotation() -> None:
    """Test parsing Euclidean rhythms with rotation."""
    result = parse_pattern("bd(3,8,1)")
    assert isinstance(result.unwrap, PatEuc)
    assert_string_value(result.unwrap.pat.unwrap.value, "bd")
    assert result.unwrap.hits == 3
    assert result.unwrap.steps == 8
    assert result.unwrap.rotation == 1


def test_parse_polymetric() -> None:
    """Test parsing polymetric sequences."""
    result = parse_pattern("{bd, sd}")
    assert isinstance(result.unwrap, PatPoly)

    patterns = list(result.unwrap.pats)
    assert len(patterns) == 2
    assert_string_value(patterns[0].unwrap.value, "bd")
    assert_string_value(patterns[1].unwrap.value, "sd")


def test_parse_complex_pattern() -> None:
    """Test parsing a complex pattern with multiple features."""
    result = parse_pattern("bd*2 [sd cp] ~ {hh, oh}")
    assert isinstance(result.unwrap, PatSeq)

    children = list(result.unwrap.pats)
    assert len(children) == 4

    # First: bd*2 (should be repetition)
    assert isinstance(children[0].unwrap, PatSpeed)

    # Second: [sd cp] (should be sequence)
    assert isinstance(children[1].unwrap, PatSeq)

    # Third: ~ (should be silence)
    assert isinstance(children[2].unwrap, PatSilent)

    # Fourth: {hh, oh} (should be polymetric)
    assert isinstance(children[3].unwrap, PatPoly)


def test_parse_nested_groups() -> None:
    """Test parsing nested grouping structures."""
    result = parse_pattern("[[bd sd] cp] hh")
    assert isinstance(result.unwrap, PatSeq)

    children = list(result.unwrap.pats)
    assert len(children) == 2

    # First should be nested sequence
    nested = children[0]
    assert isinstance(nested.unwrap, PatSeq)

    # Second should be "hh"
    assert_string_value(children[1].unwrap.value, "hh")


def test_parse_empty_pattern() -> None:
    """Test parsing edge cases."""
    with pytest.raises(Exception):
        parse_pattern("")


def test_parse_whitespace_handling() -> None:
    """Test that whitespace is properly ignored."""
    result1 = parse_pattern("bd sd")
    result2 = parse_pattern("bd    sd")
    result3 = parse_pattern("  bd   sd  ")

    # All should produce equivalent patterns
    for result in [result1, result2, result3]:
        assert isinstance(result.unwrap, PatSeq)
        children = list(result.unwrap.pats)
        assert len(children) == 2
        assert_string_value(children[0].unwrap.value, "bd")
        assert_string_value(children[1].unwrap.value, "sd")


def test_parse_numeric_symbols() -> None:
    """Test parsing symbols with numbers."""
    result = parse_pattern("bd909 tr808")
    assert isinstance(result.unwrap, PatSeq)

    children = list(result.unwrap.pats)
    assert len(children) == 2
    assert_string_value(children[0].unwrap.value, "bd909")
    assert_string_value(children[1].unwrap.value, "tr808")


def test_parse_symbols_with_underscores_and_dashes() -> None:
    """Test parsing symbols with underscores and dashes."""
    result = parse_pattern("bd_kick snare-roll hi_hat_open bass-drum_1")
    assert isinstance(result.unwrap, PatSeq)

    children = list(result.unwrap.pats)
    assert len(children) == 4
    assert_string_value(children[0].unwrap.value, "bd_kick")
    assert_string_value(children[1].unwrap.value, "snare-roll")
    assert_string_value(children[2].unwrap.value, "hi_hat_open")
    assert_string_value(children[3].unwrap.value, "bass-drum_1")


def test_parse_complex_sample_selection() -> None:
    """Test complex sample selection patterns."""
    result = parse_pattern("bd:0*2 sd:1/2")
    assert isinstance(result.unwrap, PatSeq)

    children = list(result.unwrap.pats)
    assert len(children) == 2

    # First should be bd:0*2 (repetition of sample selection)
    first = children[0]
    assert isinstance(first.unwrap, PatSpeed)
    assert first.unwrap.op == SpeedOp.Fast
    assert first.unwrap.factor == 2
    assert isinstance(first.unwrap.pat.unwrap, PatPure)
    selected_first = first.unwrap.pat.unwrap.value
    assert selected_first == "bd:0"

    # Second should be sd:1/2 (repetition with division of sample selection)
    second = children[1]
    assert isinstance(second.unwrap, PatSpeed)
    assert second.unwrap.op == SpeedOp.Slow
    assert second.unwrap.factor == 2
    assert isinstance(second.unwrap.pat.unwrap, PatPure)
    selected_second = second.unwrap.pat.unwrap.value
    assert selected_second == "sd:1"


def test_kick_snare_pattern() -> None:
    """Test basic kick-snare pattern."""
    result = parse_pattern("bd sd bd sd")
    assert isinstance(result.unwrap, PatSeq)
    children = list(result.unwrap.pats)
    assert len(children) == 4
    # Check each pattern value individually
    assert_string_value(children[0].unwrap.value, "bd")
    assert_string_value(children[1].unwrap.value, "sd")
    assert_string_value(children[2].unwrap.value, "bd")
    assert_string_value(children[3].unwrap.value, "sd")


def test_hihat_subdivision() -> None:
    """Test hihat subdivision pattern."""
    result = parse_pattern("bd [hh hh] sd [hh hh]")
    assert isinstance(result.unwrap, PatSeq)
    children = list(result.unwrap.pats)
    assert len(children) == 4

    # Check that the hihat groups are properly parsed as sequences
    assert isinstance(children[1].unwrap, PatSeq)
    assert isinstance(children[3].unwrap, PatSeq)


def test_polyrhythmic_pattern() -> None:
    """Test polyrhythmic pattern."""
    result = parse_pattern("{bd sd, hh*8}")
    assert isinstance(result.unwrap, PatPoly)

    patterns = list(result.unwrap.pats)
    assert len(patterns) == 2

    # First should be "bd sd" sequence
    assert isinstance(patterns[0].unwrap, PatSeq)

    # Second should be "hh*8" (repetition)
    assert isinstance(patterns[1].unwrap, PatSpeed)


# New TidalCycles features tests


def test_parse_replicate() -> None:
    """Test parsing replicate patterns."""
    result = parse_pattern("bd!3")
    assert isinstance(result.unwrap, PatRepeat)
    assert result.unwrap.count == 3
    assert isinstance(result.unwrap.pat.unwrap, PatPure)
    assert_string_value(result.unwrap.pat.unwrap.value, "bd")


def test_parse_ratio() -> None:
    """Test parsing fractional repetition patterns."""
    result = parse_pattern("bd*3%2")
    assert isinstance(result.unwrap, PatSpeed)
    assert result.unwrap.op == SpeedOp.Fast
    assert result.unwrap.factor == Fraction(3, 2)
    assert isinstance(result.unwrap.pat.unwrap, PatPure)
    assert_string_value(result.unwrap.pat.unwrap.value, "bd")


def test_parse_polymetric_subdivision() -> None:
    """Test parsing polymetric subdivision patterns."""
    result = parse_pattern("{bd, sd}%4")
    assert isinstance(result.unwrap, PatPoly)
    assert result.unwrap.subdiv == 4
    assert len(result.unwrap.pats) == 2


def test_parse_dot_grouping() -> None:
    """Test parsing dot grouping patterns."""
    result = parse_pattern("bd sd . hh cp")
    assert isinstance(result.unwrap, PatSeq)
    assert len(result.unwrap.pats) == 2
    # Both sides should be sequences
    assert isinstance(result.unwrap.pats[0].unwrap, PatSeq)
    assert isinstance(result.unwrap.pats[1].unwrap, PatSeq)


def test_parse_complex_new_features() -> None:
    """Test parsing complex combinations of new features."""
    # Replicate with selection
    result = parse_pattern("bd:0!2")
    assert isinstance(result.unwrap, PatRepeat)
    assert isinstance(result.unwrap.pat.unwrap, PatPure)
    selected = result.unwrap.pat.unwrap.value
    assert selected == "bd:0"

    # Fractional repetition with probability
    result = parse_pattern("bd?*2%3")
    assert isinstance(result.unwrap, PatSpeed)
    assert isinstance(result.unwrap.pat.unwrap, PatProb)

    # Polymetric subdivision with replicate inside
    result = parse_pattern("{bd!2, sd}%4")
    assert isinstance(result.unwrap, PatPoly)
    patterns = list(result.unwrap.pats)
    assert isinstance(patterns[0].unwrap, PatRepeat)


def test_parse_nested_new_features() -> None:
    """Test parsing nested new features."""
    # Replicate of sequence
    result = parse_pattern("[bd sd]!2")
    assert isinstance(result.unwrap, PatRepeat)
    assert isinstance(result.unwrap.pat.unwrap, PatSeq)

    # Fractional repetition of choice
    result = parse_pattern("[bd | sd]*2%1")
    assert isinstance(result.unwrap, PatSpeed)
    assert isinstance(result.unwrap.pat.unwrap, PatRand)
