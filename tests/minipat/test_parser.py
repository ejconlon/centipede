from fractions import Fraction

import pytest

from minipat.parser import parse_pattern
from minipat.pat import (
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
    PatSeq,
    PatSilence,
    RepetitionOp,
)


def assert_string_value(value, expected_value):
    """Helper function to assert string values."""
    assert value == expected_value


def test_parse_basic_symbol():
    """Test parsing a single symbol."""
    result = parse_pattern("bd")
    assert isinstance(result.unwrap, PatPure)
    assert_string_value(result.unwrap.value, "bd")


def test_parse_simple_sequence():
    """Test parsing a simple sequence of symbols."""
    result = parse_pattern("bd sd hh")
    assert isinstance(result.unwrap, PatSeq)

    children = list(result.unwrap.patterns)
    assert len(children) == 3
    assert all(isinstance(child.unwrap, PatPure) for child in children)
    assert_string_value(children[0].unwrap.value, "bd")
    assert_string_value(children[1].unwrap.value, "sd")
    assert_string_value(children[2].unwrap.value, "hh")


def test_parse_silence():
    """Test parsing silence (~)."""
    result = parse_pattern("bd ~ sd")
    assert isinstance(result.unwrap, PatSeq)

    children = list(result.unwrap.patterns)
    assert len(children) == 3
    assert isinstance(children[0].unwrap, PatPure)
    assert isinstance(children[1].unwrap, PatSilence)
    assert isinstance(children[2].unwrap, PatPure)
    assert_string_value(children[0].unwrap.value, "bd")
    assert_string_value(children[2].unwrap.value, "sd")


def test_parse_sample_selection():
    """Test parsing sample selection (symbol:number)."""
    result = parse_pattern("bd:2 sd:0")
    assert isinstance(result.unwrap, PatSeq)

    children = list(result.unwrap.patterns)
    assert len(children) == 2
    assert isinstance(children[0].unwrap, PatPure)
    selected_0 = children[0].unwrap.value
    assert selected_0 == "bd:2"
    assert isinstance(children[1].unwrap, PatPure)
    selected_1 = children[1].unwrap.value
    assert selected_1 == "sd:0"


def test_parse_sample_selection_strings():
    """Test parsing sample selection with string selectors."""
    result = parse_pattern("bd:kick sd:snare")
    assert isinstance(result.unwrap, PatSeq)

    children = list(result.unwrap.patterns)
    assert len(children) == 2
    assert isinstance(children[0].unwrap, PatPure)
    selected_0 = children[0].unwrap.value
    assert selected_0 == "bd:kick"
    assert isinstance(children[1].unwrap, PatPure)
    selected_1 = children[1].unwrap.value
    assert selected_1 == "sd:snare"


def test_parse_seq_brackets():
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


def test_parse_choice_pattern():
    """Test parsing choice patterns [a|b|c]."""
    result = parse_pattern("[bd|sd|cp]")
    # Should return choice pattern
    assert isinstance(result.unwrap, PatChoice)
    choices = list(result.unwrap.patterns)
    assert len(choices) == 3
    assert_string_value(choices[0].unwrap.value, "bd")
    assert_string_value(choices[1].unwrap.value, "sd")
    assert_string_value(choices[2].unwrap.value, "cp")


def test_parse_parallel_pattern():
    """Test parsing parallel patterns [a,b,c]."""
    result = parse_pattern("[bd,sd,cp]")
    # Should return parallel pattern
    assert isinstance(result.unwrap, PatPar)
    patterns = list(result.unwrap.patterns)
    assert len(patterns) == 3
    assert_string_value(patterns[0].unwrap.value, "bd")
    assert_string_value(patterns[1].unwrap.value, "sd")
    assert_string_value(patterns[2].unwrap.value, "cp")


def test_parse_alternating_pattern():
    """Test parsing alternating patterns <a b c>."""
    result = parse_pattern("<bd sd cp>")
    # Should return alternating pattern
    assert isinstance(result.unwrap, PatAlternating)
    patterns = list(result.unwrap.patterns)
    assert len(patterns) == 3
    assert_string_value(patterns[0].unwrap.value, "bd")
    assert_string_value(patterns[1].unwrap.value, "sd")
    assert_string_value(patterns[2].unwrap.value, "cp")


def test_parse_repetition_multiply():
    """Test parsing repetition with * operator."""
    result = parse_pattern("bd*2")
    assert isinstance(result.unwrap, PatRepetition)
    assert result.unwrap.operator == RepetitionOp.Fast
    assert result.unwrap.count == 2
    assert_string_value(result.unwrap.pattern.unwrap.value, "bd")


def test_parse_repetition_divide():
    """Test parsing repetition with / operator (slowdown)."""
    result = parse_pattern("bd/2")
    assert isinstance(result.unwrap, PatRepetition)
    assert result.unwrap.operator == RepetitionOp.Slow
    assert result.unwrap.count == 2
    assert_string_value(result.unwrap.pattern.unwrap.value, "bd")


def test_parse_multiple_repetition():
    """Test parsing multiple repetition operators."""
    result = parse_pattern("bd*2/4")
    # Should be a repetition with SLOW operator
    assert isinstance(result.unwrap, PatRepetition)
    assert result.unwrap.operator == RepetitionOp.Slow
    assert result.unwrap.count == 4

    # The child should be the multiplication repetition
    child = result.unwrap.pattern
    assert isinstance(child.unwrap, PatRepetition)
    assert child.unwrap.operator == RepetitionOp.Fast
    assert child.unwrap.count == 2


def test_parse_elongation():
    """Test parsing elongation with _ or @."""
    result = parse_pattern("bd_")
    assert isinstance(result.unwrap, PatElongation)
    assert result.unwrap.count == 1  # Single elongation
    assert_string_value(result.unwrap.pattern.unwrap.value, "bd")

    result2 = parse_pattern("bd__")
    assert isinstance(result2.unwrap, PatElongation)
    assert result2.unwrap.count == 2  # Double elongation
    assert_string_value(result2.unwrap.pattern.unwrap.value, "bd")


def test_parse_probability():
    """Test parsing probability with ?."""
    result = parse_pattern("bd?")
    assert isinstance(result.unwrap, PatProbability)
    assert_string_value(result.unwrap.pattern.unwrap.value, "bd")
    assert result.unwrap.probability == 0.5


def test_parse_euclidean_rhythm():
    """Test parsing Euclidean rhythms."""
    result = parse_pattern("bd(3,8)")
    assert isinstance(result.unwrap, PatEuclidean)
    assert_string_value(result.unwrap.pattern.unwrap.value, "bd")
    assert result.unwrap.hits == 3
    assert result.unwrap.steps == 8
    assert result.unwrap.rotation == 0


def test_parse_euclidean_with_rotation():
    """Test parsing Euclidean rhythms with rotation."""
    result = parse_pattern("bd(3,8,1)")
    assert isinstance(result.unwrap, PatEuclidean)
    assert_string_value(result.unwrap.pattern.unwrap.value, "bd")
    assert result.unwrap.hits == 3
    assert result.unwrap.steps == 8
    assert result.unwrap.rotation == 1


def test_parse_polymetric():
    """Test parsing polymetric sequences."""
    result = parse_pattern("{bd, sd}")
    assert isinstance(result.unwrap, PatPolymetric)

    patterns = list(result.unwrap.patterns)
    assert len(patterns) == 2
    assert_string_value(patterns[0].unwrap.value, "bd")
    assert_string_value(patterns[1].unwrap.value, "sd")


def test_parse_complex_pattern():
    """Test parsing a complex pattern with multiple features."""
    result = parse_pattern("bd*2 [sd cp] ~ {hh, oh}")
    assert isinstance(result.unwrap, PatSeq)

    children = list(result.unwrap.patterns)
    assert len(children) == 4

    # First: bd*2 (should be repetition)
    assert isinstance(children[0].unwrap, PatRepetition)

    # Second: [sd cp] (should be sequence)
    assert isinstance(children[1].unwrap, PatSeq)

    # Third: ~ (should be silence)
    assert isinstance(children[2].unwrap, PatSilence)

    # Fourth: {hh, oh} (should be polymetric)
    assert isinstance(children[3].unwrap, PatPolymetric)


def test_parse_nested_groups():
    """Test parsing nested grouping structures."""
    result = parse_pattern("[[bd sd] cp] hh")
    assert isinstance(result.unwrap, PatSeq)

    children = list(result.unwrap.patterns)
    assert len(children) == 2

    # First should be nested sequence
    nested = children[0]
    assert isinstance(nested.unwrap, PatSeq)

    # Second should be "hh"
    assert_string_value(children[1].unwrap.value, "hh")


def test_parse_empty_pattern():
    """Test parsing edge cases."""
    with pytest.raises(Exception):
        parse_pattern("")


def test_parse_whitespace_handling():
    """Test that whitespace is properly ignored."""
    result1 = parse_pattern("bd sd")
    result2 = parse_pattern("bd    sd")
    result3 = parse_pattern("  bd   sd  ")

    # All should produce equivalent patterns
    for result in [result1, result2, result3]:
        assert isinstance(result.unwrap, PatSeq)
        children = list(result.unwrap.patterns)
        assert len(children) == 2
        assert_string_value(children[0].unwrap.value, "bd")
        assert_string_value(children[1].unwrap.value, "sd")


def test_parse_numeric_symbols():
    """Test parsing symbols with numbers."""
    result = parse_pattern("bd909 tr808")
    assert isinstance(result.unwrap, PatSeq)

    children = list(result.unwrap.patterns)
    assert len(children) == 2
    assert_string_value(children[0].unwrap.value, "bd909")
    assert_string_value(children[1].unwrap.value, "tr808")


def test_parse_symbols_with_underscores_and_dashes():
    """Test parsing symbols with underscores and dashes."""
    result = parse_pattern("bd_kick snare-roll hi_hat_open bass-drum_1")
    assert isinstance(result.unwrap, PatSeq)

    children = list(result.unwrap.patterns)
    assert len(children) == 4
    assert_string_value(children[0].unwrap.value, "bd_kick")
    assert_string_value(children[1].unwrap.value, "snare-roll")
    assert_string_value(children[2].unwrap.value, "hi_hat_open")
    assert_string_value(children[3].unwrap.value, "bass-drum_1")


def test_parse_complex_sample_selection():
    """Test complex sample selection patterns."""
    result = parse_pattern("bd:0*2 sd:1/2")
    assert isinstance(result.unwrap, PatSeq)

    children = list(result.unwrap.patterns)
    assert len(children) == 2

    # First should be bd:0*2 (repetition of sample selection)
    first = children[0]
    assert isinstance(first.unwrap, PatRepetition)
    assert first.unwrap.operator == RepetitionOp.Fast
    assert first.unwrap.count == 2
    assert isinstance(first.unwrap.pattern.unwrap, PatPure)
    selected_first = first.unwrap.pattern.unwrap.value
    assert selected_first == "bd:0"

    # Second should be sd:1/2 (repetition with division of sample selection)
    second = children[1]
    assert isinstance(second.unwrap, PatRepetition)
    assert second.unwrap.operator == RepetitionOp.Slow
    assert second.unwrap.count == 2
    assert isinstance(second.unwrap.pattern.unwrap, PatPure)
    selected_second = second.unwrap.pattern.unwrap.value
    assert selected_second == "sd:1"


def test_kick_snare_pattern():
    """Test basic kick-snare pattern."""
    result = parse_pattern("bd sd bd sd")
    assert isinstance(result.unwrap, PatSeq)
    children = list(result.unwrap.patterns)
    assert len(children) == 4
    # Check each pattern value individually
    assert_string_value(children[0].unwrap.value, "bd")
    assert_string_value(children[1].unwrap.value, "sd")
    assert_string_value(children[2].unwrap.value, "bd")
    assert_string_value(children[3].unwrap.value, "sd")


def test_hihat_subdivision():
    """Test hihat subdivision pattern."""
    result = parse_pattern("bd [hh hh] sd [hh hh]")
    assert isinstance(result.unwrap, PatSeq)
    children = list(result.unwrap.patterns)
    assert len(children) == 4

    # Check that the hihat groups are properly parsed as sequences
    assert isinstance(children[1].unwrap, PatSeq)
    assert isinstance(children[3].unwrap, PatSeq)


def test_polyrhythmic_pattern():
    """Test polyrhythmic pattern."""
    result = parse_pattern("{bd sd, hh*8}")
    assert isinstance(result.unwrap, PatPolymetric)

    patterns = list(result.unwrap.patterns)
    assert len(patterns) == 2

    # First should be "bd sd" sequence
    assert isinstance(patterns[0].unwrap, PatSeq)

    # Second should be "hh*8" (repetition)
    assert isinstance(patterns[1].unwrap, PatRepetition)


# New TidalCycles features tests


def test_parse_replicate():
    """Test parsing replicate patterns."""
    result = parse_pattern("bd!3")
    assert isinstance(result.unwrap, PatReplicate)
    assert result.unwrap.count == 3
    assert isinstance(result.unwrap.pattern.unwrap, PatPure)
    assert_string_value(result.unwrap.pattern.unwrap.value, "bd")


def test_parse_ratio():
    """Test parsing fractional repetition patterns."""
    result = parse_pattern("bd*3%2")
    assert isinstance(result.unwrap, PatRepetition)
    assert result.unwrap.operator == RepetitionOp.Fast
    assert result.unwrap.count == Fraction(3, 2)
    assert isinstance(result.unwrap.pattern.unwrap, PatPure)
    assert_string_value(result.unwrap.pattern.unwrap.value, "bd")


def test_parse_polymetric_subdivision():
    """Test parsing polymetric subdivision patterns."""
    result = parse_pattern("{bd, sd}%4")
    assert isinstance(result.unwrap, PatPolymetric)
    assert result.unwrap.subdivision == 4
    assert len(result.unwrap.patterns) == 2


def test_parse_dot_grouping():
    """Test parsing dot grouping patterns."""
    result = parse_pattern("bd sd . hh cp")
    assert isinstance(result.unwrap, PatSeq)
    assert len(result.unwrap.patterns) == 2
    # Both sides should be sequences
    assert isinstance(result.unwrap.patterns[0].unwrap, PatSeq)
    assert isinstance(result.unwrap.patterns[1].unwrap, PatSeq)


def test_parse_complex_new_features():
    """Test parsing complex combinations of new features."""
    # Replicate with selection
    result = parse_pattern("bd:0!2")
    assert isinstance(result.unwrap, PatReplicate)
    assert isinstance(result.unwrap.pattern.unwrap, PatPure)
    selected = result.unwrap.pattern.unwrap.value
    assert selected == "bd:0"

    # Fractional repetition with probability
    result = parse_pattern("bd?*2%3")
    assert isinstance(result.unwrap, PatRepetition)
    assert isinstance(result.unwrap.pattern.unwrap, PatProbability)

    # Polymetric subdivision with replicate inside
    result = parse_pattern("{bd!2, sd}%4")
    assert isinstance(result.unwrap, PatPolymetric)
    patterns = list(result.unwrap.patterns)
    assert isinstance(patterns[0].unwrap, PatReplicate)


def test_parse_nested_new_features():
    """Test parsing nested new features."""
    # Replicate of sequence
    result = parse_pattern("[bd sd]!2")
    assert isinstance(result.unwrap, PatReplicate)
    assert isinstance(result.unwrap.pattern.unwrap, PatSeq)

    # Fractional repetition of choice
    result = parse_pattern("[bd | sd]*2%1")
    assert isinstance(result.unwrap, PatRepetition)
    assert isinstance(result.unwrap.pattern.unwrap, PatChoice)
