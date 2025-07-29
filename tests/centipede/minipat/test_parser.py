import pytest

from centipede.minipat.parser import parse_pattern
from centipede.minipat.pat import (
    PatChoice,
    PatElongation,
    PatEuclidean,
    PatGroup,
    PatPolymetric,
    PatProbability,
    PatPure,
    PatRepetition,
    PatSelect,
    PatSeq,
    PatSilence,
    RepetitionOp,
)


def test_parse_basic_symbol():
    """Test parsing a single symbol."""
    result = parse_pattern("bd")
    assert isinstance(result.unwrap, PatPure)
    assert result.unwrap.val == "bd"


def test_parse_simple_sequence():
    """Test parsing a simple sequence of symbols."""
    result = parse_pattern("bd sd hh")
    assert isinstance(result.unwrap, PatSeq)

    children = list(result.unwrap.children)
    assert len(children) == 3
    assert all(isinstance(child.unwrap, PatPure) for child in children)
    assert children[0].unwrap.val == "bd"
    assert children[1].unwrap.val == "sd"
    assert children[2].unwrap.val == "hh"


def test_parse_silence():
    """Test parsing silence (~)."""
    result = parse_pattern("bd ~ sd")
    assert isinstance(result.unwrap, PatSeq)

    children = list(result.unwrap.children)
    assert len(children) == 3
    assert isinstance(children[0].unwrap, PatPure)
    assert isinstance(children[1].unwrap, PatSilence)
    assert isinstance(children[2].unwrap, PatPure)
    assert children[0].unwrap.val == "bd"
    assert children[2].unwrap.val == "sd"


def test_parse_sample_selection():
    """Test parsing sample selection (symbol:number)."""
    result = parse_pattern("bd:2 sd:0")
    assert isinstance(result.unwrap, PatSeq)

    children = list(result.unwrap.children)
    assert len(children) == 2
    assert isinstance(children[0].unwrap, PatSelect)
    assert children[0].unwrap.pattern.unwrap.val == "bd"
    assert children[0].unwrap.selector == "2"
    assert isinstance(children[1].unwrap, PatSelect)
    assert children[1].unwrap.pattern.unwrap.val == "sd"
    assert children[1].unwrap.selector == "0"


def test_parse_group_brackets():
    """Test parsing grouped patterns with brackets."""
    result = parse_pattern("[bd sd] cp")
    assert isinstance(result.unwrap, PatSeq)

    children = list(result.unwrap.children)
    assert len(children) == 2

    # First element should be the grouped pattern
    first_group = children[0]
    assert isinstance(first_group.unwrap, PatGroup)
    # The grouped pattern should contain a sequence
    inner_pattern = first_group.unwrap.pattern
    assert isinstance(inner_pattern.unwrap, PatSeq)
    group_children = list(inner_pattern.unwrap.children)
    assert len(group_children) == 2
    assert group_children[0].unwrap.val == "bd"
    assert group_children[1].unwrap.val == "sd"

    # Second element should be "cp"
    assert children[1].unwrap.val == "cp"


def test_parse_group_dots():
    """Test parsing grouped patterns with dots."""
    result = parse_pattern(".bd sd. cp")
    assert isinstance(result.unwrap, PatSeq)

    children = list(result.unwrap.children)
    assert len(children) == 2

    # First element should be the grouped pattern
    first_group = children[0]
    assert isinstance(first_group.unwrap, PatGroup)


def test_parse_choice_pattern():
    """Test parsing choice patterns [a|b|c]."""
    result = parse_pattern("[bd|sd|cp]")
    # Should return choice pattern
    assert isinstance(result.unwrap, PatChoice)
    choices = list(result.unwrap.choices)
    assert len(choices) == 3
    assert choices[0].unwrap.val == "bd"
    assert choices[1].unwrap.val == "sd"
    assert choices[2].unwrap.val == "cp"


def test_parse_repetition_multiply():
    """Test parsing repetition with * operator."""
    result = parse_pattern("bd*2")
    assert isinstance(result.unwrap, PatRepetition)
    assert result.unwrap.operator == RepetitionOp.FAST
    assert result.unwrap.count == 2
    assert result.unwrap.pattern.unwrap.val == "bd"


def test_parse_repetition_divide():
    """Test parsing repetition with / operator (slowdown)."""
    result = parse_pattern("bd/2")
    assert isinstance(result.unwrap, PatRepetition)
    assert result.unwrap.operator == RepetitionOp.SLOW
    assert result.unwrap.count == 2
    assert result.unwrap.pattern.unwrap.val == "bd"


def test_parse_multiple_repetition():
    """Test parsing multiple repetition operators."""
    result = parse_pattern("bd*2/4")
    # Should be a repetition with SLOW operator
    assert isinstance(result.unwrap, PatRepetition)
    assert result.unwrap.operator == RepetitionOp.SLOW
    assert result.unwrap.count == 4

    # The child should be the multiplication repetition
    child = result.unwrap.pattern
    assert isinstance(child.unwrap, PatRepetition)
    assert child.unwrap.operator == RepetitionOp.FAST
    assert child.unwrap.count == 2


def test_parse_elongation():
    """Test parsing elongation with _ or @."""
    result = parse_pattern("bd_")
    assert isinstance(result.unwrap, PatElongation)
    assert result.unwrap.count == 1  # Single elongation
    assert result.unwrap.pattern.unwrap.val == "bd"

    result2 = parse_pattern("bd__")
    assert isinstance(result2.unwrap, PatElongation)
    assert result2.unwrap.count == 2  # Double elongation
    assert result2.unwrap.pattern.unwrap.val == "bd"


def test_parse_probability():
    """Test parsing probability with ?."""
    result = parse_pattern("bd?")
    assert isinstance(result.unwrap, PatProbability)
    assert result.unwrap.pattern.unwrap.val == "bd"
    assert result.unwrap.probability == 0.5


def test_parse_euclidean_rhythm():
    """Test parsing Euclidean rhythms."""
    result = parse_pattern("bd(3,8)")
    assert isinstance(result.unwrap, PatEuclidean)
    assert result.unwrap.atom.unwrap.val == "bd"
    assert result.unwrap.hits == 3
    assert result.unwrap.steps == 8
    assert result.unwrap.rotation == 0


def test_parse_euclidean_with_rotation():
    """Test parsing Euclidean rhythms with rotation."""
    result = parse_pattern("bd(3,8,1)")
    assert isinstance(result.unwrap, PatEuclidean)
    assert result.unwrap.atom.unwrap.val == "bd"
    assert result.unwrap.hits == 3
    assert result.unwrap.steps == 8
    assert result.unwrap.rotation == 1


def test_parse_polymetric():
    """Test parsing polymetric sequences."""
    result = parse_pattern("{bd, sd}")
    assert isinstance(result.unwrap, PatPolymetric)

    patterns = list(result.unwrap.patterns)
    assert len(patterns) == 2
    assert patterns[0].unwrap.val == "bd"
    assert patterns[1].unwrap.val == "sd"


def test_parse_complex_pattern():
    """Test parsing a complex pattern with multiple features."""
    result = parse_pattern("bd*2 [sd cp] ~ {hh, oh}")
    assert isinstance(result.unwrap, PatSeq)

    children = list(result.unwrap.children)
    assert len(children) == 4

    # First: bd*2 (should be repetition)
    assert isinstance(children[0].unwrap, PatRepetition)

    # Second: [sd cp] (should be grouped)
    assert isinstance(children[1].unwrap, PatGroup)

    # Third: ~ (should be silence)
    assert isinstance(children[2].unwrap, PatSilence)

    # Fourth: {hh, oh} (should be polymetric)
    assert isinstance(children[3].unwrap, PatPolymetric)


def test_parse_nested_groups():
    """Test parsing nested grouping structures."""
    result = parse_pattern("[[bd sd] cp] hh")
    assert isinstance(result.unwrap, PatSeq)

    children = list(result.unwrap.children)
    assert len(children) == 2

    # First should be nested group
    nested = children[0]
    assert isinstance(nested.unwrap, PatGroup)

    # Second should be "hh"
    assert children[1].unwrap.val == "hh"


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
        children = list(result.unwrap.children)
        assert len(children) == 2
        assert children[0].unwrap.val == "bd"
        assert children[1].unwrap.val == "sd"


def test_parse_numeric_symbols():
    """Test parsing symbols with numbers."""
    result = parse_pattern("bd909 tr808")
    assert isinstance(result.unwrap, PatSeq)

    children = list(result.unwrap.children)
    assert len(children) == 2
    assert children[0].unwrap.val == "bd909"
    assert children[1].unwrap.val == "tr808"


def test_parse_complex_sample_selection():
    """Test complex sample selection patterns."""
    result = parse_pattern("bd:0*2 sd:1/2")
    assert isinstance(result.unwrap, PatSeq)

    children = list(result.unwrap.children)
    assert len(children) == 2

    # First should be bd:0*2 (repetition of sample selection)
    first = children[0]
    assert isinstance(first.unwrap, PatRepetition)
    assert first.unwrap.operator == RepetitionOp.FAST
    assert first.unwrap.count == 2
    assert isinstance(first.unwrap.pattern.unwrap, PatSelect)
    assert first.unwrap.pattern.unwrap.pattern.unwrap.val == "bd"
    assert first.unwrap.pattern.unwrap.selector == "0"

    # Second should be sd:1/2 (repetition with division of sample selection)
    second = children[1]
    assert isinstance(second.unwrap, PatRepetition)
    assert second.unwrap.operator == RepetitionOp.SLOW
    assert second.unwrap.count == 2
    assert isinstance(second.unwrap.pattern.unwrap, PatSelect)
    assert second.unwrap.pattern.unwrap.pattern.unwrap.val == "sd"
    assert second.unwrap.pattern.unwrap.selector == "1"


# Integration tests with actual patterns
class TestRealWorldPatterns:
    """Test patterns that might be used in real Tidal compositions."""

    def test_kick_snare_pattern(self):
        """Test basic kick-snare pattern."""
        result = parse_pattern("bd sd bd sd")
        assert isinstance(result.unwrap, PatSeq)
        children = list(result.unwrap.children)
        assert len(children) == 4
        assert [child.unwrap.val for child in children] == ["bd", "sd", "bd", "sd"]

    def test_hihat_subdivision(self):
        """Test hihat subdivision pattern."""
        result = parse_pattern("bd [hh hh] sd [hh hh]")
        assert isinstance(result.unwrap, PatSeq)
        children = list(result.unwrap.children)
        assert len(children) == 4

        # Check that the hihat groups are properly parsed
        assert isinstance(children[1].unwrap, PatGroup)
        assert isinstance(children[3].unwrap, PatGroup)

    def test_polyrhythmic_pattern(self):
        """Test polyrhythmic pattern."""
        result = parse_pattern("{bd sd, hh*8}")
        assert isinstance(result.unwrap, PatPolymetric)

        patterns = list(result.unwrap.patterns)
        assert len(patterns) == 2

        # First should be "bd sd" sequence
        assert isinstance(patterns[0].unwrap, PatSeq)

        # Second should be "hh*8" (repetition)
        assert isinstance(patterns[1].unwrap, PatRepetition)
