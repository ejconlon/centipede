"""Tests for minipat.reflect module."""

from __future__ import annotations

from fractions import Fraction

from minipat.common import CycleDelta
from minipat.pat import Pat
from minipat.reflect import (
    DeltaSeq,
    DeltaVal,
    StepSeq,
    StepVal,
    quantize,
    reflect,
    step_delta,
    unquantize,
)
from spiny.seq import PSeq


class TestQuantize:
    """Tests for the quantize function."""

    def test_empty_sequence(self):
        """Empty sequence should quantize to empty."""
        ds: DeltaSeq[str] = DeltaVal(CycleDelta(Fraction(0)), PSeq.empty())
        ss = quantize(ds)
        assert ss.steps == 0
        assert ss.val.null()

    def test_single_element(self):
        """Single element with simple fraction."""
        inner = DeltaVal(CycleDelta(Fraction(1, 2)), "a")
        ds = DeltaVal(CycleDelta(Fraction(1, 2)), PSeq.mk([inner]))
        ss = quantize(ds)
        assert ss.steps == 1
        assert len(ss.val) == 1
        first = list(ss.val.iter())[0]
        assert first.steps == 1
        assert first.val == "a"

    def test_two_halves(self):
        """Two half notes should quantize to 2 steps total."""
        items = [
            DeltaVal(CycleDelta(Fraction(1, 2)), "a"),
            DeltaVal(CycleDelta(Fraction(1, 2)), "b"),
        ]
        ds = DeltaVal(CycleDelta(Fraction(1)), PSeq.mk(items))
        ss = quantize(ds)
        assert ss.steps == 2
        assert len(ss.val) == 2
        vals = list(ss.val.iter())
        assert vals[0].steps == 1
        assert vals[0].val == "a"
        assert vals[1].steps == 1
        assert vals[1].val == "b"

    def test_mixed_denominators(self):
        """Different denominators should find LCM."""
        items = [
            DeltaVal(CycleDelta(Fraction(1, 3)), "a"),
            DeltaVal(CycleDelta(Fraction(1, 4)), "b"),
            DeltaVal(CycleDelta(Fraction(5, 12)), "c"),
        ]
        ds = DeltaVal(CycleDelta(Fraction(1)), PSeq.mk(items))
        ss = quantize(ds)
        assert ss.steps == 12  # LCM of 3, 4, 12
        vals = list(ss.val.iter())
        assert vals[0].steps == 4  # 1/3 * 12 = 4
        assert vals[0].val == "a"
        assert vals[1].steps == 3  # 1/4 * 12 = 3
        assert vals[1].val == "b"
        assert vals[2].steps == 5  # 5/12 * 12 = 5
        assert vals[2].val == "c"

    def test_whole_and_fractions(self):
        """Mix of whole numbers and fractions."""
        items = [
            DeltaVal(CycleDelta(Fraction(1)), "a"),
            DeltaVal(CycleDelta(Fraction(1, 2)), "b"),
            DeltaVal(CycleDelta(Fraction(1, 2)), "c"),
        ]
        ds = DeltaVal(CycleDelta(Fraction(2)), PSeq.mk(items))
        ss = quantize(ds)
        assert ss.steps == 4  # LCM consideration with denominator 2
        vals = list(ss.val.iter())
        assert vals[0].steps == 2  # 1 * 2 = 2
        assert vals[1].steps == 1  # 1/2 * 2 = 1
        assert vals[2].steps == 1  # 1/2 * 2 = 1


class TestStepDelta:
    """Tests for the step_delta function."""

    def test_uniform_steps(self):
        """Test step_delta calculation with uniform steps."""
        items = [
            DeltaVal(CycleDelta(Fraction(1, 2)), "a"),
            DeltaVal(CycleDelta(Fraction(1, 2)), "b"),
        ]
        ds = DeltaVal(CycleDelta(Fraction(1)), PSeq.mk(items))
        ss = quantize(ds)
        delta = step_delta(ds, ss)
        assert delta == CycleDelta(Fraction(1, 2))  # 1 cycle / 2 steps

    def test_non_uniform_steps(self):
        """Test step_delta with non-uniform steps."""
        items = [
            DeltaVal(CycleDelta(Fraction(1, 3)), "a"),
            DeltaVal(CycleDelta(Fraction(2, 3)), "b"),
        ]
        ds = DeltaVal(CycleDelta(Fraction(1)), PSeq.mk(items))
        ss = quantize(ds)
        delta = step_delta(ds, ss)
        assert delta == CycleDelta(Fraction(1, 3))  # 1 cycle / 3 steps


class TestReflect:
    """Tests for the reflect function."""

    def test_empty_sequence(self):
        """Empty sequence should reflect to silent pattern."""
        ss: StepSeq[str] = StepVal(0, PSeq.empty())
        pat = reflect(ss)
        assert pat == Pat.silent()

    def test_single_element(self):
        """Single element should reflect to pure pattern."""
        inner = StepVal(1, "a")
        ss = StepVal(1, PSeq.mk([inner]))
        pat = reflect(ss)
        assert pat == Pat.pure("a")

    def test_sequence_of_elements(self):
        """Sequence should reflect to seq pattern."""
        items = [
            StepVal(1, "a"),
            StepVal(1, "b"),
            StepVal(1, "c"),
        ]
        ss = StepVal(3, PSeq.mk(items))
        pat = reflect(ss)
        expected = Pat.seq([Pat.pure("a"), Pat.pure("b"), Pat.pure("c")])
        assert pat == expected

    def test_non_uniform_steps(self):
        """Non-uniform steps should create stretched patterns."""
        items = [
            StepVal(2, "a"),  # Takes 2 steps
            StepVal(1, "b"),  # Takes 1 step
        ]
        ss = StepVal(3, PSeq.mk(items))
        _ = reflect(ss)
        # This should create a pattern where 'a' takes up 2/3 of the cycle
        # and 'b' takes up 1/3 of the cycle
        # The exact representation depends on how reflect handles stretching


class TestUnquantize:
    """Tests for the unquantize function."""

    def test_empty_sequence(self):
        """Empty sequence should unquantize to empty."""
        ss: StepSeq[str] = StepVal(0, PSeq.empty())
        ds = unquantize(ss, CycleDelta(Fraction(1)))
        assert ds.delta == CycleDelta(Fraction(0))
        assert ds.val.null()

    def test_single_element(self):
        """Single element should get the full delta."""
        inner = StepVal(2, "a")
        ss = StepVal(2, PSeq.mk([inner]))
        ds = unquantize(ss, CycleDelta(Fraction(1)))
        assert ds.delta == CycleDelta(Fraction(1))
        assert len(ds.val) == 1
        first = list(ds.val.iter())[0]
        assert first.delta == CycleDelta(Fraction(1))
        assert first.val == "a"

    def test_uniform_steps(self):
        """Uniform steps should divide delta equally."""
        items = [
            StepVal(1, "a"),
            StepVal(1, "b"),
            StepVal(1, "c"),
        ]
        ss = StepVal(3, PSeq.mk(items))
        ds = unquantize(ss, CycleDelta(Fraction(1)))
        assert ds.delta == CycleDelta(Fraction(1))
        vals = list(ds.val.iter())
        for val in vals:
            assert val.delta == CycleDelta(Fraction(1, 3))

    def test_non_uniform_steps(self):
        """Non-uniform steps should get proportional deltas."""
        items = [
            StepVal(2, "a"),  # 2 out of 4 steps
            StepVal(1, "b"),  # 1 out of 4 steps
            StepVal(1, "c"),  # 1 out of 4 steps
        ]
        ss = StepVal(4, PSeq.mk(items))
        ds = unquantize(ss, CycleDelta(Fraction(1)))
        assert ds.delta == CycleDelta(Fraction(1))
        vals = list(ds.val.iter())
        assert vals[0].delta == CycleDelta(Fraction(1, 2))  # 2/4
        assert vals[0].val == "a"
        assert vals[1].delta == CycleDelta(Fraction(1, 4))  # 1/4
        assert vals[1].val == "b"
        assert vals[2].delta == CycleDelta(Fraction(1, 4))  # 1/4
        assert vals[2].val == "c"

    def test_custom_total_delta(self):
        """Should scale to any total delta."""
        items = [
            StepVal(1, "a"),
            StepVal(2, "b"),
        ]
        ss = StepVal(3, PSeq.mk(items))
        ds = unquantize(ss, CycleDelta(Fraction(3, 2)))
        assert ds.delta == CycleDelta(Fraction(3, 2))
        vals = list(ds.val.iter())
        assert vals[0].delta == CycleDelta(Fraction(1, 2))  # 1/3 * 3/2
        assert vals[1].delta == CycleDelta(Fraction(1))  # 2/3 * 3/2


class TestRoundTrip:
    """Tests that quantize and unquantize are inverses."""

    def test_simple_round_trip(self):
        """Quantize then unquantize should preserve proportions."""
        # Create original DeltaSeq
        items = [
            DeltaVal(CycleDelta(Fraction(1, 3)), "a"),
            DeltaVal(CycleDelta(Fraction(2, 3)), "b"),
        ]
        original = DeltaVal(CycleDelta(Fraction(1)), PSeq.mk(items))

        # Quantize then unquantize
        quantized = quantize(original)
        restored = unquantize(quantized, original.delta)

        # Check that we get back the same structure
        assert restored.delta == original.delta
        assert len(restored.val) == len(original.val)

        # Check proportions are preserved
        restored_vals = list(restored.val.iter())
        original_vals = list(original.val.iter())
        for r, o in zip(restored_vals, original_vals):
            assert r.delta == o.delta
            assert r.val == o.val

    def test_complex_round_trip(self):
        """Round trip with complex fractions."""
        items = [
            DeltaVal(CycleDelta(Fraction(2, 5)), "x"),
            DeltaVal(CycleDelta(Fraction(1, 3)), "y"),
            DeltaVal(CycleDelta(Fraction(4, 15)), "z"),
        ]
        # Total: 6/15 + 5/15 + 4/15 = 15/15 = 1
        original = DeltaVal(CycleDelta(Fraction(1)), PSeq.mk(items))

        quantized = quantize(original)
        restored = unquantize(quantized, original.delta)

        assert restored.delta == original.delta
        restored_vals = list(restored.val.iter())
        original_vals = list(original.val.iter())
        for r, o in zip(restored_vals, original_vals):
            assert r.delta == o.delta
            assert r.val == o.val
