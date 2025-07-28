from fractions import Fraction

from centipede.minipat.arc import Arc
from centipede.minipat.pat import (
    Pat,
    PatClip,
    PatMask,
    PatPar,
    PatPure,
    PatScale,
    PatSeq,
    PatShift,
)


def test_pat_silence():
    """Test silence pattern creation."""
    empty: Pat[str] = Pat.silence()
    assert empty.unwrap is not None


def test_pat_pure():
    """Test pure pattern creation."""
    pure = Pat.pure("test")

    assert isinstance(pure.unwrap, PatPure)
    assert pure.unwrap.val == "test"


def test_pat_seq():
    """Test sequence pattern creation."""
    pats = [Pat.pure("a"), Pat.pure("b"), Pat.pure("c")]
    seq = Pat.seq(pats)

    assert isinstance(seq.unwrap, PatSeq)


def test_pat_par():
    """Test parallel pattern creation."""
    pats = [Pat.pure("a"), Pat.pure("b"), Pat.pure("c")]
    par = Pat.par(pats)

    assert isinstance(par.unwrap, PatPar)


def test_pat_mask():
    """Test pattern masking."""
    pat = Pat.pure("test")
    arc = Arc(Fraction(0), Fraction(1))
    masked = pat.mask(arc)

    assert isinstance(masked.unwrap, PatMask)
    assert masked.unwrap.arc == arc


def test_pat_shift():
    """Test pattern shifting."""
    pat = Pat.pure("test")
    shifted = pat.shift(Fraction(1))

    assert isinstance(shifted.unwrap, PatShift)
    assert shifted.unwrap.delta == Fraction(1)


def test_pat_scale():
    """Test pattern scaling."""
    pat = Pat.pure("test")
    scaled = pat.scale(Fraction(2))

    assert isinstance(scaled.unwrap, PatScale)
    assert scaled.unwrap.factor == Fraction(2)


def test_pat_clip():
    """Test pattern clipping."""
    pat = Pat.pure("test")
    clipped = pat.clip(Fraction(1, 2))

    assert isinstance(clipped.unwrap, PatClip)
    assert clipped.unwrap.factor == Fraction(1, 2)


def test_pat_map():
    """Test pattern mapping."""
    pat = Pat.pure(5)
    mapped = pat.map(lambda x: x * 2)

    assert isinstance(mapped.unwrap, PatPure)
    assert mapped.unwrap.val == 10
