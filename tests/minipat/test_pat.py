from minipat.pat import (
    Pat,
    PatPar,
    PatPure,
    PatSeq,
)


def test_pat_silence():
    """Test silence pattern creation."""
    empty: Pat[str] = Pat.silence()
    assert empty.unwrap is not None


def test_pat_pure():
    """Test pure pattern creation."""
    pure = Pat.pure("test")

    assert isinstance(pure.unwrap, PatPure)
    assert pure.unwrap.value == "test"


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


def test_pat_map():
    """Test pattern mapping."""
    pat = Pat.pure(5)
    mapped = pat.map(lambda x: x * 2)

    assert isinstance(mapped.unwrap, PatPure)
    assert mapped.unwrap.value == 10
