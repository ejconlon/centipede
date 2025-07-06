from centipede.main import Arc


def test_arc_split():
    assert list(Arc.empty().split_cycles()) == []
    assert list(Arc.cycle(0).split_cycles()) == [(0, Arc.cycle(0))]
