from flydeck.validation_protocol import build_chronological_protocol, build_walk_forward_folds


def test_chronological_protocol_is_purged_and_disjoint() -> None:
    protocol = build_chronological_protocol(1000, context=32, purge=1)
    protocol.assert_disjoint()
    assert protocol.train.start == 31
    assert protocol.train.end < protocol.validation.start
    assert protocol.validation.end < protocol.test.start
    assert protocol.test.end == 999


def test_walk_forward_folds_are_strictly_causal() -> None:
    folds = build_walk_forward_folds(
        2000,
        context=32,
        train_size=500,
        validation_size=100,
        test_size=100,
        purge=1,
    )
    assert len(folds) > 1
    for fold in folds:
        assert fold.train.end < fold.validation.start
        assert fold.validation.end < fold.test.start
        assert fold.test.end <= 1999
