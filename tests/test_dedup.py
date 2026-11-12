from beauty_scorer.detector import _dedup


def test_identical_boxes_merge():
    detections = [(10, 10, 100, 100, 0.9), (10, 10, 100, 100, 0.8)]
    assert len(_dedup(detections)) == 1


def test_nested_box_merges():
    detections = [(0, 0, 200, 200, 0.9), (50, 50, 50, 50, 0.8)]
    assert len(_dedup(detections)) == 1


def test_distinct_faces_are_kept():
    detections = [(0, 0, 80, 80, 0.9), (200, 0, 80, 80, 0.8)]
    assert len(_dedup(detections)) == 2


def test_transitive_chain_merges():
    detections = [
        (0, 0, 100, 100, 0.9),
        (50, 0, 100, 100, 0.8),
        (100, 0, 100, 100, 0.7),
    ]
    assert len(_dedup(detections)) == 1


def test_union_box_covers_cluster():
    detections = [(0, 0, 100, 100, 0.9), (10, 0, 100, 100, 0.8)]
    (x, y, w, h, _) = _dedup(detections)[0]
    assert (x, y, w, h) == (0, 0, 110, 100)


def test_invalid_boxes_are_dropped():
    detections = [(0, 0, 0, 100, 0.9), (0, 0, 100, 100, 0.8)]
    assert len(_dedup(detections)) == 1
