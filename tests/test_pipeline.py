import numpy as np
import pytest

from carcounter.counter import CountingSetup
from carcounter.draw import ignored_mask
from carcounter.geometry import Line
from carcounter.pipeline import anchor_points, class_mapping

SETUP = CountingSetup(Line((0, 100), (400, 100)), Line((0, 300), (400, 300)), (200, 400))


def test_class_mapping_merges_bus_into_truck():
    names = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    m = class_mapping(names, {"car": "car", "truck": "truck", "bus": "truck"})
    assert m == {2: "car", 5: "truck", 7: "truck"}


def test_class_mapping_finetuned_model():
    assert class_mapping({0: "car", 1: "truck"}, {"car": "car", "truck": "truck", "bus": "truck"}) == {0: "car", 1: "truck"}


def test_class_mapping_no_match():
    with pytest.raises(ValueError):
        class_mapping({0: "vehicle"}, {"car": "car"})


def test_anchor_points():
    xyxy = np.array([[10, 20, 30, 60]], dtype=np.float32)
    assert anchor_points(xyxy, "bottom_center").tolist() == [[20, 60]]
    assert anchor_points(xyxy, "center").tolist() == [[20, 40]]


def test_ignored_mask_is_outside_the_zone():
    mask = ignored_mask(SETUP, (400, 400), margin=10)
    assert mask[50, 200] and mask[89, 200] and mask[350, 200]  # outside both lines, beyond the margin
    assert not mask[95, 200] and not mask[200, 200] and not mask[305, 200]  # zone and margin bands


def test_resolve_weights(tmp_path):
    from carcounter.paths import WEIGHTS, resolve_weights

    assert resolve_weights("yolo11s.pt") == str(WEIGHTS / "yolo11s.pt")  # official name -> weights/
    existing = tmp_path / "custom.pt"
    existing.write_bytes(b"")
    assert resolve_weights(existing) == str(existing.resolve())
    with pytest.raises(FileNotFoundError):
        resolve_weights("weights/does-not-exist.pt")
