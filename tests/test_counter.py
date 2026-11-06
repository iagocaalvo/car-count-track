import pytest

from carcounter.counter import IN, OUT, CountingSetup, LineCounter
from carcounter.geometry import Line

# Road going downwards (y grows). Zone between y=100 and y=300.
# IN point below y=300 -> line_2 (y=300) is the IN line, line_1 (y=100) the OUT line.
SETUP = CountingSetup(
    line_1=Line((0, 100), (400, 100)),
    line_2=Line((0, 300), (400, 300)),
    in_point=(200, 400),
)


def run(counter, tid, ys, x=200, start=0, cat="car"):
    events = []
    for i, y in enumerate(ys):
        events += counter.update(start + i, (start + i) / 25, [(tid, (x, y), cat)])
    return events


def test_crossing_segment_extent():
    line = Line((0, 0), (10, 0))
    assert line.crossed_by((5, -1), (5, 1))
    assert not line.crossed_by((15, -1), (15, 1))  # beyond the segment
    assert not line.crossed_by((5, 1), (5, 2))  # same side


def test_in_and_out_lines_from_in_point():
    assert SETUP.in_line is SETUP.line_2
    assert SETUP.out_line is SETUP.line_1
    flipped = CountingSetup(SETUP.line_1, SETUP.line_2, (200, 0))
    assert flipped.in_line is flipped.line_1


def test_leaving_through_in_line_counts_in():
    c = LineCounter(SETUP)
    ev = run(c, 1, [200, 250, 290, 310])
    assert [e.direction for e in ev] == [IN]


def test_leaving_through_out_line_counts_out():
    c = LineCounter(SETUP)
    ev = run(c, 2, [200, 150, 110, 90])
    assert [e.direction for e in ev] == [OUT]


def test_entering_is_not_counted():
    c = LineCounter(SETUP)
    run(c, 3, [90, 110, 200])  # enters through the OUT line, stays inside
    run(c, 4, [310, 290, 200])  # enters through the IN line, stays inside
    assert c.total == 0


def test_full_pass_counted_once_at_exit():
    c = LineCounter(SETUP)
    ev = run(c, 5, [90, 110, 200, 290, 310, 350])
    assert [e.direction for e in ev] == [IN]
    assert c.counts == {IN: 1, OUT: 0}


def test_counted_once_with_jitter():
    c = LineCounter(SETUP)
    run(c, 6, [250, 299, 301, 299, 301, 305])
    assert c.counts == {IN: 1, OUT: 0}


def test_point_on_line_does_not_break_crossing():
    c = LineCounter(SETUP)
    run(c, 7, [250, 300, 320])
    assert c.counts[IN] == 1


def test_crossing_outside_segment_ignored():
    c = LineCounter(SETUP)
    run(c, 8, [200, 290, 320], x=600)  # x beyond the segments
    assert c.total == 0


def test_zone_and_validation():
    assert SETUP.in_zone((200, 200))
    assert not SETUP.in_zone((200, 50))
    assert not SETUP.in_zone((200, 350))
    assert SETUP.in_zone((200, 70), margin=40)
    with pytest.raises(ValueError):  # IN point inside the zone
        CountingSetup(SETUP.line_1, SETUP.line_2, (200, 200))
    with pytest.raises(ValueError):  # IN point on a line
        CountingSetup(SETUP.line_1, SETUP.line_2, (200, 300))


def test_category_majority_vote():
    c = LineCounter(SETUP, categories=["car", "truck"])
    ys = [200, 230, 250, 270, 290, 310]
    cats = ["car", "truck", "truck", "truck", "car", "truck"]
    for i, (y, cat) in enumerate(zip(ys, cats)):
        c.update(i, i / 25, [(9, (200, y), cat)])
    assert c.counts_by_category == {"car": {IN: 0, OUT: 0}, "truck": {IN: 1, OUT: 0}}
    assert c.events[0].category == "truck"


def test_category_frozen_after_count():
    c = LineCounter(SETUP, categories=["car", "truck"])
    run(c, 10, [250, 310])  # counted as car
    for i in range(10):
        c.update(10 + i, 1.0, [(10, (200, 320), "truck")])
    assert c.category(10) == "car"
    assert c.counts_by_category["car"][IN] == 1


def test_scaled_setup_preserves_logic():
    c = LineCounter(SETUP.scaled(0.5, 0.5))
    run(c, 11, [100, 140, 160], x=100)
    assert c.counts[IN] == 1


def test_line_extended_to_frame_edges():
    ext = Line((100, 100), (200, 100)).extended_to(400, 300)
    assert {ext.a, ext.b} == {(0.0, 100.0), (400.0, 100.0)}
    diag = Line((100, 100), (110, 110)).extended_to(400, 300)
    assert {diag.a, diag.b} == {(0.0, 0.0), (300.0, 300.0)}


def test_extended_setup_counts_beyond_clicked_segment():
    short = CountingSetup(Line((150, 100), (250, 100)), Line((150, 300), (250, 300)), (200, 400))
    c = LineCounter(short.extended_to(400, 400))
    run(c, 12, [200, 290, 320], x=380)  # far beyond the clicked segments, inside the frame
    assert c.counts[IN] == 1


def test_extended_lines_crossing_in_frame_rejected():
    # Converging lines: valid as clicked, but once extended they meet at x=400.
    converging = CountingSetup(Line((0, 100), (100, 100)), Line((0, 300), (100, 250)), (50, 400))
    converging.extended_to(300, 400)  # they don't meet inside a narrow frame
    with pytest.raises(ValueError):
        converging.extended_to(1000, 1000)
