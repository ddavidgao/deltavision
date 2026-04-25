"""
Unit tests for vision.diff.merge_bboxes_for_min_cost.

Token cost model:
    cost(w, h) = 85 + ceil(w/512) * ceil(h/512) * 170

These are the only costs that matter for the merge invariants — the merge
should never make total cost go up, and should always merge a pair if the
union is strictly cheaper than the sum.
"""
from __future__ import annotations

from vision.diff import (
    _bbox_token_cost,
    _bbox_union,
    merge_bboxes_for_min_cost,
)


def total_cost(boxes):
    return sum(_bbox_token_cost(w, h) for _, _, w, h in boxes)


# -------------------------------------------------------------------- cost model

def test_cost_smallest_region():
    """A 1×1 region pays base + one tile."""
    assert _bbox_token_cost(1, 1) == 85 + 170


def test_cost_one_tile():
    assert _bbox_token_cost(512, 512) == 85 + 170


def test_cost_just_over_one_tile_horizontal():
    """513×512 tips into a 2-tile-wide region."""
    assert _bbox_token_cost(513, 512) == 85 + 2 * 170


def test_cost_full_1280x800():
    """A full 1280×800 frame is ceil(1280/512)=3 × ceil(800/512)=2 = 6 tiles."""
    assert _bbox_token_cost(1280, 800) == 85 + 6 * 170


# -------------------------------------------------------------------- merge basics

def test_merge_empty_returns_empty():
    assert merge_bboxes_for_min_cost([]) == []


def test_merge_singleton_unchanged():
    box = (10, 10, 100, 100)
    assert merge_bboxes_for_min_cost([box]) == [box]


def test_merge_overlapping_pair_merges():
    """Two overlapping bboxes — union is small enough that it's cheaper than sum."""
    a = (0, 0, 100, 100)
    b = (50, 50, 100, 100)
    out = merge_bboxes_for_min_cost([a, b])
    # Each costs 85 + 170 = 255 (1 tile each). Sum = 510.
    # Union is (0, 0, 150, 150) — still 1 tile = 255. Strictly cheaper, must merge.
    assert len(out) == 1
    assert out[0] == _bbox_union(a, b)


def test_merge_distant_pair_does_not_merge():
    """Two corners of the screen — union spans 1280×800 (full frame ≈ 1105),
    each tiny region costs only 255. Sum (510) < full-frame cost. DON'T merge."""
    a = (0, 0, 100, 100)
    b = (1180, 700, 100, 100)
    out = merge_bboxes_for_min_cost([a, b])
    assert len(out) == 2
    assert total_cost(out) < _bbox_token_cost(1280, 800)


# -------------------------------------------------------------------- invariants

def test_merge_never_increases_total_cost():
    """The greedy must never produce a result MORE expensive than the input."""
    inputs = [
        [(0, 0, 100, 100), (50, 50, 100, 100)],
        [(0, 0, 600, 100), (0, 200, 600, 100)],  # two horizontal bands close
        [(0, 0, 100, 100)] * 5,                   # 5 stacked identical
        [(i*100, 0, 50, 50) for i in range(6)],   # 6 row of dots
        [(0, 0, 1000, 100), (1100, 0, 100, 100)], # one big + one corner
    ]
    for boxes in inputs:
        out = merge_bboxes_for_min_cost(boxes)
        assert total_cost(out) <= total_cost(boxes), (
            f"merge increased cost for {boxes}: {total_cost(boxes)} -> "
            f"{total_cost(out)} via {out}"
        )


def test_merge_realworld_step_38_fragmented_diff():
    """
    The actual step-38 contour set from run_20_dv_v105 — a one-row sheet edit
    plus 5 scattered tiny UI shifts. Naive: 6 crops totaling 1700 tokens (>
    1365 → cap fallback to full frame). After merge: should be ≤ 2 regions
    and < 1365 (so the proxy keeps it as a delta instead of full-framing).
    """
    boxes = [
        (0, 177, 778, 168),     # 12.76% — the real change
        (263, 0, 122, 49),
        (122, 103, 118, 50),
        (1152, 750, 107, 50),
        (728, 72, 104, 44),
        (629, 53, 66, 66),
    ]
    out = merge_bboxes_for_min_cost(boxes)
    naive_total = total_cost(boxes)
    merged_total = total_cost(out)
    assert merged_total < naive_total, (
        f"merge didn't save tokens: {naive_total} -> {merged_total}"
    )
    # And critically: merged total should be under the full-frame cost so the
    # proxy doesn't fall back to a full frame.
    full_frame_cost = _bbox_token_cost(1280, 800)  # 1105
    assert merged_total < full_frame_cost, (
        f"merged total {merged_total} should beat full frame {full_frame_cost}"
    )


def test_merge_idempotent():
    """Running merge twice should give the same result."""
    boxes = [(0, 0, 100, 100), (50, 50, 100, 100), (200, 200, 50, 50)]
    once = merge_bboxes_for_min_cost(boxes)
    twice = merge_bboxes_for_min_cost(once)
    assert once == twice


def test_merge_preserves_full_coverage():
    """Every input pixel must remain covered by at least one output bbox."""
    inputs = [
        [(0, 0, 100, 100), (50, 50, 100, 100)],
        [(0, 0, 600, 100), (0, 200, 600, 100)],
        [(i*100, 0, 50, 50) for i in range(4)],
    ]
    for boxes in inputs:
        out = merge_bboxes_for_min_cost(boxes)
        for (x, y, w, h) in boxes:
            corners = [(x, y), (x + w - 1, y), (x, y + h - 1), (x + w - 1, y + h - 1)]
            for cx, cy in corners:
                covered = any(
                    ox <= cx < ox + ow and oy <= cy < oy + oh
                    for (ox, oy, ow, oh) in out
                )
                assert covered, f"corner ({cx},{cy}) of input box {(x,y,w,h)} not covered by {out}"
