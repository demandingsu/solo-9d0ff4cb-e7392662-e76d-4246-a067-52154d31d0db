"""Hierarchy validation: cycles, multiple parents, missing levels, shape."""

from __future__ import annotations

import pytest

from app.csv_io import MappingRow
from app.errors import MappingValidationError
from app.hierarchy import build_hierarchy


def row(node_id, parent_id, level=None, weight=1.0, name=None):
    return MappingRow(node_id, parent_id, level, name or node_id, weight)


def test_valid_tree():
    h = build_hierarchy(
        [
            row("P1", None, "plant"),
            row("S1", "P1", "shop"),
            row("L1", "S1", "line"),
        ]
    )
    assert h.roots == ["P1"]
    assert h.by_level == {"plant": ["P1"], "shop": ["S1"], "line": ["L1"]}
    assert h.ancestors("L1") == ["S1", "P1"]


def test_cycle_rejected():
    with pytest.raises(MappingValidationError, match="成环"):
        build_hierarchy(
            [
                row("P1", None, "plant"),
                row("S1", "S2", "shop"),
                row("S2", "S1", "shop"),
                row("L1", "S1", "line"),
            ]
        )


def test_self_parent_cycle_rejected():
    with pytest.raises(MappingValidationError, match="成环"):
        build_hierarchy(
            [
                row("P1", None, "plant"),
                row("S1", "S1", "shop"),
                row("L1", "S1", "line"),
            ]
        )


def test_multiple_parents_rejected():
    # Same node with two distinct parents, split across rows.
    with pytest.raises(MappingValidationError, match="多个父节点"):
        build_hierarchy(
            [
                row("P1", None, "plant"),
                row("S1", "P1", "shop"),
                row("S2", "P1", "shop"),
                row("L1", "S1", "line"),
                row("L1", "S2", "line"),
            ]
        )


def test_duplicate_rows_rejected():
    with pytest.raises(MappingValidationError, match="重复"):
        build_hierarchy(
            [
                row("P1", None, "plant"),
                row("S1", "P1", "shop"),
                row("L1", "S1", "line"),
                row("L1", "S1", "line"),
            ]
        )


def test_missing_parent_rejected():
    with pytest.raises(MappingValidationError, match="父节点"):
        build_hierarchy(
            [
                row("S1", "P1", "shop"),
                row("L1", "S1", "line"),
            ]
        )


def test_missing_shop_level_rejected():
    # A line hanging directly under the plant skips the shop level.
    with pytest.raises(MappingValidationError, match="层级"):
        build_hierarchy(
            [
                row("P1", None, "plant"),
                row("L1", "P1", "line"),
            ]
        )


def test_missing_line_level_rejected():
    with pytest.raises(MappingValidationError, match="缺层|没有产线"):
        build_hierarchy(
            [
                row("P1", None, "plant"),
                row("S1", "P1", "shop"),
            ]
        )


def test_multiple_roots_rejected():
    with pytest.raises(MappingValidationError, match="根节点"):
        build_hierarchy(
            [
                row("P1", None, "plant"),
                row("P2", None, "plant"),
                row("S1", "P1", "shop"),
                row("S2", "P2", "shop"),
                row("L1", "S1", "line"),
                row("L2", "S2", "line"),
            ]
        )


def test_explicit_level_conflict_rejected():
    with pytest.raises(MappingValidationError, match="层级与树结构不一致"):
        build_hierarchy(
            [
                row("P1", None, "plant"),
                row("S1", "P1", "line"),  # depth says shop
                row("L1", "S1", "line"),
            ]
        )


def test_all_zero_weights_rejected():
    with pytest.raises(MappingValidationError, match="权重"):
        build_hierarchy(
            [
                row("P1", None, "plant", weight=0),
                row("S1", "P1", "shop", weight=0),
                row("L1", "S1", "line", weight=0),
            ]
        )


def test_by_level_follows_tree_order():
    h = build_hierarchy(
        [
            row("P1", None, "plant"),
            row("S2", "P1", "shop"),
            row("S1", "P1", "shop"),
            row("L3", "S2", "line"),
            row("L1", "S1", "line"),
            row("L2", "S1", "line"),
        ]
    )
    assert h.by_level["shop"] == ["S1", "S2"]
    assert h.by_level["line"] == ["L1", "L2", "L3"]
    assert h.nodes["S1"].children == ["L1", "L2"]


def test_multiple_parents_via_csv_rows_rejected():
    with pytest.raises(MappingValidationError, match="多个父节点"):
        build_hierarchy(
            [
                row("P1", None, "plant"),
                row("S1", "P1", "shop"),
                row("S2", "P1", "shop"),
                row("L1", "S1", "line"),
                row("L1", "S2", "line"),
            ]
        )


def test_four_level_tree_rejected():
    with pytest.raises(MappingValidationError, match="不能有子节点|层级"):
        build_hierarchy(
            [
                row("P1", None, "plant"),
                row("S1", "P1", "shop"),
                row("L1", "S1", "line"),
                row("X1", "L1", None),
            ]
        )


def test_two_plants_via_single_root_is_fine_but_two_roots_not():
    # A factory with two shops is fine; two independent roots are not.
    h = build_hierarchy(
        [
            row("P1", None, "plant"),
            row("S1", "P1", "shop"),
            row("S2", "P1", "shop"),
            row("L1", "S1", "line"),
            row("L2", "S2", "line"),
        ]
    )
    assert h.roots == ["P1"]
