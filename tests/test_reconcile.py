"""Reconciliation maths: constraints, weight sensitivity, failures."""

from __future__ import annotations

import pytest

from app.csv_io import MappingRow
from app.errors import SnapshotValidationError
from app.hierarchy import build_hierarchy
from app.reconcile import reconcile, verify_consistency


def build(weights=None):
    weights = weights or {}
    rows = [
        MappingRow("P1", None, "plant", "P1", weights.get("P1", 1.0)),
        MappingRow("S1", "P1", "shop", "S1", weights.get("S1", 1.0)),
        MappingRow("S2", "P1", "shop", "S2", weights.get("S2", 1.0)),
        MappingRow("L1", "S1", "line", "L1", weights.get("L1", 1.0)),
        MappingRow("L2", "S1", "line", "L2", weights.get("L2", 1.0)),
        MappingRow("L3", "S2", "line", "L3", weights.get("L3", 1.0)),
    ]
    return build_hierarchy(rows)


BASE = [
    {"node_id": "L1", "period": "2026-01", "value": 10.0},
    {"node_id": "L2", "period": "2026-01", "value": 20.0},
    {"node_id": "L3", "period": "2026-01", "value": 30.0},
]


def publish(target, value, kind="node", period="2026-01"):
    return [{"target": target, "kind": kind, "period": period, "value": value}]


def test_consistent_input_is_unchanged():
    result = reconcile(build(), BASE, [])
    period = result.periods[0]
    assert period.adjusted is False
    assert period.corrected == pytest.approx({"L1": 10, "L2": 20, "L3": 30, "S1": 30, "S2": 30, "P1": 60})


def test_hierarchy_sums_hold_after_correction():
    result = reconcile(build(), BASE, publish("S1", 36))
    assert verify_consistency(result) == []
    period = result.periods[0]
    assert period.corrected["S1"] == pytest.approx(36)
    assert period.corrected["L1"] + period.corrected["L2"] == pytest.approx(36)
    # S2/P1 absorb the knock-on movement; nothing is left inconsistent.
    assert period.corrected["P1"] == pytest.approx(
        period.corrected["S1"] + period.corrected["S2"]
    )


def test_equal_weights_split_adjustment_evenly():
    result = reconcile(build(), BASE, publish("S1", 36))
    p = result.periods[0]
    assert p.corrected["L1"] == pytest.approx(13)
    assert p.corrected["L2"] == pytest.approx(23)


def test_weights_shift_the_adjustment():
    # Lines inside S1 weigh 1 and 3.  The correction is spread in inverse
    # proportion to weight, so L1 (cheap to move) takes most of it.
    result = reconcile(build({"L1": 1.0, "L2": 3.0}), BASE, publish("S1", 36))
    p = result.periods[0]
    assert p.corrected["S1"] == pytest.approx(36)
    assert p.corrected["L1"] - 10 > p.corrected["L2"] - 20
    assert (p.corrected["L1"] - 10) + (p.corrected["L2"] - 20) == pytest.approx(6)


def test_line_weight_ratio_sets_the_split():
    # With only the S1 total constrained, the internal nodes contribute a
    # constant, so the line split is exactly the inverse-weight split.
    result = reconcile(build({"L1": 1.0, "L2": 3.0}), BASE, publish("S1", 36))
    p = result.periods[0]
    d1 = p.corrected["L1"] - 10
    d2 = p.corrected["L2"] - 20
    assert d1 / d2 == pytest.approx(3.0, rel=1e-6)


def test_zero_weight_line_absorbs_the_correction():
    # Weight 0 means "moving this node costs nothing", so the optimiser leaves
    # the unit-weight sibling at its base and pushes the whole correction into
    # the zero-weight line.
    result = reconcile(build({"L1": 0.0}), BASE, publish("S1", 36))
    p = result.periods[0]
    assert p.corrected["L2"] == pytest.approx(20, abs=1e-4)
    assert p.corrected["L1"] == pytest.approx(16, abs=1e-4)
    assert p.corrected["S1"] == pytest.approx(36)


def test_shop_and_plant_weights_do_not_change_line_split_alone():
    # With only an S1 constraint, the S1 weight is a constant offset in the
    # objective and cancels; only line weights decide the split.
    a = reconcile(build({"S1": 1.0}), BASE, publish("S1", 36)).periods[0]
    b = reconcile(build({"S1": 100.0}), BASE, publish("S1", 36)).periods[0]
    assert a.corrected["L1"] == pytest.approx(b.corrected["L1"])
    assert a.corrected["L2"] == pytest.approx(b.corrected["L2"])


def test_parent_weight_matters_when_both_parent_and_child_constrained():
    # Publishing both the plant total and one shop: the plant weight now trades
    # off against shop/line movement.
    published = publish("P1", 60) + publish("S1", 50)
    light = reconcile(build({"P1": 0.01}), BASE, published).periods[0]
    heavy = reconcile(build({"P1": 100.0}), BASE, published).periods[0]
    # P1 is hard-pinned to 60 in both; S1 is pinned too, so L3 absorbs the
    # remainder and the P1 weight does not alter it. Sanity: total holds.
    assert light.corrected["P1"] == pytest.approx(60)
    assert heavy.corrected["P1"] == pytest.approx(60)
    assert light.corrected["S1"] == pytest.approx(50)
    assert heavy.corrected["S1"] == pytest.approx(50)
    assert light.corrected["L3"] == pytest.approx(10)
    assert heavy.corrected["L3"] == pytest.approx(10)


def test_level_total_constraint():
    # Published total across all shops = 54, base 60.
    result = reconcile(build(), BASE, publish("shop", 54, kind="level"))
    p = result.periods[0]
    assert p.corrected["S1"] + p.corrected["S2"] == pytest.approx(54)
    # Total drops 10%; lines with the smallest weight relative to their sibling
    # absorb most of the reduction, so all three lines still sum to 54.
    assert sum(p.corrected[l] for l in ("L1", "L2", "L3")) == pytest.approx(54)
    assert p.corrected["P1"] == pytest.approx(54)
    assert verify_consistency(result) == []


def test_level_total_weights_change_the_split():
    light = reconcile(build(), BASE, publish("shop", 54, kind="level")).periods[0]
    heavy = reconcile(build({"S1": 100.0}), BASE, publish("shop", 54, kind="level")).periods[0]
    # Heavily weighting S1 makes the optimiser prefer moving S2's lines.
    assert heavy.corrected["S2"] < light.corrected["S2"]
    assert heavy.corrected["S1"] > light.corrected["S1"]


def test_plant_only_constraint_moves_all_lines():
    result = reconcile(build(), BASE, publish("P1", 120))
    p = result.periods[0]
    assert p.corrected["P1"] == pytest.approx(120)
    assert all(p.corrected[l] > b for l, b in (("L1", 10), ("L2", 20), ("L3", 30)))
    assert sum(p.corrected[l] for l in ("L1", "L2", "L3")) == pytest.approx(120)
    assert p.corrected["L1"] < 120 and p.corrected["L3"] < 120


def test_infeasible_published_values_rejected():
    published = publish("P1", 99) + publish("S1", 10) + publish("S2", 30)
    with pytest.raises(SnapshotValidationError, match="自相矛盾"):
        reconcile(build(), BASE, published)


def test_contradictory_level_and_node_rejected():
    published = publish("shop", 100, kind="level") + publish("S1", 10) + publish("S2", 30)
    with pytest.raises(SnapshotValidationError, match="自相矛盾"):
        reconcile(build(), BASE, published)


def test_duplicate_node_publish_with_same_value_ok():
    published = publish("S1", 36) + publish("S1", 36)
    result = reconcile(build(), BASE, published)
    assert result.periods[0].corrected["S1"] == pytest.approx(36)


def test_duplicate_node_publish_with_different_value_rejected():
    published = publish("S1", 36) + publish("S1", 40)
    with pytest.raises(SnapshotValidationError, match="冲突"):
        reconcile(build(), BASE, published)


def test_missing_line_period_rejected():
    with pytest.raises(SnapshotValidationError, match="缺少"):
        reconcile(build(), BASE[:2], [])


def test_unknown_line_rejected():
    rows = BASE + [{"node_id": "NOPE", "period": "2026-01", "value": 1.0}]
    with pytest.raises(SnapshotValidationError, match="不存在"):
        reconcile(build(), rows, [])


def test_duplicate_forecast_row_rejected():
    rows = BASE + [{"node_id": "L1", "period": "2026-01", "value": 11.0}]
    with pytest.raises(SnapshotValidationError, match="重复"):
        reconcile(build(), rows, [])


def test_published_outside_window_is_warned_not_fatal():
    result = reconcile(build(), BASE, publish("S1", 99, period="2027-05"))
    assert result.warnings
    assert result.periods[0].adjusted is False


def test_conflicts_recorded_with_source():
    result = reconcile(build(), BASE, publish("S1", 36))
    p = result.periods[0]
    assert len(p.conflicts) == 1
    assert p.conflicts[0]["scope"] == "S1"
    assert p.conflicts[0]["diff"] == pytest.approx(6)
    sources = {a["node_id"]: a["source"] for a in p.node_adjustments}
    assert sources["S1"] == "published_node"
    assert sources["L1"] == "weighted_balance"
    assert p.line_adjustments[0]["source_nodes"] == ["S1（shop）"]


def test_multiple_periods_solved_independently():
    base = BASE + [
        {"node_id": "L1", "period": "2026-02", "value": 100.0},
        {"node_id": "L2", "period": "2026-02", "value": 100.0},
        {"node_id": "L3", "period": "2026-02", "value": 100.0},
    ]
    published = publish("S1", 36) + publish("S1", 300, period="2026-02")
    result = reconcile(build(), base, published)
    assert [p.period for p in result.periods] == ["2026-01", "2026-02"]
    assert result.periods[0].corrected["S1"] == pytest.approx(36)
    assert result.periods[1].corrected["S1"] == pytest.approx(300)
    assert verify_consistency(result) == []


def test_inconsistent_negative_values_are_allowed_if_constraints_demand():
    # Negative published values are unusual but mathematically solvable.
    result = reconcile(build(), BASE, publish("S1", -6))
    assert result.periods[0].corrected["S1"] == pytest.approx(-6)


def test_period_ordering():
    base = []
    for period in ["2026-10", "2026-02", "2025-12"]:
        for line, value in (("L1", 1), ("L2", 2), ("L3", 3)):
            base.append({"node_id": line, "period": period, "value": value})
    result = reconcile(build(), base, [])
    assert [p.period for p in result.periods] == ["2025-12", "2026-02", "2026-10"]


def test_line_level_total_rejected():
    with pytest.raises(SnapshotValidationError, match="line"):
        reconcile(build(), BASE, publish("line", 60, kind="level"))


def test_single_line_can_be_pinned():
    result = reconcile(build(), BASE, publish("L1", 50))
    p = result.periods[0]
    assert p.corrected["L1"] == pytest.approx(50)
    assert p.corrected["S1"] == pytest.approx(p.corrected["L1"] + p.corrected["L2"])
    assert verify_consistency(result) == []


def test_plant_and_shop_level_totals_must_agree():
    with pytest.raises(Exception):
        reconcile(
            build(),
            BASE,
            publish("plant", 100, kind="level") + publish("shop", 54, kind="level"),
        )
