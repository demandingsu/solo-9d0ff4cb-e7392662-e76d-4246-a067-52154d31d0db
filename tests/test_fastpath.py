"""Closed-form solver: equivalence with the dense solver, dispatch rules,
line-count limit enforcement, and large-scale sanity checks."""

from __future__ import annotations

import random

import pytest
from fastapi.testclient import TestClient

from app import reconcile as reconcile_module
from app.csv_io import MappingRow
from app.db import Database
from app.errors import LineLimitExceededError, OptimizationError
from app.hierarchy import build_hierarchy
from app.main import _max_lines_from_env, create_app
from app.reconcile import (
    _solve_period,
    _solve_period_fast,
    descendant_lines,
    reconcile,
    verify_consistency,
)
from app.service import DEFAULT_MAX_LINES, ReconciliationService

from .conftest import upload_text


def random_tree(rng: random.Random, n_shops: int, lines_per_shop: int):
    # Shop/plant weights may be zero; line weights are always positive so the
    # periods stay eligible for the closed-form solver.
    rows = [MappingRow("P1", None, "plant", "P1", rng.choice([0.0, 0.5, 1, 2, 5]))]
    for s in range(n_shops):
        shop = f"S{s}"
        rows.append(MappingRow(shop, "P1", "shop", shop, rng.choice([0.0, 0.5, 1, 2, 5])))
        for l in range(lines_per_shop):
            line = f"L{s}_{l}"
            rows.append(MappingRow(line, shop, "line", line, rng.choice([0.1, 0.5, 1, 3, 10])))
    return build_hierarchy(rows)


# ----------------------------------------------------------------------
# Equivalence with the dense solver
# ----------------------------------------------------------------------
def test_fast_solver_matches_general_solver_on_random_trees():
    rng = random.Random(20261122)
    for trial in range(40):
        hierarchy = random_tree(rng, rng.randint(1, 4), rng.randint(1, 5))
        lines = hierarchy.by_level["line"]
        shops = hierarchy.by_level["shop"]
        line_base = {l: rng.uniform(-50, 150) for l in lines}

        constraints = []
        chosen = rng.sample(shops, rng.randint(0, len(shops)))
        pins: dict[str, float] = {}
        for shop in chosen:
            base = sum(line_base[l] for l in descendant_lines(hierarchy, shop))
            pins[shop] = base * rng.uniform(0.5, 1.6)
            constraints.append(
                {"kind": "node", "node_id": shop, "level": "shop",
                 "period": "p", "value": pins[shop], "label": f"{shop}（shop）"}
            )
        all_pinned = len(chosen) == len(shops)

        def feasible_grand_total() -> float:
            if all_pinned:
                return sum(pins.values())  # fully determined by the shop pins
            return sum(line_base.values()) * rng.uniform(0.6, 1.5)

        grand: float | None = None
        if rng.random() < 0.5:
            grand = feasible_grand_total()
            constraints.append(
                {"kind": "node", "node_id": "P1", "level": "plant",
                 "period": "p", "value": grand, "label": "P1（plant）"}
            )
        if rng.random() < 0.3:
            if grand is None:
                grand = feasible_grand_total()
            constraints.append(
                {"kind": "level", "level": "shop", "period": "p",
                 "value": grand, "label": "shop 合计"}
            )
        if not constraints:
            continue

        expected, _ = _solve_period(hierarchy, lines, line_base, constraints)
        got, _ = _solve_period_fast(hierarchy, lines, line_base, constraints)
        assert set(got) == set(expected)
        for line in lines:
            assert got[line] == pytest.approx(expected[line], rel=1e-6, abs=1e-6), (
                f"trial {trial} line {line}"
            )


def test_fast_solver_rejects_conflicting_grand_totals():
    rng = random.Random(3)
    hierarchy = random_tree(rng, 2, 2)
    lines = hierarchy.by_level["line"]
    line_base = {l: 10.0 for l in lines}
    constraints = [
        {"kind": "node", "node_id": "P1", "level": "plant", "period": "p",
         "value": 100.0, "label": "P1（plant）"},
        {"kind": "level", "level": "shop", "period": "p", "value": 54.0,
         "label": "shop 合计"},
    ]
    with pytest.raises(OptimizationError):
        _solve_period_fast(hierarchy, lines, line_base, constraints)
    with pytest.raises(OptimizationError):
        _solve_period(hierarchy, lines, line_base, constraints)


def test_fast_solver_agrees_on_redundant_grand_totals():
    rng = random.Random(11)
    hierarchy = random_tree(rng, 3, 2)
    lines = hierarchy.by_level["line"]
    line_base = {l: rng.uniform(1, 50) for l in lines}
    constraints = [
        {"kind": "node", "node_id": "P1", "level": "plant", "period": "p",
         "value": 200.0, "label": "P1（plant）"},
        {"kind": "level", "level": "shop", "period": "p", "value": 200.0,
         "label": "shop 合计"},
        {"kind": "level", "level": "plant", "period": "p", "value": 200.0,
         "label": "plant 合计"},
    ]
    expected, _ = _solve_period(hierarchy, lines, line_base, constraints)
    got, _ = _solve_period_fast(hierarchy, lines, line_base, constraints)
    for line in lines:
        assert got[line] == pytest.approx(expected[line], rel=1e-6, abs=1e-6)
    assert sum(got.values()) == pytest.approx(200.0)


def test_remaining_shops_jointly_cover_the_difference():
    """工厂总量与部分车间值同时指定时，其余车间共同补足差额。"""
    rows = [
        MappingRow("P1", None, "plant", "P1", 1.0),
        MappingRow("S1", "P1", "shop", "S1", 1.0),
        MappingRow("S2", "P1", "shop", "S2", 1.0),
        MappingRow("S3", "P1", "shop", "S3", 1.0),
        MappingRow("L1", "S1", "line", "L1", 1.0),
        MappingRow("L2", "S2", "line", "L2", 1.0),
        MappingRow("L3", "S3", "line", "L3", 1.0),
    ]
    hierarchy = build_hierarchy(rows)
    base = [
        {"node_id": "L1", "period": "2026-01", "value": 20.0},
        {"node_id": "L2", "period": "2026-01", "value": 30.0},
        {"node_id": "L3", "period": "2026-01", "value": 40.0},
    ]
    published = [
        {"target": "P1", "kind": "node", "period": "2026-01", "value": 120.0},
        {"target": "S1", "kind": "node", "period": "2026-01", "value": 30.0},
    ]
    result = reconcile(hierarchy, base, published)
    p = result.periods[0]
    # S1 pinned; S2+S3 must jointly supply the remaining 120 - 30 = 90.
    assert p.corrected["S1"] == pytest.approx(30.0)
    assert p.corrected["S2"] + p.corrected["S3"] == pytest.approx(90.0)
    assert p.corrected["P1"] == pytest.approx(120.0)
    # Both free shops move up from base (30 + 40 = 70 -> 90).
    assert p.corrected["S2"] > 30.0
    assert p.corrected["S3"] > 40.0
    # Equal weights -> the +20 gap is shared equally between S2 and S3.
    assert p.corrected["S2"] - 30.0 == pytest.approx(p.corrected["S3"] - 40.0)
    assert verify_consistency(result) == []


# ----------------------------------------------------------------------
# Dispatch rules
# ----------------------------------------------------------------------
def _mixed_setup():
    rows = [
        MappingRow("P1", None, "plant", "P1", 1.0),
        MappingRow("S1", "P1", "shop", "S1", 1.0),
        MappingRow("S2", "P1", "shop", "S2", 1.0),
        MappingRow("L1", "S1", "line", "L1", 1.0),
        MappingRow("L2", "S1", "line", "L2", 1.0),
        MappingRow("L3", "S2", "line", "L3", 1.0),
    ]
    hierarchy = build_hierarchy(rows)
    base = [
        {"node_id": l, "period": p, "value": v}
        for p, vals in (
            ("2026-01", (10.0, 20.0, 30.0)),
            ("2026-02", (11.0, 21.0, 31.0)),
            ("2026-03", (12.0, 22.0, 32.0)),
        )
        for l, v in zip(("L1", "L2", "L3"), vals)
    ]
    published = [
        {"target": "S1", "kind": "node", "period": "2026-01", "value": 36.0},
        {"target": "L1", "kind": "node", "period": "2026-02", "value": 99.0},
    ]
    return hierarchy, base, published


def test_dispatch_per_period(monkeypatch):
    hierarchy, base, published = _mixed_setup()
    calls = {"fast": [], "general": []}
    real_fast = reconcile_module._solve_period_fast
    real_general = reconcile_module._solve_period

    def spy_fast(h, lines, lb, cons):
        calls["fast"].append(cons[0]["period"] if cons else "unconstrained")
        return real_fast(h, lines, lb, cons)

    def spy_general(h, lines, lb, cons):
        calls["general"].append(cons[0]["period"])
        return real_general(h, lines, lb, cons)

    monkeypatch.setattr(reconcile_module, "_solve_period_fast", spy_fast)
    monkeypatch.setattr(reconcile_module, "_solve_period", spy_general)
    result = reconcile(hierarchy, base, published, max_lines=1000)

    # 2026-01 (shop pin) -> closed form; 2026-02 (line pin) -> general;
    # 2026-03 (unconstrained) -> closed form trivially (no solver work).
    assert calls["general"] == ["2026-02"]
    assert "2026-01" in calls["fast"]
    by_period = {p.period: p for p in result.periods}
    assert by_period["2026-01"].corrected["S1"] == pytest.approx(36.0)
    assert by_period["2026-02"].corrected["L1"] == pytest.approx(99.0)
    assert by_period["2026-03"].adjusted is False
    assert verify_consistency(result) == []


def test_zero_line_weight_forces_general_solver(monkeypatch):
    rows = [
        MappingRow("P1", None, "plant", "P1", 1.0),
        MappingRow("S1", "P1", "shop", "S1", 1.0),
        MappingRow("L1", "S1", "line", "L1", 0.0),
        MappingRow("L2", "S1", "line", "L2", 1.0),
    ]
    hierarchy = build_hierarchy(rows)
    base = [
        {"node_id": "L1", "period": "2026-01", "value": 10.0},
        {"node_id": "L2", "period": "2026-01", "value": 20.0},
    ]
    published = [{"target": "S1", "kind": "node", "period": "2026-01", "value": 36.0}]

    called = []
    real_general = reconcile_module._solve_period

    def spy(h, lines, lb, cons):
        called.append(True)
        return real_general(h, lines, lb, cons)

    monkeypatch.setattr(reconcile_module, "_solve_period", spy)
    result = reconcile(hierarchy, base, published)
    assert called, "zero line weight must fall back to the general solver"
    assert result.periods[0].corrected["S1"] == pytest.approx(36.0)


def test_fast_solver_never_touches_numpy(monkeypatch):
    """The closed-form path must not build dense matrices at all."""

    class NoNumpy:
        def __getattr__(self, name):  # pragma: no cover - failure path
            raise AssertionError(f"fast path touched numpy.{name}")

    hierarchy, base, published = _mixed_setup()
    # Keep only the shop-pin period so every constrained period is eligible.
    published = [row for row in published if row["target"] == "S1"]
    monkeypatch.setattr(reconcile_module, "np", NoNumpy())
    result = reconcile(hierarchy, base, published)
    assert result.periods[0].corrected["S1"] == pytest.approx(36.0)


# ----------------------------------------------------------------------
# Line-count limit
# ----------------------------------------------------------------------
def test_over_limit_periods_are_listed_and_nothing_is_solved(monkeypatch):
    hierarchy, base, published = _mixed_setup()

    def boom(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("solver ran despite the over-limit rejection")

    monkeypatch.setattr(reconcile_module, "_solve_period", boom)
    monkeypatch.setattr(reconcile_module, "_solve_period_fast", boom)
    with pytest.raises(LineLimitExceededError) as excinfo:
        reconcile(hierarchy, base, published, max_lines=2)  # 3 lines > 2
    exc = excinfo.value
    # Only the line-pin period needs the general solver; the shop-pin and
    # unconstrained periods must not be listed.
    assert exc.periods == ["2026-02"]
    assert "2026-02" in str(exc)
    assert "2026-01" not in str(exc)
    assert exc.code == "line_limit_exceeded"
    assert exc.detail["n_lines"] == 3
    assert exc.detail["limit"] == 2


def test_under_limit_line_pin_still_uses_general_solver():
    hierarchy, base, published = _mixed_setup()
    result = reconcile(hierarchy, base, published, max_lines=3)
    assert result.periods[1].corrected["L1"] == pytest.approx(99.0)


def test_no_limit_by_default():
    hierarchy, base, published = _mixed_setup()
    result = reconcile(hierarchy, base, published)
    assert result.periods[1].corrected["L1"] == pytest.approx(99.0)


def test_zero_weight_over_limit_is_rejected():
    rows = [
        MappingRow("P1", None, "plant", "P1", 1.0),
        MappingRow("S1", "P1", "shop", "S1", 1.0),
        MappingRow("L1", "S1", "line", "L1", 0.0),
        MappingRow("L2", "S1", "line", "L2", 1.0),
    ]
    hierarchy = build_hierarchy(rows)
    base = [
        {"node_id": "L1", "period": "2026-01", "value": 10.0},
        {"node_id": "L2", "period": "2026-01", "value": 20.0},
    ]
    published = [{"target": "S1", "kind": "node", "period": "2026-01", "value": 36.0}]
    with pytest.raises(LineLimitExceededError):
        reconcile(hierarchy, base, published, max_lines=1)


# ----------------------------------------------------------------------
# Scale: 1000 lines stay flat in memory and exempt from the limit
# ----------------------------------------------------------------------
def _big_hierarchy(n_shops: int, lines_per_shop: int):
    rows = [MappingRow("P1", None, "plant", "P1", 1.0)]
    for s in range(n_shops):
        rows.append(MappingRow(f"S{s}", "P1", "shop", f"S{s}", 1.0 + (s % 3)))
        for l in range(lines_per_shop):
            rows.append(
                MappingRow(f"L{s}_{l}", f"S{s}", "line", f"L{s}_{l}", 0.5 + (l % 4))
            )
    return build_hierarchy(rows)


def test_thousand_lines_closed_form_stays_under_tiny_limit():
    hierarchy = _big_hierarchy(50, 20)  # 1000 lines
    lines = hierarchy.by_level["line"]
    assert len(lines) == 1000
    periods = [f"2026-{m:02d}" for m in range(1, 13)]
    base = [
        {"node_id": line, "period": p, "value": 10.0 + ((i * 7 + m) % 100) / 10.0}
        for m, p in enumerate(periods)
        for i, line in enumerate(lines)
    ]
    published = [
        {"target": "S0", "kind": "node", "period": p, "value": 500.0}
        for p in periods
    ] + [
        {"target": "P1", "kind": "node", "period": "2026-06", "value": 20000.0},
        {"target": "shop", "kind": "level", "period": "2026-07", "value": 21000.0},
    ]
    # Limit far below the line count: eligible periods are exempt.
    result = reconcile(hierarchy, base, published, max_lines=10)
    assert verify_consistency(result) == []
    by_period = {p.period: p for p in result.periods}
    for p in periods:
        assert by_period[p].corrected["S0"] == pytest.approx(500.0)
    assert by_period["2026-06"].corrected["P1"] == pytest.approx(20000.0)
    july_total = sum(by_period["2026-07"].corrected[l] for l in lines)
    assert july_total == pytest.approx(21000.0)


def test_thousand_lines_matches_general_solver_on_one_period():
    hierarchy = _big_hierarchy(20, 50)  # 1000 lines
    lines = hierarchy.by_level["line"]
    rng = random.Random(5)
    line_base = {l: rng.uniform(1, 100) for l in lines}
    constraints = [
        {"kind": "node", "node_id": "S3", "level": "shop", "period": "p",
         "value": 1234.0, "label": "S3（shop）"},
        {"kind": "node", "node_id": "P1", "level": "plant", "period": "p",
         "value": 55555.0, "label": "P1（plant）"},
    ]
    expected, _ = _solve_period(hierarchy, lines, line_base, constraints)
    got, _ = _solve_period_fast(hierarchy, lines, line_base, constraints)
    for line in lines:
        assert got[line] == pytest.approx(expected[line], rel=1e-6, abs=1e-6)


# ----------------------------------------------------------------------
# Service-level failure semantics
# ----------------------------------------------------------------------
MAPPING4 = """node_id,parent_id,level,name,weight
P1,,plant,工厂,1
S1,P1,shop,车间A,1
S2,P1,shop,车间B,1
L1,S1,line,产线1,1
L2,S1,line,产线2,1
L3,S2,line,产线3,1
"""

FORECAST4 = """node_id,period,value
L1,2026-01,10
L2,2026-01,20
L3,2026-01,30
"""

MAPPING5 = """node_id,parent_id,level,name,weight
P1,,plant,工厂,1
S1,P1,shop,车间A,1
S2,P1,shop,车间B,1
L1,S1,line,产线1,1
L2,S1,line,产线2,1
L3,S2,line,产线3,1
L4,S2,line,产线4,1
"""

FORECAST5 = """node_id,period,value
L1,2026-01,10
L2,2026-01,20
L3,2026-01,30
L4,2026-01,40
"""

PUBLISHED_LINE_PIN = """node_id,period,value
L4,2026-01,45
"""


def test_over_limit_preview_rejected_and_published_snapshot_survives(tmp_path):
    service = ReconciliationService(Database(tmp_path / "svc.db"), max_lines=3)
    ds = service.create_dataset("超限演示").id

    # 3 lines == limit: general solver still allowed; publish snapshot v1.
    upload_text(service, ds, "mapping", MAPPING4)
    upload_text(service, ds, "line_forecast", FORECAST4)
    upload_text(service, ds, "published", "node_id,period,value\nS1,2026-01,36\n")
    first = service.preview(ds)
    published = service.publish(ds, first["preview_id"])
    assert published["version_number"] == 1

    # Grow to 4 lines (> limit 3) and pin a line -> whole batch rejected.
    upload_text(service, ds, "mapping", MAPPING5)
    upload_text(service, ds, "line_forecast", FORECAST5)
    upload_text(service, ds, "published", PUBLISHED_LINE_PIN)
    with pytest.raises(LineLimitExceededError) as excinfo:
        service.preview(ds)
    assert "2026-01" in str(excinfo.value)

    state = service.dataset_state(ds)
    # No new preview was stored; the old one is stale and cannot be published.
    assert state["preview"]["id"] == first["preview_id"]
    assert state["preview"]["stale"] is True
    # The published snapshot is untouched and still served.
    assert state["latest_snapshot"]["version_number"] == 1
    assert service.db.get_snapshot(ds)["version_number"] == 1
    # The rejection was logged with the offending periods.
    failed = [e for e in state["events"] if e["kind"] == "preview_failed"]
    assert failed, "expected a preview_failed event"
    assert failed[0]["detail"]["error_code"] == "line_limit_exceeded"
    assert failed[0]["detail"]["periods"] == ["2026-01"]


def test_service_default_limit(tmp_path):
    service = ReconciliationService(Database(tmp_path / "default.db"))
    assert service.max_lines == DEFAULT_MAX_LINES


def test_env_configures_limit(monkeypatch, tmp_path):
    monkeypatch.setenv("RECONCILE_MAX_LINES", "3")
    assert _max_lines_from_env() == 3
    monkeypatch.delenv("RECONCILE_MAX_LINES")
    assert _max_lines_from_env() == DEFAULT_MAX_LINES
    monkeypatch.setenv("RECONCILE_MAX_LINES", "not-a-number")
    assert _max_lines_from_env() == DEFAULT_MAX_LINES

    monkeypatch.setenv("RECONCILE_MAX_LINES", "3")
    app = create_app(str(tmp_path / "env.db"))
    with TestClient(app) as client:
        resp = client.post("/api/datasets", json={"name": "env"})
        ds = resp.json()["id"]
        for kind, text in (
            ("mapping", MAPPING5),
            ("line_forecast", FORECAST5),
            ("published", PUBLISHED_LINE_PIN),
        ):
            assert client.post(
                f"/api/datasets/{ds}/upload",
                params={"kind": kind},
                files={"file": (f"{kind}.csv", text.encode(), "text/csv")},
            ).status_code == 200
        resp = client.post(f"/api/datasets/{ds}/preview")
        assert resp.status_code == 400
        body = resp.json()
        assert body["code"] == "line_limit_exceeded"
        assert "2026-01" in body["error"]
