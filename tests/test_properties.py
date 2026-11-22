"""Randomised cross-checks and invariants.

These tests build random trees and random published targets, then compare the
engine's answer against an independent KKT solve and check every invariant the
business rules require.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from app.csv_io import MappingRow
from app.errors import SnapshotValidationError
from app.hierarchy import Hierarchy, build_hierarchy
from app.reconcile import descendant_lines, reconcile, verify_consistency


def random_tree(rng: random.Random, n_shops: int, lines_per_shop: int) -> Hierarchy:
    rows = [MappingRow("P1", None, "plant", "P1", rng.choice([0.5, 1, 2, 5]))]
    for s in range(n_shops):
        shop = f"S{s}"
        rows.append(MappingRow(shop, "P1", "shop", shop, rng.choice([0.5, 1, 2, 5])))
        for l in range(lines_per_shop):
            line = f"L{s}_{l}"
            rows.append(MappingRow(line, shop, "line", line, rng.choice([0.1, 0.5, 1, 3, 10])))
    return build_hierarchy(rows)


def kkt_solution(hierarchy: Hierarchy, line_base: dict[str, float], targets: dict[str, float]):
    lines = hierarchy.by_level["line"]
    n = len(lines)
    idx = {l: i for i, l in enumerate(lines)}
    nodes = sorted(hierarchy.nodes)

    A = np.zeros((len(nodes), n))
    b = np.zeros(len(nodes))
    w = np.zeros(len(nodes))
    for r, node in enumerate(nodes):
        for l in descendant_lines(hierarchy, node):
            A[r, idx[l]] = 1.0
        b[r] = sum(line_base[l] for l in descendant_lines(hierarchy, node))
        w[r] = hierarchy.nodes[node].weight
    H = A.T @ (w[:, None] * A)
    g = A.T @ (w * b)

    C = np.zeros((len(targets), n))
    d = np.zeros(len(targets))
    for r, (node, value) in enumerate(sorted(targets.items())):
        for l in descendant_lines(hierarchy, node):
            C[r, idx[l]] = 1.0
        d[r] = value

    kkt = np.block([[2 * H, C.T], [C, np.zeros((len(targets), len(targets)))]])
    rhs = np.concatenate([2 * g, d])
    sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
    return {l: float(sol[idx[l]]) for l in lines}, H, g


def objective(hierarchy, line_base, values) -> float:
    total = 0.0
    for node in hierarchy.nodes.values():
        v = sum(values[l] for l in descendant_lines(hierarchy, node.node_id))
        b = sum(line_base[l] for l in descendant_lines(hierarchy, node.node_id))
        total += node.weight * (v - b) ** 2
    return total


def test_random_trees_match_independent_kkt_solve():
    rng = random.Random(20260101)
    for trial in range(25):
        hierarchy = random_tree(rng, rng.randint(1, 3), rng.randint(1, 3))
        lines = hierarchy.by_level["line"]
        period = "2026-01"
        line_base = {l: rng.uniform(0, 100) for l in lines}
        rows = [{"node_id": l, "period": period, "value": v} for l, v in line_base.items()]

        # Constrain a random non-empty subset of shops and possibly the plant.
        shops = hierarchy.by_level["shop"]
        chosen = rng.sample(shops, rng.randint(1, len(shops)))
        targets = {}
        for node in chosen:
            base_val = sum(line_base[l] for l in descendant_lines(hierarchy, node))
            targets[node] = base_val * rng.uniform(0.7, 1.3)
        include_plant = rng.random() < 0.3
        all_shops = set(chosen) == set(shops)
        if include_plant:
            plant_base = sum(line_base[l] for l in descendant_lines(hierarchy, "P1"))
            if all_shops:
                # Every shop is pinned, so the plant total is fully determined.
                targets["P1"] = sum(targets[s] for s in shops)
            else:
                targets["P1"] = plant_base * rng.uniform(0.8, 1.2)
        published = [
            {"target": n, "kind": "node", "period": period, "value": v}
            for n, v in targets.items()
        ]

        result = reconcile(hierarchy, rows, published)
        got = {l: result.periods[0].corrected[l] for l in lines}
        expected, _, _ = kkt_solution(hierarchy, line_base, targets)

        for line in lines:
            assert got[line] == pytest.approx(expected[line], rel=1e-6, abs=1e-6), (
                f"trial {trial} line {line}"
            )
        assert verify_consistency(result) == []
        # Published targets are hit exactly.
        for node, value in targets.items():
            actual = sum(got[l] for l in descendant_lines(hierarchy, node))
            assert actual == pytest.approx(value, rel=1e-7, abs=1e-6)
        # Objective is optimal: no feasible direction reduces it.  Build the
        # null space of the constraint matrix, which is exactly the set of
        # feasible perturbation directions.
        C = np.zeros((len(targets), len(lines)))
        idx = {l: i for i, l in enumerate(lines)}
        for r, node in enumerate(sorted(targets)):
            for l in descendant_lines(hierarchy, node):
                C[r, idx[l]] = 1.0
        _, s, vt = np.linalg.svd(C, full_matrices=True)
        rank = int((s > (max(C.shape) * s[0] * np.finfo(float).eps)).sum()) if len(s) else 0
        Z = vt[rank:].T
        base_obj = objective(hierarchy, line_base, got)
        for k in range(Z.shape[1]):
            for sign in (1.0, -1.0):
                step = 1e-3 * sign
                perturbed = {l: got[l] + step * Z[idx[l], k] for l in lines}
                assert objective(hierarchy, line_base, perturbed) >= base_obj - 1e-9


def test_random_infeasible_constraints_are_rejected():
    rng = random.Random(7)
    for _ in range(15):
        hierarchy = random_tree(rng, 2, 2)
        lines = hierarchy.by_level["line"]
        rows = [{"node_id": l, "period": "p", "value": 10.0} for l in lines]
        shops = hierarchy.by_level["shop"]
        published = [
            {"target": "P1", "kind": "node", "period": "p", "value": 999.0},
            {"target": shops[0], "kind": "node", "period": "p", "value": 10.0},
            {"target": shops[1], "kind": "node", "period": "p", "value": 10.0},
        ]
        with pytest.raises(SnapshotValidationError):
            reconcile(hierarchy, rows, published)


def test_random_weights_only_change_magnitude_not_consistency():
    rng = random.Random(99)
    hierarchy = random_tree(rng, 3, 3)
    lines = hierarchy.by_level["line"]
    rows = [{"node_id": l, "period": "p", "value": float(i + 1) * 10} for i, l in enumerate(lines)]
    published = [{"target": "P1", "kind": "node", "period": "p", "value": 1000.0}]
    for _ in range(10):
        for node in hierarchy.nodes.values():
            node.weight = rng.choice([0.01, 0.1, 1, 10, 100])
        result = reconcile(hierarchy, rows, published)
        assert verify_consistency(result) == []
        assert result.periods[0].corrected["P1"] == pytest.approx(1000.0)
