"""Core reconciliation engine.

Problem statement
-----------------
We are given

* a tree ``line -> shop -> plant``,
* a *base* forecast per line per period (the analyst's upload),
* optional *published* aggregates: a value for a shop node, a plant node, or a
  whole level (the total across all nodes of that level), which must hold
  exactly for the corrected snapshot.

The corrected snapshot must satisfy, for every period and every internal node,
``parent == sum(children)``.  Among all snapshots satisfying the published
constraints we pick the one minimising

    sum_over_all_nodes (weight_node * (x_node - base_node)^2)

where ``base_node`` for an internal node is the sum of its line descendants in
the uploaded forecast.  Internal values are exact sums of line values, so the
problem reduces to a weighted least-squares over line variables with linear
equality constraints.  It is solved per period by null-space reduction: find one
particular solution, then minimise over the null space of the constraint matrix
with numpy.  Infeasible constraints are reported, never silently violated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import OptimizationError, SnapshotValidationError
from .hierarchy import Hierarchy

def _close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


@dataclass
class PeriodResult:
    period: str
    corrected: dict[str, float]
    base: dict[str, float]
    deltas: dict[str, float]
    conflicts: list[dict[str, Any]]
    constrained: list[dict[str, Any]]
    node_adjustments: list[dict[str, Any]]
    line_adjustments: list[dict[str, Any]]

    @property
    def adjusted(self) -> bool:
        return bool(self.node_adjustments)

    @property
    def inconsistent(self) -> bool:
        return bool(self.conflicts)

    def to_dict(self) -> dict:
        return {
            "period": self.period,
            "corrected": self.corrected,
            "base": self.base,
            "deltas": self.deltas,
            "conflicts": self.conflicts,
            "constrained": self.constrained,
            "node_adjustments": self.node_adjustments,
            "line_adjustments": self.line_adjustments,
            "adjusted": self.adjusted,
            "inconsistent": self.inconsistent,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PeriodResult":
        return cls(
            period=data["period"],
            corrected=data["corrected"],
            base=data["base"],
            deltas=data.get("deltas", {}),
            conflicts=data.get("conflicts", []),
            constrained=data.get("constrained", []),
            node_adjustments=data.get("node_adjustments", []),
            line_adjustments=data.get("line_adjustments", []),
        )


@dataclass
class ReconciliationResult:
    periods: list[PeriodResult]
    hierarchy: Hierarchy
    warnings: list[str] = field(default_factory=list)

    @property
    def adjusted_periods(self) -> list[str]:
        return [p.period for p in self.periods if p.adjusted]

    def to_dict(self) -> dict:
        return {
            "periods": [p.to_dict() for p in self.periods],
            "hierarchy": self.hierarchy.to_dict(),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReconciliationResult":
        return cls(
            periods=[PeriodResult.from_dict(p) for p in data["periods"]],
            hierarchy=Hierarchy.from_dict(data["hierarchy"]),
            warnings=list(data.get("warnings", [])),
        )


# ----------------------------------------------------------------------
# Hierarchy helpers
# ----------------------------------------------------------------------
def descendant_lines(hierarchy: Hierarchy, node_id: str) -> list[str]:
    """Line leaves under (or at) ``node_id``, sorted."""
    out: list[str] = []
    stack = [node_id]
    seen: set[str] = set()
    while stack:
        cur = stack.pop()
        if cur in seen:  # a validated hierarchy is a tree; guard against cycles
            continue
        seen.add(cur)
        node = hierarchy.nodes[cur]
        if node.level == "line":
            out.append(cur)
        else:
            stack.extend(node.children)
    return sorted(out)


def node_base_value(hierarchy: Hierarchy, node_id: str, line_base: dict[str, float]) -> float:
    return sum(line_base[line] for line in descendant_lines(hierarchy, node_id))


# ----------------------------------------------------------------------
# Input normalisation
# ----------------------------------------------------------------------
def _period_sort_key(period: str) -> tuple:
    match = re.match(r"^(\d{4})-?(\d{2})?-?(\d{2})?", period or "")
    if match:
        return tuple(int(g) if g else 0 for g in match.groups()) + (period,)
    return (9999, 0, 0, 0, period)


def _normalise_line_rows(
    hierarchy: Hierarchy, line_rows: list[dict[str, Any]]
) -> tuple[dict[tuple[str, str], float], list[str]]:
    known = set(hierarchy.by_level["line"])
    line_values: dict[tuple[str, str], float] = {}
    duplicates: set[tuple[str, str]] = set()
    unknown: set[str] = set()
    for row in line_rows:
        node_id = str(row["node_id"])
        period = str(row["period"])
        if node_id not in known:
            unknown.add(node_id)
            continue
        key = (node_id, period)
        if key in line_values:
            duplicates.add(key)
        line_values[key] = float(row["value"])
    if unknown:
        raise SnapshotValidationError(
            "产线预测中包含映射里不存在的产线：" + ", ".join(sorted(unknown)[:10])
        )
    if duplicates:
        raise SnapshotValidationError(
            "产线预测存在重复的产线/时段行："
            + ", ".join(f"{n}@{p}" for n, p in sorted(duplicates)[:8])
        )
    if not line_values:
        raise SnapshotValidationError("产线预测为空。")
    periods = sorted({p for (_n, p) in line_values}, key=_period_sort_key)
    missing = [
        f"{line}@{period}"
        for period in periods
        for line in sorted(known)
        if (line, period) not in line_values
    ]
    if missing:
        preview = ", ".join(missing[:8])
        more = f" 等 {len(missing)} 处" if len(missing) > 8 else ""
        raise SnapshotValidationError(f"产线预测缺少以下时段/产线：{preview}{more}。")
    return line_values, periods


def _build_constraints(
    hierarchy: Hierarchy,
    published_rows: list[dict[str, Any]],
    periods: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Convert published rows into node pins and level-total constraints.

    Returns ``(constraints, warnings)``.  Each constraint has ``period``,
    ``value``, ``label`` and either ``node_id`` or ``level``.
    """
    warnings: list[str] = []
    period_set = set(periods)
    constraints: list[dict[str, Any]] = []
    seen_node: dict[tuple[str, str], dict[str, Any]] = {}

    for row in published_rows:
        target = str(row["target"])
        kind = row.get("kind", "node")
        period = str(row["period"])
        value = float(row["value"])
        if period not in period_set:
            warnings.append(
                f"已发布预测 {target}@ {period} 不在产线预测窗口内，未参与校正。"
            )
            continue
        if kind == "level":
            if target not in hierarchy.by_level:
                raise SnapshotValidationError(
                    f"已发布预测引用了未知层级 {target!r}（应为 plant / shop / line）。"
                )
            if target == "line":
                # Every line would be forced to the published number, which is
                # almost certainly a mistaken file.  Reject instead of guessing.
                raise SnapshotValidationError(
                    "不支持按 line 层级发布合计；请给出具体产线、车间或工厂的发布值。"
                )
            constraints.append(
                {
                    "kind": "level",
                    "level": target,
                    "period": period,
                    "value": value,
                    "label": f"{target} 合计",
                }
            )
            continue
        if target not in hierarchy.nodes:
            raise SnapshotValidationError(f"已发布预测引用了未知节点 {target!r}。")
        key = (target, period)
        if key in seen_node:
            prev = seen_node[key]
            if not _close(prev["value"], value):
                raise SnapshotValidationError(
                    f"节点 {target}@ {period} 有两条互相冲突的已发布预测："
                    f"{prev['value']} 与 {value}。"
                )
            continue
        constraint = {
            "kind": "node",
            "node_id": target,
            "level": hierarchy.nodes[target].level,
            "period": period,
            "value": value,
            "label": f"{target}（{hierarchy.nodes[target].level}）",
        }
        seen_node[key] = constraint
        constraints.append(constraint)

    # A published parent that is also fully decomposed by published children
    # must equal their sum, otherwise no consistent snapshot exists.
    for (node_id, period), constraint in sorted(seen_node.items()):
        node = hierarchy.nodes[node_id]
        if node.level == "line":
            continue
        child_constraints = [seen_node.get((c, period)) for c in node.children]
        if all(c is not None for c in child_constraints):
            child_sum = sum(c["value"] for c in child_constraints)
            if not _close(child_sum, constraint["value"]):
                raise SnapshotValidationError(
                    f"已发布预测自相矛盾：{node_id}@ {period} 为 {constraint['value']}，"
                    f"但其子节点已发布值合计为 {child_sum}。"
                )

    # A level total for a level whose nodes are *all* individually pinned would
    # be a redundant (and possibly conflicting) constraint; reject a conflict
    # now with a clear message instead of a generic solver failure.
    for constraint in constraints:
        if constraint["kind"] != "level":
            continue
        members = hierarchy.by_level[constraint["level"]]
        pinned = [seen_node.get((m, constraint["period"])) for m in members]
        if all(p is not None for p in pinned):
            total = sum(p["value"] for p in pinned)
            if not _close(total, constraint["value"]):
                raise SnapshotValidationError(
                    f"已发布预测自相矛盾：{constraint['label']}@ {constraint['period']} 为 "
                    f"{constraint['value']}，但其节点发布值合计为 {total}。"
                )
    return constraints, warnings


# ----------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------
def _solve_period(
    hierarchy: Hierarchy,
    lines: list[str],
    line_base: dict[str, float],
    constraints: list[dict[str, Any]],
) -> tuple[dict[str, float], list[str]]:
    """Solve one period; returns corrected line values and solver notes."""
    n = len(lines)
    index = {line: i for i, line in enumerate(lines)}

    def indicator(node_id: str) -> np.ndarray:
        vec = np.zeros(n, dtype=float)
        for line in descendant_lines(hierarchy, node_id):
            vec[index[line]] = 1.0
        return vec

    # Objective over all nodes: || sqrt(W) (A x - b) ||^2.
    all_nodes = sorted(hierarchy.nodes)
    A = np.zeros((len(all_nodes), n), dtype=float)
    b = np.zeros(len(all_nodes), dtype=float)
    sqrt_w = np.zeros(len(all_nodes), dtype=float)
    for row, node_id in enumerate(all_nodes):
        A[row] = indicator(node_id)
        b[row] = node_base_value(hierarchy, node_id, line_base)
        sqrt_w[row] = float(np.sqrt(max(hierarchy.nodes[node_id].weight, 0.0)))
    Aw = A * sqrt_w[:, None]
    bw = b * sqrt_w
    hess = Aw.T @ Aw
    grad_const = Aw.T @ bw

    # Tie-breaker for zero-weight nodes.  A node with weight 0 has no cost, so
    # infinitely many optima exist; a vanishing quadratic pull towards the base
    # keeps the result deterministic without measurably changing the optimum.
    epsilon = 1e-10 * max(1.0, float(np.max(sqrt_w) ** 2))
    hess = hess + epsilon * np.eye(n)
    grad_const = grad_const + epsilon * np.array([line_base[line] for line in lines])

    C_rows: list[np.ndarray] = []
    d_vals: list[float] = []
    for constraint in constraints:
        if constraint["kind"] == "node":
            C_rows.append(indicator(constraint["node_id"]))
        else:
            combined = np.zeros(n, dtype=float)
            for node_id in hierarchy.by_level[constraint["level"]]:
                combined += indicator(node_id)
            C_rows.append(combined)
        d_vals.append(constraint["value"])

    if not C_rows:
        return {line: float(line_base[line]) for line in lines}, []

    C = np.vstack(C_rows)
    d = np.array(d_vals, dtype=float)

    # Rank / feasibility check: redundant constraints must agree.
    x_ls, *_ = np.linalg.lstsq(C, d, rcond=None)
    residual = C @ x_ls - d
    if np.max(np.abs(residual)) > 1e-6 * max(1.0, float(np.max(np.abs(d)))):
        raise OptimizationError(
            "已发布预测无法同时满足（约束不相容），请核对该时段的车间/工厂发布值。"
        )

    x0 = x_ls
    _, s, vt = np.linalg.svd(C, full_matrices=True)
    tol = max(C.shape) * (s[0] if len(s) else 0.0) * np.finfo(float).eps
    rank = int((s > tol).sum())
    Z = vt[rank:].T  # null space basis, n x (n-rank)

    if Z.shape[1] > 0:
        Hz = Z.T @ hess @ Z
        gz = Z.T @ (hess @ x0 - grad_const)
        # Ridge keeps the solve stable for nodes with zero weight; with a
        # positive-definite reduced Hessian it does not move the optimum.
        Hz = Hz + np.eye(Hz.shape[0]) * 1e-12 * max(1.0, float(np.trace(Hz)))
        try:
            z = np.linalg.solve(Hz, -gz)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
            raise OptimizationError("加权最小二乘求解失败。") from exc
        x = x0 + Z @ z
    else:
        x = x0

    if not np.all(np.isfinite(x)):
        raise OptimizationError("加权最小二乘求解得到非有限结果。")
    # Guard against numerical drift on the constraints.
    if np.max(np.abs(C @ x - d)) > 1e-6 * max(1.0, float(np.max(np.abs(d)))):
        raise OptimizationError("校正结果未满足已发布约束。")

    return {line: float(x[index[line]]) for line in lines}, []


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def reconcile(
    hierarchy: Hierarchy,
    line_rows: list[dict[str, Any]],
    published_rows: list[dict[str, Any]],
) -> ReconciliationResult:
    """Reconcile line forecasts against published aggregates.

    ``line_rows``: ``{node_id, period, value}``.
    ``published_rows``: ``{target, kind: 'node'|'level', period, value}``.
    """
    line_values, periods = _normalise_line_rows(hierarchy, line_rows)
    constraints, warnings = _build_constraints(hierarchy, published_rows, periods)

    lines = hierarchy.by_level["line"]
    constraints_by_period: dict[str, list[dict[str, Any]]] = {p: [] for p in periods}
    for constraint in constraints:
        constraints_by_period[constraint["period"]].append(constraint)

    periods_out: list[PeriodResult] = []
    for period in periods:
        line_base = {line: line_values[(line, period)] for line in lines}
        base_all = {
            node_id: node_base_value(hierarchy, node_id, line_base)
            for node_id in sorted(hierarchy.nodes)
        }
        period_constraints = constraints_by_period[period]
        corrected_lines, _notes = _solve_period(
            hierarchy, lines, line_base, period_constraints
        )
        corrected_all = dict(corrected_lines)
        for node in hierarchy.nodes.values():
            if node.level != "line":
                corrected_all[node.node_id] = float(
                    sum(corrected_lines[l] for l in descendant_lines(hierarchy, node.node_id))
                )

        # Conflicts are published values that disagree with the uploaded base.
        conflicts: list[dict[str, Any]] = []
        constrained_entries: list[dict[str, Any]] = []
        for constraint in period_constraints:
            label = constraint["label"]
            constrained_entries.append(
                {
                    "label": label,
                    "kind": constraint["kind"],
                    "node_id": constraint.get("node_id"),
                    "level": constraint.get("level"),
                    "value": constraint["value"],
                }
            )
            if constraint["kind"] == "node":
                node_id = constraint["node_id"]
                diff = constraint["value"] - base_all[node_id]
                if abs(diff) > 1e-9 * max(1.0, abs(base_all[node_id]), abs(constraint["value"])):
                    conflicts.append(
                        {
                            "scope": node_id,
                            "level": hierarchy.nodes[node_id].level,
                            "base": base_all[node_id],
                            "published": constraint["value"],
                            "diff": diff,
                        }
                    )
            else:
                level = constraint["level"]
                base_total = sum(base_all[n] for n in hierarchy.by_level[level])
                diff = constraint["value"] - base_total
                if abs(diff) > 1e-9 * max(1.0, abs(base_total), abs(constraint["value"])):
                    conflicts.append(
                        {
                            "scope": f"{level} 合计",
                            "level": level,
                            "base": base_total,
                            "published": constraint["value"],
                            "diff": diff,
                        }
                    )

        # Node-level adjustments, with their source: a node is "published" if
        # it is directly constrained; "level_total" if only its level total is
        # constrained; otherwise it moved to rebalance the tree.
        constrained_node_ids = {
            c["node_id"] for c in period_constraints if c["kind"] == "node"
        }
        constrained_levels = {
            c["level"] for c in period_constraints if c["kind"] == "level"
        }
        node_adjustments: list[dict[str, Any]] = []
        for node_id in sorted(hierarchy.nodes):
            node = hierarchy.nodes[node_id]
            delta = corrected_all[node_id] - base_all[node_id]
            if abs(delta) <= 1e-9 * max(1.0, abs(base_all[node_id])):
                continue
            if node_id in constrained_node_ids:
                source = "published_node"
            elif node.level in constrained_levels:
                source = "published_level"
            else:
                source = "weighted_balance"
            node_adjustments.append(
                {
                    "node_id": node_id,
                    "level": node.level,
                    "name": node.name,
                    "base": base_all[node_id],
                    "corrected": corrected_all[node_id],
                    "delta": delta,
                    "source": source,
                    "path": [node_id] + hierarchy.ancestors(node_id),
                }
            )

        # Line-level explanation used by the web page: which published nodes
        # influenced this line.
        line_adjustments: list[dict[str, Any]] = []
        for line in lines:
            delta = corrected_lines[line] - line_base[line]
            if abs(delta) <= 1e-9 * max(1.0, abs(line_base[line])):
                continue
            sources: list[str] = []
            ancestors = set(hierarchy.ancestors(line))
            for constraint in period_constraints:
                if constraint["kind"] == "node":
                    if line in descendant_lines(hierarchy, constraint["node_id"]):
                        sources.append(constraint["label"])
                else:
                    # A level total spans every node of that level.  It can
                    # influence this line when the line's own ancestors include
                    # a node of that level.
                    if constraint["level"] in {"plant", "shop"}:
                        if any(
                            hierarchy.nodes[a].level == constraint["level"]
                            for a in ancestors
                        ):
                            sources.append(constraint["label"])
            line_adjustments.append(
                {
                    "node_id": line,
                    "level": "line",
                    "name": hierarchy.nodes[line].name,
                    "base": line_base[line],
                    "corrected": corrected_lines[line],
                    "delta": delta,
                    "source_nodes": sorted(set(sources)) or ["weighted_balance"],
                    "path": [line] + hierarchy.ancestors(line),
                }
            )

        periods_out.append(
            PeriodResult(
                period=period,
                corrected=corrected_all,
                base=base_all,
                deltas={
                    n: corrected_all[n] - base_all[n]
                    for n in sorted(hierarchy.nodes)
                    if abs(corrected_all[n] - base_all[n])
                    > 1e-9 * max(1.0, abs(base_all[n]))
                },
                conflicts=conflicts,
                constrained=constrained_entries,
                node_adjustments=node_adjustments,
                line_adjustments=line_adjustments,
            )
        )

    return ReconciliationResult(periods=periods_out, hierarchy=hierarchy, warnings=warnings)


def verify_consistency(result: ReconciliationResult, tolerance: float = 1e-6) -> list[str]:
    """Every internal node must equal the sum of its children after correction."""
    problems: list[str] = []
    hierarchy = result.hierarchy
    for period in result.periods:
        for node in hierarchy.nodes.values():
            if node.level == "line":
                continue
            child_sum = sum(period.corrected[c] for c in node.children)
            if abs(child_sum - period.corrected[node.node_id]) > tolerance * max(
                1.0, abs(child_sum), abs(period.corrected[node.node_id])
            ):
                problems.append(
                    f"{node.node_id}@{period.period}: "
                    f"{period.corrected[node.node_id]} != 子节点合计 {child_sum}"
                )
    return problems


def verify_hierarchy(hierarchy: Hierarchy) -> list[str]:
    """Structural safety net used before publishing a stored preview.

    Re-checks the properties enforced at upload time so a corrupted or
    hand-edited preview can never be published.
    """
    problems: list[str] = []
    for node in hierarchy.nodes.values():
        if node.parent_id is not None and node.parent_id not in hierarchy.nodes:
            problems.append(f"{node.node_id} 的父节点 {node.parent_id} 不存在")
    for start in hierarchy.nodes:
        seen: set[str] = set()
        cur: str | None = start
        while cur is not None:
            if cur in seen:
                problems.append(f"映射成环（涉及 {start}）")
                break
            seen.add(cur)
            parent = hierarchy.nodes.get(cur)
            if parent is None:
                break
            cur = parent.parent_id
    if len(hierarchy.roots) != 1:
        problems.append("根节点数量不为 1")
    for level in ("plant", "shop", "line"):
        if not hierarchy.by_level.get(level):
            problems.append(f"缺少 {level} 层级")
    return problems
