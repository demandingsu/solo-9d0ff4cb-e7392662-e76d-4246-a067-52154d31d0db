"""Validation and indexing of the line -> shop -> plant tree mapping."""

from __future__ import annotations

from dataclasses import dataclass, field

from .csv_io import MappingRow
from .errors import MappingValidationError

KNOWN_LEVELS = ("plant", "shop", "line")


@dataclass
class Node:
    node_id: str
    parent_id: str | None
    level: str
    name: str
    weight: float
    children: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "level": self.level,
            "name": self.name,
            "weight": self.weight,
            "children": list(self.children),
        }


@dataclass
class Hierarchy:
    nodes: dict[str, Node]
    roots: list[str]
    by_level: dict[str, list[str]]

    def ancestors(self, node_id: str) -> list[str]:
        chain: list[str] = []
        seen: set[str] = set()
        cur = self.nodes[node_id].parent_id
        while cur is not None:
            if cur in seen:  # defensive; validation already rejects cycles
                raise MappingValidationError("映射中存在环。")
            seen.add(cur)
            chain.append(cur)
            cur = self.nodes[cur].parent_id
        return chain

    def to_dict(self) -> dict:
        return {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "roots": list(self.roots),
            "by_level": {k: list(v) for k, v in self.by_level.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Hierarchy":
        nodes = {
            k: Node(
                node_id=v["node_id"],
                parent_id=v.get("parent_id"),
                level=v["level"],
                name=v.get("name", v["node_id"]),
                weight=float(v["weight"]),
                children=list(v.get("children", [])),
            )
            for k, v in data["nodes"].items()
        }
        return cls(nodes, list(data["roots"]), {k: list(v) for k, v in data["by_level"].items()})


def _infer_level(depth: int) -> str:
    """Fallback level when the mapping omits a level column.

    Plant is the root, then shop, then line.  Deeper nodes stay ``line`` because
    the forecast grain is the line.
    """
    return ("plant", "shop", "line")[min(depth, 2)]


def build_hierarchy(rows: list[MappingRow]) -> Hierarchy:
    if not rows:
        raise MappingValidationError("映射文件没有有效行。")

    raw: dict[str, MappingRow] = {}
    duplicates: list[str] = []
    parent_candidates: dict[str, set[str | None]] = {}
    for row in rows:
        parent_candidates.setdefault(row.node_id, set()).add(row.parent_id)
        if row.node_id in raw:
            duplicates.append(row.node_id)
        else:
            raw[row.node_id] = row

    multi_parent = sorted(
        node for node, parents in parent_candidates.items() if len(parents) > 1
    )
    if multi_parent:
        details = "; ".join(
            f"{node} -> {sorted(str(p) for p in parent_candidates[node])}"
            for node in multi_parent[:10]
        )
        raise MappingValidationError(
            "同一节点出现多个父节点，映射必须是单亲树：" + details
        )
    if duplicates:
        raise MappingValidationError(
            "映射中存在重复节点行：" + ", ".join(sorted(set(duplicates))[:10])
        )

    # Parent references must exist.
    missing_parents = sorted(
        {r.parent_id for r in raw.values() if r.parent_id is not None and r.parent_id not in raw}
    )
    if missing_parents:
        raise MappingValidationError(
            "以下父节点在映射中不存在（缺层或拼写错误）：" + ", ".join(missing_parents[:10])
        )

    # Cycle detection with DFS colouring.  Self-parenting is a one-node cycle.
    state: dict[str, int] = {}  # 0 unvisited, 1 in stack, 2 done
    for start in raw:
        if state.get(start) == 2:
            continue
        path: list[str] = []
        cur: str | None = start
        while cur is not None and state.get(cur, 0) == 0:
            state[cur] = 1
            path.append(cur)
            cur = raw[cur].parent_id
        if cur is not None and state.get(cur) == 1:
            cycle_start = path.index(cur)
            cycle = path[cycle_start:] + [cur]
            raise MappingValidationError(
                "映射成环，无法确定层级：" + " -> ".join(cycle)
            )
        for node in path:
            state[node] = 2

    # Depth / level inference with explicit-level conflict detection.
    depth: dict[str, int] = {}

    def compute_depth(node_id: str) -> int:
        if node_id in depth:
            return depth[node_id]
        parent = raw[node_id].parent_id
        depth[node_id] = 0 if parent is None else compute_depth(parent) + 1
        return depth[node_id]

    nodes: dict[str, Node] = {}
    for node_id, row in raw.items():
        d = compute_depth(node_id)
        explicit = row.level.lower() if row.level else None
        inferred = _infer_level(d)
        if explicit is not None:
            if explicit not in KNOWN_LEVELS:
                raise MappingValidationError(
                    f"节点 {node_id} 的 level={row.level!r} 不受支持，应为 plant / shop / line。"
                )
            if explicit != inferred:
                raise MappingValidationError(
                    f"节点 {node_id} 的层级与树结构不一致：根深 {d} 应为 {inferred}，"
                    f"但标注为 {explicit}（缺层或层级错位）。"
                )
        level = explicit or inferred
        nodes[node_id] = Node(
            node_id=node_id,
            parent_id=row.parent_id,
            level=level,
            name=row.name or node_id,
            weight=row.weight if row.weight is not None else 1.0,
        )

    # Every leaf must be at line level; the forecast grain is the line.
    roots = sorted(n for n, node in nodes.items() if node.parent_id is None)
    if len(roots) != 1:
        raise MappingValidationError(
            "映射必须且只能有一个根节点（工厂），当前根节点为：" + (", ".join(roots) or "无")
        )
    if nodes[roots[0]].level != "plant":
        raise MappingValidationError("根节点必须是 plant 层级。")

    by_level: dict[str, list[str]] = {lvl: [] for lvl in KNOWN_LEVELS}
    for node in nodes.values():
        if node.parent_id is not None:
            nodes[node.parent_id].children.append(node.node_id)
    for node in nodes.values():
        node.children.sort()
        if node.level == "line" and node.children:
            raise MappingValidationError(f"产线节点 {node.node_id} 不能有子节点。")
        if node.level == "shop" and not node.children:
            raise MappingValidationError(f"车间节点 {node.node_id} 下没有产线（缺层）。")
        if node.level == "plant" and not node.children:
            raise MappingValidationError(f"工厂节点 {node.node_id} 下没有车间（缺层）。")
    # Present nodes level-by-level in tree order so pages read top-down and
    # siblings appear in a stable, sorted order.
    by_level = {lvl: [] for lvl in KNOWN_LEVELS}

    def walk(node_id: str) -> None:
        node = nodes[node_id]
        by_level[node.level].append(node_id)
        for child in node.children:
            walk(child)

    walk(roots[0])

    # A structurally complete tree has at least one node on every level.
    missing_levels = [lvl for lvl in KNOWN_LEVELS if not by_level[lvl]]
    if missing_levels:
        raise MappingValidationError("映射缺少层级：" + ", ".join(missing_levels))

    # Weights must be defined on every node in a weighted optimisation.  Zero is
    # allowed to pin a node, but the total weight must be positive.
    if sum(node.weight for node in nodes.values()) <= 0:
        raise MappingValidationError("所有节点权重均为 0，无法进行加权校正。")

    return Hierarchy(nodes=nodes, roots=roots, by_level=by_level)
