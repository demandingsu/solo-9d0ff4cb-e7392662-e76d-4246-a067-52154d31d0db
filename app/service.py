"""Application service layer: uploads, previews, publishing and versioning.

Versioning model
----------------
* Each dataset has an ``input_version`` counter.  Uploading a *different*
  mapping, forecast or published file increments it.
* Every preview records the input version and the content hashes of the
  revisions it was built from.  Publishing re-checks all of those, so a preview
  built on older inputs can never be published ("旧预览不得发布").
* Failed reconciliations never create a preview or snapshot, and never modify
  the previous published snapshot ("校正失败时保留上一版结果").
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from typing import Any

from . import db as db_module
from .csv_io import parse_forecast, parse_mapping, parse_published
from .errors import (
    CsvParseError,
    PublishError,
    ReconciliationError,
    SnapshotValidationError,
    StalePreviewError,
)
from .hierarchy import build_hierarchy
from .reconcile import (
    ReconciliationResult,
    reconcile,
    verify_consistency,
    verify_hierarchy,
)

KIND_LABELS = {
    "mapping": "层级映射",
    "line_forecast": "产线预测",
    "published": "已发布车间/工厂预测",
}

# Default cap on the line count for periods that need the dense general
# solver.  ``RECONCILE_MAX_LINES`` (see docker-compose.yml) overrides it.
DEFAULT_MAX_LINES = 500


def payload_hash(payload: Any) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class UploadOutcome:
    kind: str
    changed: bool
    input_version: int
    row_count: int
    filename: str


class ReconciliationService:
    def __init__(self, database: db_module.Database, max_lines: int | None = None) -> None:
        self.db = database
        self.max_lines = DEFAULT_MAX_LINES if max_lines is None else max(1, int(max_lines))

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    def create_dataset(self, name: str) -> db_module.Dataset:
        return self.db.create_dataset(name.strip() or "未命名数据集")

    def dataset_state(self, dataset_id: int) -> dict[str, Any]:
        dataset = self.db.get_dataset(dataset_id)
        if dataset is None:
            raise PublishError("数据集不存在。")
        revisions = self.db.latest_revisions(dataset_id)
        snapshots = self.db.list_snapshots(dataset_id)
        preview = self.db.latest_preview(dataset_id)
        current_hashes = {k: v["content_hash"] for k, v in revisions.items()}
        preview_state = None
        if preview is not None:
            preview_state = {
                "id": preview["id"],
                "created_at": preview["created_at"],
                "input_version": preview["input_version"],
                "superseded": preview["superseded"],
                "stale": self._preview_is_stale(preview, current_hashes, dataset.input_version),
                "result": preview["result"],
            }
        return {
            "dataset": {
                "id": dataset.id,
                "name": dataset.name,
                "created_at": dataset.created_at,
                "input_version": dataset.input_version,
            },
            "revisions": revisions,
            "snapshots": snapshots,
            "latest_snapshot": self.db.get_snapshot(dataset_id),
            "preview": preview_state,
            "events": self.db.list_events(dataset_id),
        }

    @staticmethod
    def _preview_is_stale(
        preview: dict[str, Any], current_hashes: dict[str, str], input_version: int
    ) -> bool:
        if preview.get("superseded"):
            return True
        if preview.get("input_version") != input_version:
            return True
        request_hashes = (preview.get("request") or {}).get("content_hashes", {})
        return request_hashes != current_hashes

    # ------------------------------------------------------------------
    # Uploads
    # ------------------------------------------------------------------
    def upload(
        self,
        dataset_id: int,
        kind: str,
        filename: str,
        data: bytes,
    ) -> UploadOutcome:
        if kind not in KIND_LABELS:
            raise PublishError(f"未知的上传类型 {kind!r}。")

        # Parsing and validation happen before touching the database so a bad
        # file leaves the previous revision (and published result) intact.
        try:
            payload = self._parse_upload(kind, filename, data)
            if kind == "mapping":
                build_hierarchy_from_payload(payload)  # validate eagerly
        except ReconciliationError as exc:
            self._log(dataset_id, "upload_rejected", f"{KIND_LABELS[kind]}上传被拒绝：{exc}")
            raise

        content_hash = payload_hash(payload)

        with self.db.connect() as conn:
            dataset = self.db.get_dataset(dataset_id, conn)
            if dataset is None:
                raise PublishError("数据集不存在。")
            previous = self.db.latest_revisions(dataset_id, conn).get(kind)
            changed = previous is None or previous["content_hash"] != content_hash
            if not changed:
                self.db.log_event(
                    conn,
                    dataset_id,
                    "upload_unchanged",
                    f"{KIND_LABELS[kind]}内容与当前版本相同，未产生新版本。",
                    {"filename": filename},
                )
                return UploadOutcome(
                    kind, False, dataset.input_version, payload["row_count"], filename
                )
            new_version = self.db.bump_input_version(conn, dataset_id)
            self.db.add_revision(
                conn, dataset_id, new_version, kind, filename, content_hash, payload
            )
            self.db.log_event(
                conn,
                dataset_id,
                "upload",
                f"上传{KIND_LABELS[kind]} {filename}，输入版本更新为 v{new_version}。",
                {"kind": kind, "rows": payload["row_count"]},
            )
        return UploadOutcome(kind, True, new_version, payload["row_count"], filename)

    @staticmethod
    def _parse_upload(kind: str, filename: str, data: bytes) -> dict[str, Any]:
        if not data:
            raise CsvParseError(f"文件 {filename} 为空。")
        if kind == "mapping":
            rows = parse_mapping(data, filename)
            return {
                "rows": [
                    {
                        "node_id": r.node_id,
                        "parent_id": r.parent_id,
                        "level": r.level,
                        "name": r.name,
                        "weight": r.weight if r.weight is not None else 1.0,
                    }
                    for r in rows
                ],
                "row_count": len(rows),
            }
        if kind == "line_forecast":
            rows = parse_forecast(data, filename)
            return {
                "rows": [
                    {"node_id": r.node_id, "period": r.period, "value": r.value} for r in rows
                ],
                "row_count": len(rows),
            }
        rows = parse_published(data, filename)
        return {
            "rows": [
                {"target": r.target, "kind": r.kind, "period": r.period, "value": r.value}
                for r in rows
            ],
            "row_count": len(rows),
        }

    # ------------------------------------------------------------------
    # Preview (reconciliation)
    # ------------------------------------------------------------------
    def preview(self, dataset_id: int) -> dict[str, Any]:
        # The whole read -> compute -> store sequence runs inside one write
        # transaction.  An upload racing with this call therefore either lands
        # before (and is included) or after (and marks the preview stale), never
        # in between, so a preview can never claim an input version it did not
        # actually use.
        try:
            with self.db.connect() as conn:
                dataset = self.db.get_dataset(dataset_id, conn)
                if dataset is None:
                    raise PublishError("数据集不存在。")
                revisions = self.db.latest_revisions(dataset_id, conn)
                missing = [
                    KIND_LABELS[k]
                    for k in ("mapping", "line_forecast")
                    if k not in revisions
                ]
                if missing:
                    raise SnapshotValidationError(
                        "缺少必要输入：" + "、".join(missing) + "。"
                    )

                mapping_payload = revisions["mapping"]["payload"]
                forecast_payload = revisions["line_forecast"]["payload"]
                published_payload = revisions.get("published", {}).get(
                    "payload", {"rows": [], "row_count": 0}
                )
                content_hashes = {k: v["content_hash"] for k, v in revisions.items()}

                hierarchy = build_hierarchy_from_payload(mapping_payload)
                result = reconcile(
                    hierarchy,
                    forecast_payload["rows"],
                    published_payload["rows"],
                    max_lines=self.max_lines,
                )
                problems = verify_consistency(result)
                if problems:
                    raise SnapshotValidationError(
                        "校正结果未通过一致性校验：" + "; ".join(problems[:5])
                    )

                request = {
                    "content_hashes": content_hashes,
                    "revisions": {k: v["id"] for k, v in revisions.items()},
                }
                result_dict = result.to_dict()
                preview_id = self.db.create_preview(
                    conn, dataset_id, dataset.input_version, request, result_dict
                )
                self.db.log_event(
                    conn,
                    dataset_id,
                    "preview",
                    f"生成校正预览 #{preview_id}（输入版本 v{dataset.input_version}）。",
                    {"periods": len(result.periods)},
                )
                return {"preview_id": preview_id, "result": result_dict}
        except ReconciliationError as exc:
            # Failure: nothing is stored; the last published snapshot survives.
            # `connect()` already rolled the transaction back.
            detail = {"error_code": exc.code}
            extra = getattr(exc, "detail", None)
            if isinstance(extra, dict):
                detail.update(extra)
            self._log(
                dataset_id,
                "preview_failed",
                f"校正失败：{exc}",
                detail,
            )
            raise

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------
    def publish(self, dataset_id: int, preview_id: int) -> dict[str, Any]:
        try:
            with self.db.connect() as conn:
                dataset = self.db.get_dataset(dataset_id, conn)
                if dataset is None:
                    raise PublishError("数据集不存在。")
                preview = self._get_preview(dataset_id, preview_id, conn)
                revisions = self.db.latest_revisions(dataset_id, conn)
                current_hashes = {k: v["content_hash"] for k, v in revisions.items()}

                if preview["superseded"]:
                    raise StalePreviewError("该预览已被新的输入版本取代，不能发布。")
                if preview["input_version"] != dataset.input_version:
                    raise StalePreviewError(
                        f"该预览基于输入版本 v{preview['input_version']}，"
                        f"当前为 v{dataset.input_version}，请重新校正。"
                    )
                request_hashes = (preview["request"] or {}).get("content_hashes", {})
                if request_hashes != current_hashes:
                    raise StalePreviewError(
                        "输入文件已变更，旧预览不得发布，请重新校正。"
                    )

                # Re-hydrate and re-verify the stored result before publishing:
                # this catches corrupted data and guarantees the hierarchy
                # constraints hold even if the preview was created by an older
                # build.
                result = ReconciliationResult.from_dict(preview["result"])
                problems = verify_hierarchy(result.hierarchy) + verify_consistency(result)
                if problems:
                    raise SnapshotValidationError(
                        "预览结果不满足层级一致性，拒绝发布：" + "; ".join(problems[:5])
                    )
                if not result.periods:
                    raise SnapshotValidationError("预览不包含任何时段，拒绝发布。")

                published = self.db.publish(
                    conn,
                    dataset_id,
                    preview["input_version"],
                    preview["result"],
                )
                self.db.log_event(
                    conn,
                    dataset_id,
                    "publish",
                    f"发布快照 v{published['version_number']}（输入版本 v{preview['input_version']}）。",
                    {"preview_id": preview_id},
                )
        except ReconciliationError as exc:
            self._log(
                dataset_id,
                "publish_rejected",
                f"发布被拒绝（预览 #{preview_id}）：{exc}",
                {"error_code": exc.code},
            )
            raise
        return {
            "snapshot_id": published["id"],
            "version_number": published["version_number"],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_preview(self, dataset_id: int, preview_id: int, conn: Any) -> dict[str, Any]:
        row = conn.execute(
            "SELECT * FROM previews WHERE id=? AND dataset_id=?", (preview_id, dataset_id)
        ).fetchone()
        if row is None:
            raise PublishError("预览不存在。")
        return {
            "id": row["id"],
            "dataset_id": row["dataset_id"],
            "input_version": row["input_version"],
            "created_at": row["created_at"],
            "superseded": bool(row["superseded"]),
            "request": json.loads(row["request_json"]),
            "result": json.loads(row["result_json"]),
        }

    def _log(self, dataset_id: int, kind: str, message: str, detail: Any = None) -> None:
        try:
            with self.db.connect() as conn:
                self.db.log_event(conn, dataset_id, kind, message, detail)
        except Exception:  # pragma: no cover - logging must never mask the error
            pass


def build_hierarchy_from_payload(payload: dict[str, Any]):
    from .csv_io import MappingRow

    rows = [
        MappingRow(
            node_id=r["node_id"],
            parent_id=r.get("parent_id"),
            level=r.get("level"),
            name=r.get("name"),
            weight=r.get("weight"),
        )
        for r in payload["rows"]
    ]
    return build_hierarchy(rows)


def result_to_csv(result: ReconciliationResult) -> str:
    """Long-format CSV with base, corrected and delta for every node."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        ["period", "node_id", "level", "node_name", "parent_id", "base", "corrected", "delta"]
    )
    for period in result.periods:
        for node_id in sorted(result.hierarchy.nodes):
            node = result.hierarchy.nodes[node_id]
            writer.writerow(
                [
                    period.period,
                    node_id,
                    node.level,
                    node.name,
                    node.parent_id or "",
                    f"{period.base[node_id]:.6f}",
                    f"{period.corrected[node_id]:.6f}",
                    f"{period.deltas.get(node_id, 0.0):.6f}",
                ]
            )
    return buffer.getvalue()


def adjustments_to_csv(result: ReconciliationResult) -> str:
    """Per-period adjustment provenance, one row per adjusted node."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "period",
            "node_id",
            "level",
            "node_name",
            "base",
            "corrected",
            "delta",
            "source",
            "source_nodes",
            "path",
        ]
    )
    for period in result.periods:
        for adj in period.node_adjustments:
            writer.writerow(
                [
                    period.period,
                    adj["node_id"],
                    adj["level"],
                    adj["name"],
                    f"{adj['base']:.6f}",
                    f"{adj['corrected']:.6f}",
                    f"{adj['delta']:.6f}",
                    adj["source"],
                    ";".join(adj.get("source_nodes", []))
                    if adj.get("source_nodes")
                    else adj["source"],
                    " > ".join(adj["path"]),
                ]
            )
    return buffer.getvalue()


def conflicts_to_csv(result: ReconciliationResult) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["period", "scope", "level", "base_total", "published", "difference"])
    for period in result.periods:
        for conflict in period.conflicts:
            writer.writerow(
                [
                    period.period,
                    conflict["scope"],
                    conflict["level"],
                    f"{conflict['base']:.6f}",
                    f"{conflict['published']:.6f}",
                    f"{conflict['diff']:.6f}",
                ]
            )
    return buffer.getvalue()
