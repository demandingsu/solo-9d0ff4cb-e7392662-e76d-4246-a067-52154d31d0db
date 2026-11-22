"""CSV parsing helpers.

All uploaded files are handled as untrusted text.  We accept UTF-8 (with or
without BOM) and sniff the delimiter.  The parser is strict about headers so a
mis-labelled file fails loudly instead of silently producing an empty plan.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

from .errors import CsvParseError


@dataclass
class MappingRow:
    node_id: str
    parent_id: str | None
    level: str | None
    name: str | None
    weight: float | None


@dataclass
class ForecastRow:
    node_id: str
    period: str
    value: float


@dataclass
class PublishedRow:
    target: str
    kind: str  # "node" or "level"
    period: str
    value: float


def _read_text(data: bytes, filename: str) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise CsvParseError(f"无法解码文件 {filename}，请使用 UTF-8 或 GB18030 编码的 CSV。")


def _rows(data: bytes, filename: str) -> tuple[list[str], list[dict[str, str]]]:
    text = _read_text(data, filename)
    if not text.strip():
        raise CsvParseError(f"文件 {filename} 为空。")
    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if reader.fieldnames is None:
        raise CsvParseError(f"文件 {filename} 缺少表头。")
    headers = [(h or "").strip().lower() for h in reader.fieldnames]
    reader.fieldnames = headers
    rows = []
    for lineno, raw in enumerate(reader, start=2):
        if raw is None:
            continue
        if not any((v or "").strip() for v in raw.values()):
            continue
        rows.append({(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()})
    if not rows:
        raise CsvParseError(f"文件 {filename} 没有数据行。")
    return headers, rows


def _require(headers: list[str], required: set[str], filename: str) -> None:
    missing = required - set(headers)
    if missing:
        raise CsvParseError(
            f"文件 {filename} 缺少列：{', '.join(sorted(missing))}；实际列为 {', '.join(headers)}。"
        )


def _parse_float(value: str, what: str, lineno: int) -> float:
    try:
        num = float(value.replace(",", ""))
    except (TypeError, ValueError):
        raise CsvParseError(f"第 {lineno} 行的 {what} 不是数字：{value!r}") from None
    if num != num or num in (float("inf"), float("-inf")):
        raise CsvParseError(f"第 {lineno} 行的 {what} 不是有限数值：{value!r}")
    return num


def parse_mapping(data: bytes, filename: str) -> list[MappingRow]:
    headers, rows = _rows(data, filename)
    _require(headers, {"node_id"}, filename)
    result: list[MappingRow] = []
    for idx, row in enumerate(rows, start=2):
        node_id = row.get("node_id", "")
        if not node_id:
            raise CsvParseError(f"第 {idx} 行缺少 node_id。")
        parent = row.get("parent_id") or None
        weight_raw = row.get("weight") or None
        weight = _parse_float(weight_raw, "weight", idx) if weight_raw is not None else None
        if weight is not None and weight < 0:
            raise CsvParseError(f"第 {idx} 行的 weight 不能为负数。")
        level = (row.get("level") or "").strip() or None
        name = (row.get("name") or "").strip() or None
        result.append(MappingRow(node_id, parent, level, name, weight))
    return result


def parse_forecast(data: bytes, filename: str) -> list[ForecastRow]:
    headers, rows = _rows(data, filename)
    _require(headers, {"node_id", "period", "value"}, filename)
    result = []
    for idx, row in enumerate(rows, start=2):
        node_id = row.get("node_id", "")
        period = row.get("period", "")
        if not node_id or not period:
            raise CsvParseError(f"第 {idx} 行缺少 node_id 或 period。")
        result.append(ForecastRow(node_id, period, _parse_float(row.get("value", ""), "value", idx)))
    return result


def parse_published(data: bytes, filename: str) -> list[PublishedRow]:
    """Published shop/plant forecasts.

    Accepts either ``node_id`` for a specific node or ``level`` to mark an
    aggregate row that is matched against the implied total of all nodes on a
    level.  Exactly one of the two must be present per row.
    """
    headers, rows = _rows(data, filename)
    _require(headers, {"period", "value"}, filename)
    if "node_id" not in headers and "level" not in headers:
        raise CsvParseError(
            f"文件 {filename} 需要 node_id 或 level 列之一，用于定位已发布的车间/工厂预测。"
        )
    result: list[PublishedRow] = []
    for idx, row in enumerate(rows, start=2):
        node_id = row.get("node_id") or ""
        level = row.get("level") or ""
        period = row.get("period", "")
        if not period:
            raise CsvParseError(f"第 {idx} 行缺少 period。")
        if bool(node_id) == bool(level):
            raise CsvParseError(
                f"第 {idx} 行必须且只能提供 node_id 或 level 之一（当前 node_id={node_id!r}, level={level!r}）。"
            )
        target = node_id or level
        result.append(
            PublishedRow(
                target=target,
                kind="node" if node_id else "level",
                period=period,
                value=_parse_float(row.get("value", ""), "value", idx),
            )
        )
    return result
