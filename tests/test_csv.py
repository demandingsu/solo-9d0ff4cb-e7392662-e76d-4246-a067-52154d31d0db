"""CSV robustness and input-normalisation tests."""

from __future__ import annotations

import pytest

from app.csv_io import parse_forecast, parse_mapping, parse_published
from app.errors import CsvParseError


def test_bom_and_semicolon_delimiter():
    data = "﻿node_id;parent_id;level;weight\nP1;;plant;1\nS1;P1;shop;2\nL1;S1;line;3\n"
    rows = parse_mapping(data.encode("utf-8"), "m.csv")
    assert [(r.node_id, r.parent_id, r.weight) for r in rows] == [
        ("P1", None, 1.0),
        ("S1", "P1", 2.0),
        ("L1", "S1", 3.0),
    ]


def test_missing_column_reports_name():
    with pytest.raises(CsvParseError, match="缺少列"):
        parse_forecast(b"node_id,period\nL1,2026-01\n", "f.csv")


def test_non_numeric_value_reported_with_line():
    with pytest.raises(CsvParseError, match="第 3 行"):
        parse_forecast(b"node_id,period,value\nL1,2026-01,1\nL2,2026-01,abc\n", "f.csv")


def test_empty_file_rejected():
    with pytest.raises(CsvParseError, match="为空"):
        parse_forecast(b"", "f.csv")


def test_negative_weight_rejected():
    with pytest.raises(CsvParseError, match="负数"):
        parse_mapping(b"node_id,parent_id,weight\nL1,S1,-1\n", "m.csv")


def test_published_requires_node_or_level():
    with pytest.raises(CsvParseError, match="node_id 或 level"):
        parse_published(b"period,value\n2026-01,10\n", "p.csv")


def test_published_rejects_both_node_and_level():
    with pytest.raises(CsvParseError, match="只能"):
        parse_published(b"node_id,level,period,value\nS1,shop,2026-01,10\n", "p.csv")


def test_published_kind_detection():
    rows = parse_published(b"node_id,level,period,value\nS1,,2026-01,10\n,shop,2026-01,99\n", "p.csv")
    assert rows[0].kind == "node" and rows[0].target == "S1"
    assert rows[1].kind == "level" and rows[1].target == "shop"


def test_gb18030_encoding_supported():
    text = "node_id,period,value\n产线1,2026-01,10\n"
    rows = parse_forecast(text.encode("gb18030"), "f.csv")
    assert rows[0].node_id == "产线1"


def test_blank_lines_ignored():
    rows = parse_forecast(b"node_id,period,value\nL1,2026-01,10\n\n,\n", "f.csv")
    assert len(rows) == 1
