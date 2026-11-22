"""Shared pytest fixtures and helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db import Database  # noqa: E402
from app.service import ReconciliationService  # noqa: E402

MAPPING_CSV = """node_id,parent_id,level,name,weight
P1,,plant,华东工厂,1
S1,P1,shop,冲压车间,1
S2,P1,shop,焊装车间,1
L1,S1,line,冲压一线,1
L2,S1,line,冲压二线,1
L3,S2,line,焊装一线,1
"""

FORECAST_CSV = """node_id,period,value
L1,2026-01,10
L2,2026-01,20
L3,2026-01,30
L1,2026-02,12
L2,2026-02,22
L3,2026-02,33
"""


@pytest.fixture()
def service(tmp_path: Path) -> ReconciliationService:
    return ReconciliationService(Database(tmp_path / "test.db"))


@pytest.fixture()
def dataset_id(service: ReconciliationService) -> int:
    return service.create_dataset("测试数据集").id


def upload_text(
    service: ReconciliationService,
    dataset_id: int,
    kind: str,
    csv_text: str,
    filename: str | None = None,
):
    return service.upload(
        dataset_id, kind, filename or f"{kind}.csv", csv_text.encode("utf-8")
    )
