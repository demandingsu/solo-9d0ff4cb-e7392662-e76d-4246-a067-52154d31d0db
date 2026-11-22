"""Download endpoints must reflect exactly the snapshot that was published."""

from __future__ import annotations

import csv
import io

import pytest

from app.db import Database
from app.reconcile import ReconciliationResult
from app.service import ReconciliationService

from .conftest import FORECAST_CSV, MAPPING_CSV, upload_text


def test_download_after_a_failed_reconciliation_returns_last_snapshot(tmp_path):
    svc = ReconciliationService(Database(tmp_path / "d.db"))
    ds = svc.create_dataset("d").id
    upload_text(svc, ds, "mapping", MAPPING_CSV)
    upload_text(svc, ds, "line_forecast", FORECAST_CSV)
    upload_text(svc, ds, "published", "node_id,level,period,value\nS1,,2026-01,36\n")
    p = svc.preview(ds)
    svc.publish(ds, p["preview_id"])

    # Now break the published file and fail a reconciliation.
    upload_text(
        svc,
        ds,
        "published",
        "node_id,level,period,value\nP1,,2026-01,999\nS1,,2026-01,1\nS2,,2026-01,1\n",
    )
    with pytest.raises(Exception):
        svc.preview(ds)

    snapshot = svc.db.get_snapshot(ds)
    result = ReconciliationResult.from_dict(snapshot["result"])
    period = result.periods[0]
    assert period.corrected["S1"] == pytest.approx(36)
    # And the published CSV body still matches that snapshot.
    from app.service import result_to_csv

    reader = csv.DictReader(io.StringIO(result_to_csv(result)))
    rows = [r for r in reader if r["period"] == "2026-01" and r["node_id"] == "S1"]
    assert float(rows[0]["corrected"]) == pytest.approx(36)
