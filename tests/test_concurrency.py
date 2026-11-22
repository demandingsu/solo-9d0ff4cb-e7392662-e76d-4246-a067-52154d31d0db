"""Concurrency and race-condition tests for the versioning guarantees."""

from __future__ import annotations

import threading

import pytest

from app.db import Database
from app.errors import StalePreviewError
from app.service import ReconciliationService

from .conftest import FORECAST_CSV, MAPPING_CSV, upload_text

PUBLISHED_CSV = """node_id,level,period,value
S1,,2026-01,66
"""


def _ready(service, ds):
    upload_text(service, ds, "mapping", MAPPING_CSV)
    upload_text(service, ds, "line_forecast", FORECAST_CSV)
    upload_text(service, ds, "published", PUBLISHED_CSV)


def test_preview_records_the_version_it_used(tmp_path):
    path = tmp_path / "race.db"
    db = Database(path)
    service = ReconciliationService(db)
    ds = service.create_dataset("race").id
    _ready(service, ds)

    preview = service.preview(ds)
    state = service.dataset_state(ds)
    recorded = state["preview"]
    assert recorded["input_version"] == db.get_dataset(ds).input_version
    # The hashes stored with the preview match the revisions actually used.
    stored = recorded["result"]
    base = next(p for p in stored["periods"] if p["period"] == "2026-01")["base"]
    assert base["L1"] == 10

    upload_text(
        service,
        ds,
        "line_forecast",
        FORECAST_CSV.replace("L1,2026-01,10", "L1,2026-01,99"),
    )
    # Publishing the preview from before the upload must fail.
    try:
        service.publish(ds, preview["preview_id"])
    except Exception as exc:
        assert type(exc).__name__ == "StalePreviewError"
    else:  # pragma: no cover
        raise AssertionError("stale preview was published")


def test_concurrent_preview_and_upload_never_mix_versions(tmp_path):
    path = tmp_path / "race2.db"
    db = Database(path)
    ds = ReconciliationService(db).create_dataset("race2").id
    setup = ReconciliationService(db)
    _ready(setup, ds)

    errors: list[BaseException] = []
    previews: list[dict] = []
    barrier = threading.Barrier(2)

    def do_preview():
        svc = ReconciliationService(db)
        try:
            barrier.wait(timeout=10)
            previews.append(svc.preview(ds))
        except BaseException as exc:  # pragma: no cover - recorded below
            errors.append(exc)

    def do_upload():
        svc = ReconciliationService(db)
        try:
            barrier.wait(timeout=10)
            upload_text(
                svc,
                ds,
                "published",
                PUBLISHED_CSV.replace("66", "80"),
            )
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=do_preview), threading.Thread(target=do_upload)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    assert not errors
    assert len(previews) == 1
    # Whatever interleaving occurred, the preview reflects exactly one input
    # version (never a mix), and publishing only succeeds if it is still the
    # current one.
    svc = ReconciliationService(db)
    preview = previews[0]
    current_hash = svc.db.latest_revisions(ds)["published"]["content_hash"]
    stored = db.get_preview(preview["preview_id"])
    stored_hash = stored["request"]["content_hashes"]["published"]
    still_current = current_hash == stored_hash
    if still_current:
        svc.publish(ds, preview["preview_id"])
    else:
        with pytest.raises(StalePreviewError):
            svc.publish(ds, preview["preview_id"])
    # The snapshot, if any, matches the preview it came from.
    if still_current:
        snapshot = db.get_snapshot(ds)
        assert snapshot["result"] == stored["result"]


def test_sequential_snapshots_keep_lineage(tmp_path):
    db = Database(tmp_path / "chain.db")
    svc = ReconciliationService(db)
    ds = svc.create_dataset("chain").id
    _ready(svc, ds)
    p1 = svc.preview(ds)
    v1 = svc.publish(ds, p1["preview_id"])
    p2 = svc.preview(ds)
    v2 = svc.publish(ds, p2["preview_id"])
    snap2 = db.get_snapshot(ds, v2["version_number"])
    assert snap2["parent_id"] == db.get_snapshot(ds, v1["version_number"])["id"]
