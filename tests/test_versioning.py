"""Data-versioning guarantees: stale previews, failed runs, immutability."""

from __future__ import annotations

import pytest

from app.errors import (
    CsvParseError,
    MappingValidationError,
    SnapshotValidationError,
    StalePreviewError,
)

from .conftest import FORECAST_CSV, MAPPING_CSV, upload_text

PUBLISHED_CSV = """node_id,level,period,value
,shop,2026-01,66
,shop,2026-02,60
"""


def ready(service, dataset_id, published=PUBLISHED_CSV):
    upload_text(service, dataset_id, "mapping", MAPPING_CSV)
    upload_text(service, dataset_id, "line_forecast", FORECAST_CSV)
    upload_text(service, dataset_id, "published", published)


def test_upload_changes_input_version(service, dataset_id):
    before = service.db.get_dataset(dataset_id).input_version
    outcome = upload_text(service, dataset_id, "line_forecast", FORECAST_CSV)
    assert outcome.changed
    assert service.db.get_dataset(dataset_id).input_version == before + 1


def test_reuploading_identical_file_does_not_bump_version(service, dataset_id):
    upload_text(service, dataset_id, "mapping", MAPPING_CSV)
    version = service.db.get_dataset(dataset_id).input_version
    outcome = upload_text(service, dataset_id, "mapping", MAPPING_CSV)
    assert outcome.changed is False
    assert service.db.get_dataset(dataset_id).input_version == version


def test_preview_then_upload_invalidates_old_preview(service, dataset_id):
    ready(service, dataset_id)
    preview = service.preview(dataset_id)
    state = service.dataset_state(dataset_id)
    assert state["preview"]["stale"] is False

    # A new forecast file changes the result -> old preview must not publish.
    upload_text(
        service,
        dataset_id,
        "line_forecast",
        FORECAST_CSV.replace("L1,2026-01,10", "L1,2026-01,11"),
    )
    state = service.dataset_state(dataset_id)
    assert state["preview"]["stale"] is True
    with pytest.raises(StalePreviewError):
        service.publish(dataset_id, preview["preview_id"])
    # Nothing was published.
    assert service.db.list_snapshots(dataset_id) == []


def test_upload_of_irrelevant_kind_still_invalidates_preview(service, dataset_id):
    # Even a mapping re-upload that happens to be identical keeps the preview
    # valid, but a *changed* published file must invalidate it.
    ready(service, dataset_id)
    preview = service.preview(dataset_id)
    upload_text(
        service,
        dataset_id,
        "published",
        PUBLISHED_CSV.replace("2026-02,60", "2026-02,61"),
    )
    with pytest.raises(StalePreviewError):
        service.publish(dataset_id, preview["preview_id"])


def test_failed_reconciliation_keeps_previous_snapshot(service, dataset_id):
    ready(service, dataset_id)
    preview = service.preview(dataset_id)
    version = service.publish(dataset_id, preview["preview_id"])["version_number"]

    # Now upload published values that contradict each other.
    bad = """node_id,level,period,value
P1,,2026-01,99
S1,,2026-01,10
S2,,2026-01,30
"""
    upload_text(service, dataset_id, "published", bad)
    with pytest.raises(SnapshotValidationError):
        service.preview(dataset_id)

    snapshots = service.db.list_snapshots(dataset_id)
    assert len(snapshots) == 1
    assert snapshots[0]["version_number"] == version
    latest = service.db.get_snapshot(dataset_id)
    assert latest["version_number"] == version

    events = service.db.list_events(dataset_id)
    assert any(e["kind"] == "preview_failed" for e in events)


def test_failed_upload_leaves_previous_revision(service, dataset_id):
    ready(service, dataset_id)
    revisions_before = service.db.latest_revisions(dataset_id)
    version_before = service.db.get_dataset(dataset_id).input_version

    cyclic = """node_id,parent_id,level,name,weight
P1,,plant,工厂,1
S1,L1,shop,车间,1
L1,S1,line,产线,1
"""
    with pytest.raises(MappingValidationError):
        upload_text(service, dataset_id, "mapping", cyclic)

    revisions_after = service.db.latest_revisions(dataset_id)
    assert revisions_after["mapping"]["content_hash"] == revisions_before["mapping"]["content_hash"]
    assert service.db.get_dataset(dataset_id).input_version == version_before


def test_bad_csv_does_not_change_state(service, dataset_id):
    ready(service, dataset_id)
    before = service.db.get_dataset(dataset_id).input_version
    with pytest.raises(CsvParseError):
        upload_text(service, dataset_id, "line_forecast", "node_id,period\nL1,2026-01\n")
    assert service.db.get_dataset(dataset_id).input_version == before


def test_published_snapshot_is_immutable(service, dataset_id):
    ready(service, dataset_id)
    p1 = service.preview(dataset_id)
    v1 = service.publish(dataset_id, p1["preview_id"])
    original = service.db.get_snapshot(dataset_id, v1["version_number"])

    upload_text(service, dataset_id, "published", PUBLISHED_CSV.replace("66", "70"))
    p2 = service.preview(dataset_id)
    v2 = service.publish(dataset_id, p2["preview_id"])
    assert v2["version_number"] == v1["version_number"] + 1

    again = service.db.get_snapshot(dataset_id, v1["version_number"])
    assert again["result"] == original["result"]
    # Parent link records lineage.
    assert service.db.get_snapshot(dataset_id, v2["version_number"])["parent_id"] == original["id"]


def test_cannot_publish_same_preview_twice_as_stale(service, dataset_id):
    ready(service, dataset_id)
    preview = service.preview(dataset_id)
    service.publish(dataset_id, preview["preview_id"])
    # A second publish of the same preview is still based on the current inputs,
    # so it is allowed and creates a new version (explicit user action).
    second = service.publish(dataset_id, preview["preview_id"])
    assert second["version_number"] == 2


def test_publish_unknown_preview_rejected(service, dataset_id):
    ready(service, dataset_id)
    with pytest.raises(Exception):
        service.publish(dataset_id, 9999)


def test_download_csv_contains_all_nodes(service, dataset_id):
    from app.service import adjustments_to_csv, conflicts_to_csv, result_to_csv
    from app.reconcile import ReconciliationResult

    ready(service, dataset_id)
    preview = service.preview(dataset_id)
    service.publish(dataset_id, preview["preview_id"])
    snapshot = service.db.get_snapshot(dataset_id)
    result = ReconciliationResult.from_dict(snapshot["result"])

    snapshot_csv = result_to_csv(result)
    lines = snapshot_csv.strip().splitlines()
    # header + 6 nodes x 2 periods
    assert len(lines) == 1 + 6 * 2
    assert "period,node_id,level,node_name,parent_id,base,corrected,delta" in lines[0]

    adj = adjustments_to_csv(result)
    assert "source" in adj.splitlines()[0]
    conflicts = conflicts_to_csv(result)
    assert "difference" in conflicts.splitlines()[0]
