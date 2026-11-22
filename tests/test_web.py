"""HTTP-level tests covering the pages and the JSON API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture()
def client(tmp_path):
    app = create_app(str(tmp_path / "web.db"))
    with TestClient(app) as c:
        yield c


MAPPING = """node_id,parent_id,level,name,weight
P1,,plant,工厂,1
S1,P1,shop,车间A,1
S2,P1,shop,车间B,1
L1,S1,line,产线1,1
L2,S1,line,产线2,1
L3,S2,line,产线3,1
"""
FORECAST = """node_id,period,value
L1,2026-01,10
L2,2026-01,20
L3,2026-01,30
"""
PUBLISHED = """node_id,level,period,value
,shop,2026-01,66
"""


def make_dataset(client) -> int:
    resp = client.post("/api/datasets", json={"name": "Web 测试"})
    assert resp.status_code == 200
    return resp.json()["id"]


def upload(client, ds, kind, text):
    return client.post(
        f"/api/datasets/{ds}/upload",
        params={"kind": kind},
        files={"file": (f"{kind}.csv", text.encode(), "text/csv")},
    )


def ready(client) -> int:
    ds = make_dataset(client)
    assert upload(client, ds, "mapping", MAPPING).status_code == 200
    assert upload(client, ds, "line_forecast", FORECAST).status_code == 200
    assert upload(client, ds, "published", PUBLISHED).status_code == 200
    return ds


def test_index_and_dataset_page(client):
    ds = ready(client)
    assert client.get("/").status_code == 200
    page = client.get(f"/datasets/{ds}")
    assert page.status_code == 200
    assert "Web 测试" in page.text
    assert "上传" in page.text


def test_preview_publish_and_snapshot_page(client):
    ds = ready(client)
    preview = client.post(f"/api/datasets/{ds}/preview").json()
    period = preview["result"]["periods"][0]
    # A `level` row publishes the total across that level, so the two shops
    # must sum to 66 while the plant follows.
    assert period["corrected"]["S1"] + period["corrected"]["S2"] == pytest.approx(66)
    assert period["corrected"]["P1"] == pytest.approx(66)
    assert period["conflicts"]

    pub = client.post(f"/api/datasets/{ds}/publish", json={"preview_id": preview["preview_id"]})
    assert pub.status_code == 200
    version = pub.json()["version_number"]

    page = client.get(f"/datasets/{ds}/snapshots/{version}")
    assert page.status_code == 200
    assert "调整来源" in page.text

    listing = client.get(f"/api/datasets/{ds}/snapshots").json()
    assert listing[0]["version_number"] == version


def test_old_preview_cannot_be_published_after_new_upload(client):
    ds = ready(client)
    preview = client.post(f"/api/datasets/{ds}/preview").json()
    upload(client, ds, "published", PUBLISHED.replace("66", "70"))
    resp = client.post(f"/api/datasets/{ds}/publish", json={"preview_id": preview["preview_id"]})
    assert resp.status_code == 400
    assert resp.json()["code"] == "stale_preview_error"
    # UI shows the stale banner rather than a publish button.
    page = client.get(f"/datasets/{ds}")
    assert "预览已过期" in page.text


def test_invalid_mapping_upload_returns_error_and_keeps_page(client):
    ds = make_dataset(client)
    cyclic = "node_id,parent_id,level,name,weight\nP1,,plant,工厂,1\nS1,L1,shop,车间,1\nL1,S1,line,产线,1\n"
    resp = upload(client, ds, "mapping", cyclic)
    assert resp.status_code == 400
    assert resp.json()["code"] == "mapping_validation_error"
    form = client.post(
        f"/datasets/{ds}/upload",
        data={"kind": "mapping"},
        files={"file": ("m.csv", cyclic.encode(), "text/csv")},
        follow_redirects=False,
    )
    assert form.status_code == 400
    assert "成环" in form.text


def test_failed_preview_shows_error_page_and_keeps_snapshot(client):
    ds = ready(client)
    preview = client.post(f"/api/datasets/{ds}/preview").json()
    client.post(f"/api/datasets/{ds}/publish", json={"preview_id": preview["preview_id"]})

    upload(client, ds, "published", "node_id,level,period,value\nP1,,2026-01,99\nS1,,2026-01,10\nS2,,2026-01,30\n")
    resp = client.post(f"/api/datasets/{ds}/preview")
    assert resp.status_code == 400
    snapshots = client.get(f"/api/datasets/{ds}/snapshots").json()
    assert len(snapshots) == 1


def test_downloads(client):
    ds = ready(client)
    preview = client.post(f"/api/datasets/{ds}/preview").json()
    pid = preview["preview_id"]
    for name in ("snapshot.csv", "adjustments.csv", "conflicts.csv"):
        resp = client.get(f"/datasets/{ds}/download/{name}?preview={pid}")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/csv")
        assert "attachment" in resp.headers["content-disposition"]
        assert "period" in resp.text.splitlines()[0]


def test_download_requires_version_or_preview(client):
    ds = ready(client)
    assert client.get(f"/datasets/{ds}/download/snapshot.csv").status_code == 400


def test_health(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_upload_to_missing_dataset_renders_error_page(client):
    resp = client.post(
        "/datasets/12345/upload",
        data={"kind": "mapping"},
        files={"file": ("m.csv", b"node_id,parent_id\n", "text/csv")},
    )
    assert resp.status_code == 400
    assert "操作失败" in resp.text
