"""FastAPI application and HTTP routes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .db import Database
from .errors import ReconciliationError
from .service import (
    KIND_LABELS,
    ReconciliationService,
    adjustments_to_csv,
    conflicts_to_csv,
    result_to_csv,
)
from .reconcile import ReconciliationResult

BASE_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))


def create_app(db_path: str | None = None) -> FastAPI:
    app = FastAPI(title="制造计划时序数据校正", version="1.0.0")
    database = Database(db_path or os.environ.get("DATABASE_PATH", "/data/app.db"))
    service = ReconciliationService(database)

    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

    def state_for(request: Request, dataset_id: int, **extra: Any) -> dict[str, Any]:
        state = service.dataset_state(dataset_id)
        state.update({"request": request, "kind_labels": KIND_LABELS, **extra})
        return state

    @app.exception_handler(ReconciliationError)
    async def handle_reconciliation_error(request: Request, exc: ReconciliationError):
        wants_json = request.url.path.startswith("/api/")
        if wants_json:
            return JSONResponse(
                status_code=400,
                content={"error": str(exc), "code": exc.code},
            )
        dataset_id = request.path_params.get("dataset_id")
        if dataset_id is None or database.get_dataset(int(dataset_id)) is None:
            return templates.TemplateResponse(
                request,
                "error.html",
                {"message": str(exc), "code": exc.code, "title": "操作失败"},
                status_code=400,
            )
        return templates.TemplateResponse(
            request,
            "dataset.html",
            state_for(request, int(dataset_id), error=str(exc), error_code=exc.code),
            status_code=400,
        )

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return templates.TemplateResponse(
            request,
            "index.html",
            {"request": request, "datasets": database.list_datasets()},
        )

    @app.post("/datasets")
    def create_dataset(name: str = Form(...)):
        dataset = service.create_dataset(name)
        return RedirectResponse(url=f"/datasets/{dataset.id}", status_code=303)

    @app.get("/datasets/{dataset_id}", response_class=HTMLResponse)
    def dataset_page(request: Request, dataset_id: int):
        if database.get_dataset(dataset_id) is None:
            raise HTTPException(status_code=404, detail="数据集不存在")
        return templates.TemplateResponse(
            request,
            "dataset.html",
            state_for(request, dataset_id),
        )

    @app.post("/datasets/{dataset_id}/upload")
    def upload(
        dataset_id: int,
        kind: str = Form(...),
        file: UploadFile = None,  # type: ignore[assignment]
    ):
        if file is None:
            raise HTTPException(status_code=400, detail="未选择文件")
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="文件超出大小限制")
        outcome = service.upload(dataset_id, kind, file.filename or "upload.csv", data)
        message = (
            f"{KIND_LABELS[outcome.kind]}：{outcome.row_count} 行"
            + ("，已更新输入版本" if outcome.changed else "，内容未变化")
        )
        return RedirectResponse(
            url=f"/datasets/{dataset_id}?uploaded=1&message={quote(message)}",
            status_code=303,
        )

    @app.post("/datasets/{dataset_id}/preview")
    def run_preview(dataset_id: int):
        preview = service.preview(dataset_id)
        return RedirectResponse(
            url=f"/datasets/{dataset_id}?preview={preview['preview_id']}", status_code=303
        )

    @app.post("/datasets/{dataset_id}/publish")
    def publish(dataset_id: int, preview_id: int = Form(...)):
        published = service.publish(dataset_id, preview_id)
        return RedirectResponse(
            url=f"/datasets/{dataset_id}?published={published['version_number']}",
            status_code=303,
        )

    @app.get("/datasets/{dataset_id}/snapshots/{version_number}", response_class=HTMLResponse)
    def snapshot_page(request: Request, dataset_id: int, version_number: int):
        snapshot = database.get_snapshot(dataset_id, version_number)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="快照不存在")
        result = ReconciliationResult.from_dict(snapshot["result"])
        return templates.TemplateResponse(
            request,
            "snapshot.html",
            {
                "request": request,
                "dataset": database.get_dataset(dataset_id),
                "snapshot": snapshot,
                "result": result,
                "kind_labels": KIND_LABELS,
                "readonly": True,
            },
        )

    # ------------------------------------------------------------------
    # Downloads
    # ------------------------------------------------------------------
    def _resolve_result(dataset_id: int, version_number: int | None, preview_id: int | None):
        if version_number is not None:
            snapshot = database.get_snapshot(dataset_id, version_number)
            if snapshot is None:
                raise HTTPException(status_code=404, detail="快照不存在")
            return ReconciliationResult.from_dict(snapshot["result"]), f"snapshot-v{version_number}"
        if preview_id is not None:
            preview = database.get_preview(preview_id)
            if preview is None or preview["dataset_id"] != dataset_id:
                raise HTTPException(status_code=404, detail="预览不存在")
            return ReconciliationResult.from_dict(preview["result"]), f"preview-{preview_id}"
        raise HTTPException(status_code=400, detail="需要 version 或 preview 参数")

    @app.get("/datasets/{dataset_id}/download/{filename}")
    def download(
        dataset_id: int,
        filename: str,
        version: int | None = None,
        preview: int | None = None,
    ):
        result, stem = _resolve_result(dataset_id, version, preview)
        if filename == "snapshot.csv":
            body = result_to_csv(result)
        elif filename == "adjustments.csv":
            body = adjustments_to_csv(result)
        elif filename == "conflicts.csv":
            body = conflicts_to_csv(result)
        else:
            raise HTTPException(status_code=404, detail="未知下载类型")
        return Response(
            content="﻿" + body,
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="reconciliation-{stem}-{filename}"'
            },
        )

    # ------------------------------------------------------------------
    # JSON API (used by tests and automation)
    # ------------------------------------------------------------------
    @app.get("/api/datasets")
    def api_datasets():
        return [
            {"id": d.id, "name": d.name, "input_version": d.input_version}
            for d in database.list_datasets()
        ]

    @app.post("/api/datasets")
    def api_create_dataset(payload: dict[str, str]):
        dataset = service.create_dataset(payload.get("name", ""))
        return {"id": dataset.id, "name": dataset.name, "input_version": dataset.input_version}

    @app.get("/api/datasets/{dataset_id}")
    def api_state(dataset_id: int):
        return service.dataset_state(dataset_id)

    @app.post("/api/datasets/{dataset_id}/upload")
    async def api_upload(dataset_id: int, kind: str, file: UploadFile):
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="文件超出大小限制")
        outcome = service.upload(dataset_id, kind, file.filename or "upload.csv", data)
        return {
            "kind": outcome.kind,
            "changed": outcome.changed,
            "input_version": outcome.input_version,
            "row_count": outcome.row_count,
        }

    @app.post("/api/datasets/{dataset_id}/preview")
    def api_preview(dataset_id: int):
        return service.preview(dataset_id)

    @app.post("/api/datasets/{dataset_id}/publish")
    def api_publish(dataset_id: int, payload: dict[str, int]):
        return service.publish(dataset_id, int(payload["preview_id"]))

    @app.get("/api/datasets/{dataset_id}/snapshots")
    def api_snapshots(dataset_id: int):
        return database.list_snapshots(dataset_id)

    @app.get("/api/datasets/{dataset_id}/snapshots/{version_number}")
    def api_snapshot(dataset_id: int, version_number: int):
        snapshot = database.get_snapshot(dataset_id, version_number)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="快照不存在")
        return snapshot

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    return app


# Run with: uvicorn app.main:create_app --factory
