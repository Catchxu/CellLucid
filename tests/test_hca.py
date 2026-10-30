import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace
import pytest
from data.hca import HCAClient
from data.download import download_file
from data.manifest import save_manifest, load_manifest


@pytest.fixture
def server():
    body = b"an expression matrix\n" * 30
    requested = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            requested.append(self.headers.get("Range"))
            offset = int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
            self.send_response(206 if offset else 200)
            self.send_header("Content-Length", str(len(body) - offset))
            if offset:
                self.send_header("Content-Range", f"bytes {offset}-{len(body) - 1}/{len(body)}")
            self.end_headers()
            self.wfile.write(body[offset:])

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_port}/matrix", body, requested
    srv.shutdown()
    srv.server_close()
    thread.join()


def test_resume_size_idempotence_and_budget(server, tmp_path):
    url, body, requested = server
    path = tmp_path / "matrix.h5ad"
    part = tmp_path / "matrix.h5ad.part"
    part.write_bytes(body[:17])
    download_file(url, path, size=len(body), max_bytes=len(body))
    assert path.read_bytes() == body and requested == ["bytes=17-"]
    download_file(url, path, size=len(body))
    assert len(requested) == 1
    with pytest.raises(ValueError, match="budget"):
        download_file(url, tmp_path / "too-large", size=len(body), max_bytes=2)
    with pytest.raises(ValueError, match="integrity"):
        download_file(url, path, size=len(body) + 1)


def test_discovery_uses_hca_filters_and_excludes_nonmatrix_and_inaccessible(tmp_path):
    template = {
        "uuid": "a",
        "version": "v1",
        "accessible": True,
        "contentDescription": ["Count matrix"],
        "format": "h5ad",
        "name": "matrix.h5ad",
        "size": 100,
        "fileSource": "Contributor",
        "azul_url": "https://example.org/a",
    }
    payload = {
        "hits": [
            {
                "projects": [{"projectId": ["p"]}],
                "files": [
                    dict(template, uuid="no", accessible=False),
                    dict(template, uuid="image", contentDescription=["Embedding"]),
                    template,
                    dict(template),
                ],
            }
        ],
        "pagination": {"next": None},
    }
    requests = []

    def get(url, params, timeout):
        requests.append(params)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)

    client = HCAClient()
    client.session = SimpleNamespace(get=get)
    rows = client.discover(max_files=2, max_file_bytes=1000, max_total_bytes=1000)
    assert len(rows) == 1 and rows[0]["project_ids"] == ["p"]
    filters = json.loads(requests[0]["filters"])
    assert filters["genusSpecies"] == {"is": ["Homo sapiens"]}
    path = tmp_path / "manifest.json"
    save_manifest(path, rows)
    assert load_manifest(path)["records"][0]["file_id"] == "a"
    with pytest.raises(ValueError, match="Duplicate"):
        save_manifest(path, rows + rows)


@pytest.mark.parametrize("mode", ["max_pages", "max_files", "per_file_bytes"])
def test_discovery_reports_limits(mode):
    item = {
        "uuid": "a",
        "version": "v1",
        "accessible": True,
        "contentDescription": ["Count matrix"],
        "format": "h5ad",
        "name": "matrix.h5ad",
        "size": 100,
        "azul_url": "https://example.org/a",
    }
    if mode == "per_file_bytes":
        item["size"] = 200
    payload = {
        "hits": [{"files": [item]}],
        "pagination": {
            "total": 2,
            "next": "https://service.azul.data.humancellatlas.org/index/files?page=2",
        },
    }
    client = HCAClient()
    client.session = SimpleNamespace(
        get=lambda *a, **k: SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
    )
    rows = client.discover(
        max_pages=1, max_files=1 if mode == "max_files" else 2, max_file_bytes=100
    )
    assert not client.report["catalog_exhausted"]
    assert client.report["stop_reason"] == ("max_files" if mode == "max_files" else "max_pages")
    assert client.report["selected_files"] == len(rows)
    if mode == "per_file_bytes":
        assert not rows and client.report["skipped"]["per_file_bytes"] == 1


def _download_config(tmp_path, records):
    selection = {
        "max_files": len(records),
        "max_file_bytes": 10000,
        "max_total_bytes": 10000,
    }
    save_manifest(tmp_path / "manifest.json", records, query=selection)
    return {"output_dir": str(tmp_path), "selection": selection, "download": {"workers": 2}}


def test_parallel_failures_are_persisted_and_retry_keeps_verified_files(server, tmp_path):
    from data.download_hca import run

    url, body, requested = server
    record = {
        "file_id": "a",
        "file_version": "v1",
        "file_name": "a.h5ad",
        "size_bytes": len(body),
        "url": url,
        "status": "selected",
        "metadata": {},
    }
    bad = dict(record, file_id="b", file_name="b.h5ad", size_bytes=len(body) + 1)
    config = _download_config(tmp_path, [record, bad])
    with pytest.raises(RuntimeError, match="1 HCA downloads failed"):
        run(config)
    manifest = load_manifest(tmp_path / "manifest.json")
    assert [r["status"] for r in manifest["records"]] == ["verified", "failed"]
    assert len(requested) == 4  # One good transfer and three size-mismatch attempts.
    manifest["records"][1]["size_bytes"] = record["size_bytes"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    manifest = run(config)
    assert all(r["status"] == "verified" for r in manifest["records"])
    assert len(requested) == 4  # Reuse the valid file and completed partial transfer.


def test_inventory_does_not_transfer_and_lock_prevents_duplicate_writers(server, tmp_path):
    from data.download_hca import run, _exclusive_lock

    url, body, requested = server
    record = {
        "file_id": "a",
        "file_version": "v1",
        "file_name": "a.h5ad",
        "size_bytes": len(body),
        "url": url,
        "status": "selected",
        "metadata": {},
    }
    config = _download_config(tmp_path, [record])
    assert len(run(config, inventory_only=True)["records"]) == 1
    assert requested == [] and not (tmp_path / "raw").exists()
    with _exclusive_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another HCA process"):
            run(config)
    run(config)
    assert len(requested) == 1


def test_space_guard_and_catalog_mismatch_stop_before_transfers(server, tmp_path, monkeypatch):
    from data import download_hca as task

    url, body, requested = server
    record = {
        "file_id": "a",
        "file_version": "v1",
        "file_name": "a.h5ad",
        "size_bytes": len(body),
        "url": url,
        "status": "selected",
        "metadata": {},
    }
    config = _download_config(tmp_path, [record])
    monkeypatch.setattr(task.shutil, "disk_usage", lambda p: SimpleNamespace(free=0))
    with pytest.raises(OSError, match="Insufficient disk space"):
        task.run(config)
    assert requested == []
    manifest = load_manifest(tmp_path / "manifest.json")
    manifest["api"] = {"catalog": "other"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="another API/catalog"):
        task.run(config, inventory_only=True)
