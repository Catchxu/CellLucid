import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace
import pytest
from celllucid.data.hca import HCAClient
from celllucid.data.download import download_file
from celllucid.data.manifest import save_manifest, load_manifest


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


def test_resume_checksum_idempotence_and_budget(server, tmp_path):
    url, body, requested = server
    path = tmp_path / "matrix.h5ad"
    part = tmp_path / "matrix.h5ad.part"
    part.write_bytes(body[:17])
    sha = hashlib.sha256(body).hexdigest()
    download_file(url, path, sha256=sha, size=len(body), max_bytes=len(body))
    assert path.read_bytes() == body and requested == ["bytes=17-"]
    download_file(url, path, sha256=sha, size=len(body))
    assert len(requested) == 1
    with pytest.raises(ValueError, match="budget"):
        download_file(url, tmp_path / "too-large", size=len(body), max_bytes=2)
    with pytest.raises(IOError, match="SHA256"):
        download_file(url, tmp_path / "bad", sha256="0" * 64, size=len(body), retries=1)
    assert not (tmp_path / "bad").exists()


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
        "sha256": "a" * 64,
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
