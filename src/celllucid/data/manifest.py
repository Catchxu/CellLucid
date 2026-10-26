"""Versioned, source-preserving dataset manifests."""

from pathlib import Path
import json
from datetime import datetime, timezone
from ..config import write_json, fingerprint


def save_manifest(path, records, *, source="HCA", query=None):
    identities = [(r["file_id"], r["file_version"]) for r in records]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate file versions in manifest")
    payload = {
        "format_version": 1,
        "source": source,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection": query or {},
        "records": records,
    }
    payload["selection_fingerprint"] = fingerprint(
        {"selection": payload["selection"], "files": identities}
    )
    write_json(path, payload)
    return payload


def load_manifest(path):
    payload = json.loads(Path(path).read_text())
    if payload.get("format_version") != 1 or not isinstance(payload.get("records"), list):
        raise ValueError("Unsupported manifest")
    return payload
