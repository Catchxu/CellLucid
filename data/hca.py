"""HCA Azul file discovery; discovery never downloads matrix contents."""

from pathlib import Path
import json
import requests
from collections import Counter
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class HCAClient:
    def __init__(self, endpoint="https://service.azul.data.humancellatlas.org", catalog="dcp60"):
        self.endpoint = endpoint.rstrip("/")
        self.catalog = catalog
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.report = {}

    def discover(
        self,
        *,
        formats=("h5ad", "loom"),
        species="Homo sapiens",
        max_files=1,
        max_file_bytes=20_000_000,
        max_total_bytes=20_000_000,
        project_ids=None,
        file_source=None,
        max_pages=20,
    ):
        if max_files < 1 or max_file_bytes < 1 or max_total_bytes < 1 or max_pages < 1:
            raise ValueError("Set explicit positive discovery budgets")
        filters = {"fileFormat": {"is": list(formats)}, "genusSpecies": {"is": [species]}}
        if project_ids:
            filters["projectId"] = {"is": list(project_ids)}
        params = {
            "catalog": self.catalog,
            "filters": json.dumps(filters),
            "size": 100,
            "sort": "fileSize",
            "order": "asc",
        }
        url = self.endpoint + "/index/files"
        selected, seen, total = [], set(), 0
        skipped = Counter()
        self.report = {
            "pages_scanned": 0,
            "catalog_exhausted": False,
            "stop_reason": "max_pages",
            "skipped": {},
        }
        for _ in range(max_pages):
            response = self.session.get(url, params=params, timeout=(20, 90))
            response.raise_for_status()
            payload = response.json()
            self.report["pages_scanned"] += 1
            self.report["catalog_matching_hits"] = payload.get("pagination", {}).get("total")
            for hit in payload["hits"]:
                projects = hit.get("projects", [])
                ids = sorted({p for x in projects for p in x.get("projectId", [])})
                for item in hit.get("files", []):
                    identity = (item["uuid"], item["version"])
                    descriptions = " ".join(item.get("contentDescription", [])).lower()
                    if (
                        identity in seen
                        or item.get("accessible") is not True
                        or item.get("isIntermediate")
                        or "matrix" not in descriptions
                    ):
                        continue
                    if item.get("format") not in formats:
                        continue
                    if file_source and item.get("fileSource") != file_source:
                        continue
                    size = int(item["size"])
                    if size <= 0 or size > max_file_bytes:
                        skipped["per_file_bytes"] += 1
                        continue
                    if total + size > max_total_bytes:
                        skipped["total_bytes"] += 1
                        continue
                    mirror = item.get("azul_mirror_uri", "")
                    download_url = item.get("azul_url")
                    if mirror.startswith("s3://humancellatlas/"):
                        download_url = mirror.replace(
                            "s3://humancellatlas/", "https://humancellatlas.s3.amazonaws.com/", 1
                        )
                    if not download_url:
                        continue
                    selected.append(
                        {
                            "file_id": item["uuid"],
                            "file_version": item["version"],
                            "project_ids": ids,
                            "file_name": Path(item["name"]).name,
                            "format": item["format"],
                            "file_source": item.get("fileSource"),
                            "matrix_role": (
                                "contributor"
                                if item.get("fileSource") == "Contributor"
                                else "unresolved"
                            ),
                            "size_bytes": size,
                            "url": download_url,
                            "azul_url": item.get("azul_url"),
                            "reported_matrix_cells": item.get("matrixCellCount"),
                            "status": "selected",
                            "metadata": hit,
                        }
                    )
                    seen.add(identity)
                    total += size
                    if len(selected) == max_files:
                        self.report.update(
                            stop_reason="max_files",
                            selected_files=len(selected),
                            selected_bytes=total,
                            skipped=dict(skipped),
                        )
                        return selected
            next_url = payload.get("pagination", {}).get("next")
            if not next_url:
                self.report.update(catalog_exhausted=True, stop_reason="catalog_exhausted")
                break
            if not next_url.startswith(self.endpoint + "/"):
                raise ValueError("Pagination unexpectedly left the HCA endpoint")
            url, params = next_url, None
        self.report.update(
            selected_files=len(selected), selected_bytes=total, skipped=dict(skipped)
        )
        return selected
