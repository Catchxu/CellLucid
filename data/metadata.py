"""Project summaries are provenance, not fabricated per-cell donor annotations."""


def project_summary(record):
    hit = record.get("metadata", {})
    projects = hit.get("projects", [])
    return {
        "source": "HCA",
        "file_id": record["file_id"],
        "file_version": record["file_version"],
        "project_ids": record.get("project_ids", []),
        "project_titles": [v for p in projects for v in p.get("projectTitle", [])],
        "file_source": record.get("file_source"),
        "matrix_role": record.get("matrix_role"),
        "per_cell_metadata": "Retained from matrix; project aggregates are not cell assignments",
    }
