"""Bounded, resumable HTTP transfers with atomic completion and file-size checks."""

from pathlib import Path
import os
import time
import requests


def download_file(url, destination, *, size=None, max_bytes=None, retries=3):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if size is not None and max_bytes is not None and size > max_bytes:
        raise ValueError("Declared file size exceeds download budget")
    if destination.exists():
        if size is None or destination.stat().st_size == size:
            return destination
        raise ValueError(f"Existing destination fails integrity check: {destination}")
    part = destination.with_name(destination.name + ".part")
    for attempt in range(retries):
        try:
            offset = part.stat().st_size if part.exists() else 0
            if size is not None and offset > size:
                part.unlink()
                offset = 0
            if size is None or offset != size:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with requests.get(url, headers=headers, stream=True, timeout=(20, 120)) as response:
                    response.raise_for_status()
                    append = offset > 0 and response.status_code == 206
                    if append:
                        content_range = response.headers.get("Content-Range", "")
                        if not content_range.startswith(f"bytes {offset}-"):
                            raise ValueError("Server returned an inconsistent Content-Range")
                    else:
                        offset = 0
                    length = response.headers.get("Content-Length")
                    if max_bytes and length and offset + int(length) > max_bytes:
                        raise ValueError("Response exceeds download budget")
                    with part.open("ab" if append else "wb") as handle:
                        for chunk in response.iter_content(1024 * 1024):
                            if not chunk:
                                continue
                            if max_bytes and handle.tell() + len(chunk) > max_bytes:
                                raise ValueError("Download exceeds byte budget")
                            handle.write(chunk)
            if size is not None and part.stat().st_size != size:
                raise IOError("Downloaded size differs from manifest")
            os.replace(part, destination)
            return destination
        except (requests.RequestException, OSError):
            if attempt + 1 == retries:
                raise
            time.sleep(min(2**attempt, 8))
    raise RuntimeError("Download failed")
