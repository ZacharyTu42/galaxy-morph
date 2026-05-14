"""Download Galaxy10 DECaLS dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path

import requests
from tqdm import tqdm

GALAXY10_URL = "https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5"
GALAXY10_SHA256 = "19AEFC477C41BB7F77FF07599A6B82A038DC042F889A111B0D4D98BB755C1571"
GALAXY10_FILENAME = "Galaxy10_DECals.h5"


def download_galaxy10(dest_dir: Path, force: bool = False) -> Path:
    """Download Galaxy10 DECaLS HDF5 file if not already present.

    Parameters
    ----------
    dest_dir
        Directory to download into. Created if it doesn't exist.
    force
        If True, redownload even if the file exists.

    Returns
    -------
    Path to the downloaded file.
    """

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / GALAXY10_FILENAME

    if dest_path.exists() and not force:
        if _verify_checksum(dest_path, GALAXY10_SHA256):
            return dest_path
        # Checksum mismatch: redownload
        dest_path.unlink()

    _download_with_progress(GALAXY10_URL, dest_path)

    if not _verify_checksum(dest_path, GALAXY10_SHA256):
        raise RuntimeError(f"Checksum mismatch for {dest_path}")

    return dest_path


def _download_with_progress(url: str, dest: Path) -> None:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with dest.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def _verify_checksum(path: Path, expected: str) -> bool:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest().lower() == expected.lower()
