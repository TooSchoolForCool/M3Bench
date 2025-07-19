from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from tongverse.utils.constant import ASSETS_VERSION

ASSETS_DIR = Path(__file__).parent / "assets"


def download_assets_if_not_existed():
    if not (ASSETS_DIR / ASSETS_VERSION).is_dir():
        download_assets()


def download_assets(assets_url=None, assets_version=None):
    if assets_version is None:
        assets_version = ASSETS_VERSION

    assets_dir = ASSETS_DIR / assets_version

    if assets_dir.is_dir():
        raise FileExistsError(f"Assets Path Exists: {assets_dir}")

    if assets_url is None:
        url_base = os.getenv("TONGVERSE_RELEASE_URLBASE")
        if url_base is None:
            raise NameError('assets_url or "TONGVERSE_RELEASE_URLBASE" is not defined')
        assets_url = f"{url_base}/tongverse-assets-{assets_version}.zip"

    with requests.get(assets_url, timeout=30 * 60, stream=True) as r:
        r.raise_for_status()

        with tqdm.wrapattr(
            tempfile.NamedTemporaryFile("wb"),
            "write",
            miniters=1,
            desc=assets_url.split("/")[-1],
            total=int(r.headers.get("content-length", 0)),
        ) as fd:
            for chunk in r.iter_content(chunk_size=8 * 1024):
                fd.write(chunk)
            fd.flush()
            with zipfile.ZipFile(fd.name) as zpf:
                zpf.extractall(assets_dir.parent.parent)


if __name__ == "__main__":
    # argparse is defined by Kit, use sys.argv instead
    import sys

    from tongverse import shutdown

    if len(sys.argv) > 1:
        download_assets(assets_version=sys.argv[1])
    else:
        download_assets()

    shutdown()
