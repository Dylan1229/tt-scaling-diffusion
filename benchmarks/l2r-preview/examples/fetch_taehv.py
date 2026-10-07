"""Fetch only the pinned TAEHV source, license, documentation, and Wan2.2 weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

COMMIT = "011dfc2112197741c540e0bdd5b7b67bcc930771"
FILES = {
    "LICENSE": "532f9e394518ffddecd294a517d5b41d79d3d3866c3fb95a6cb0e8bcc02370bf",
    "README.md": "a1c7afefb557bd09bfd4795e18da2ee3d3ff1c9fb6f3d87e42f6988b9c7718f3",
    "taehv.py": "99545f093b36c16297ad182ce18a5dce2ebc98bb611f5e8dfa2dce431ba39ce2",
    "safetensors/taew2_2.safetensors": "b84609b2a133d48434bd9636bfcb44bf05168dc436e2d3cecf26256faa1f5325",
    "examples/TAEW2.2_T2I_Demo.ipynb": "93b8efd89be21769f1323d0dac05ef394705d57fab199f38e9ea33dd38c20b6d",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/taehv_20261002/upstream"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = args.output / "revision.json"
    manifest = (
        json.loads(provenance.read_text())
        if provenance.exists()
        else {
            "repo": "https://github.com/madebyollin/taehv",
            "commit": COMMIT,
        }
    )
    if manifest["commit"] != COMMIT:
        raise ValueError("Output directory contains another upstream revision")
    downloads = []
    for name, expected in FILES.items():
        path = args.output / Path(name).name
        if path.exists():
            data = path.read_bytes()
        else:
            url = f"https://raw.githubusercontent.com/madebyollin/taehv/{COMMIT}/{name}"
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read(30 * 1024 * 1024)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Download hash mismatch: {name}")
            path.write_bytes(data)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Existing file hash mismatch: {path}")
        downloads.append({"file": path.name, "bytes": len(data), "sha256": expected})
    manifest["downloads"] = downloads
    provenance.write_text(json.dumps(manifest, indent=2))
    print(f"Verified {len(downloads)} files at {COMMIT}")


if __name__ == "__main__":
    main()
