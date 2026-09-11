"""Download the PadelTracker100 dataset from Zenodo into a local data directory.

Dataset: PadelTracker100, DOI 10.5281/zenodo.14653706, licensed CC-BY-4.0.
The data directory is git-ignored; nothing downloaded here is ever committed.
"""

import argparse
import json
import urllib.request
from pathlib import Path

ZENODO_RECORD = "14653706"
API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}"


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    percent = min(100.0, downloaded * 100.0 / total_size)
    if block_num % 2000 == 0 or percent >= 100.0:
        print(f"  {percent:5.1f}%  {downloaded / 1e9:.2f} / {total_size / 1e9:.2f} GB", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("data/padeltracker100"))
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(API_URL) as response:
        record = json.load(response)

    for entry in record["files"]:
        name = entry["key"]
        url = entry["links"]["self"]
        size_gb = entry["size"] / 1e9
        target = args.dest / name
        if target.exists() and target.stat().st_size == entry["size"]:
            print(f"skip  {name} ({size_gb:.2f} GB) - already complete", flush=True)
            continue
        print(f"fetch {name} ({size_gb:.2f} GB) -> {target}", flush=True)
        urllib.request.urlretrieve(url, target, reporthook=_progress)

    print("done", flush=True)


if __name__ == "__main__":
    main()
