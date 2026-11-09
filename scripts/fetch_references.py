"""Pull the reference repos into references/.

The uplink here is roughly 15-25 KB/s, so the two big repos (31 MB and 42 MB)
cannot be pulled whole in reasonable time. Strategy:

  * small repos  -> full zip snapshot from codeload
  * big repos    -> only the files that carry the reference value (notebooks, README)

Run:  .venv\\Scripts\\python.exe -u scripts/fetch_references.py
"""

import io
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

DEST = Path("references")
UA = {"User-Agent": "Mozilla/5.0"}

ZIP_TARGETS = [
    ("daniilnelyubin/TotalRounds", "TotalRounds", ("master", "main")),
    ("akimerslys/hltv-async-api", "hltv-async-api", ("main", "master")),
    ("M3MONs/hltv-scraper-api", "hltv-scraper-api", ("main", "master")),
]

FILE_TARGETS = [
    (
        "FCosmin27/cs2-predictive-models",
        "cs2-predictive-models",
        "main",
        [
            "README.md",
            "requirements.txt",
            "models/logistic_regression.ipynb",
            "models/random_forest.ipynb",
            "models/xgboost.ipynb",
            "models/neural_network.ipynb",
            "models/knn.ipynb",
            "models/adaboost.ipynb",
        ],
    ),
    (
        "luizcieslak/cs2-match-prediction",
        "cs2-match-prediction-llm",
        "main",
        ["README.md"],
    ),
]


def get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_zip(repo: str, name: str, branches: tuple[str, ...]) -> None:
    out = DEST / name
    if out.exists() and any(out.iterdir()):
        print(f"skip  {name} (present)")
        return
    for branch in branches:
        try:
            blob = get(f"https://codeload.github.com/{repo}/zip/refs/heads/{branch}", timeout=300)
        except Exception as exc:  # noqa: BLE001
            print(f"      {name}@{branch}: {type(exc).__name__}")
            continue
        tmp = DEST / "_tmp"
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            top = zf.namelist()[0].split("/")[0]
            zf.extractall(tmp)
        shutil.move(str(tmp / top), str(out))
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"ok    {name}  {len(blob) / 1e6:.1f} MB")
        return
    print(f"FAIL  {name}")


def fetch_files(repo: str, name: str, branch: str, files: list[str]) -> None:
    out = DEST / name
    out.mkdir(parents=True, exist_ok=True)
    for path in files:
        target = out / path
        if target.exists():
            print(f"skip  {name}/{path}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://cdn.jsdelivr.net/gh/{repo}@{branch}/{path}"
        for attempt in range(1, 4):
            try:
                blob = get(url, timeout=180)
                target.write_bytes(blob)
                print(f"ok    {name}/{path}  {len(blob) / 1e3:.0f} KB")
                break
            except Exception as exc:  # noqa: BLE001
                print(f"      retry {attempt} {name}/{path}: {type(exc).__name__}")
                time.sleep(3)
        else:
            print(f"FAIL  {name}/{path}")


def main() -> None:
    DEST.mkdir(exist_ok=True)
    for repo, name, branches in ZIP_TARGETS:
        fetch_zip(repo, name, branches)
    for repo, name, branch, files in FILE_TARGETS:
        fetch_files(repo, name, branch, files)
    print("DONE")


if __name__ == "__main__":
    main()
