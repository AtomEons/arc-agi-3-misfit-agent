"""Verify the built submission.ipynb is structurally sound."""
from __future__ import annotations

import json
import sys
from pathlib import Path

NB = Path(__file__).parent.parent / "notebooks" / "submission.ipynb"
META = Path(__file__).parent.parent / "notebooks" / "kernel-metadata.json"


def main() -> int:
    if not NB.exists():
        print(f"FAIL: {NB} missing")
        return 1
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    print(f"notebook: {NB}")
    print(f"  size = {NB.stat().st_size:,} bytes")
    print(f"  cells = {len(cells)}")
    for i, c in enumerate(cells):
        body = "".join(c.get("source", []))
        first = body.splitlines()[0] if body else "(empty)"
        print(f"    cell {i}  {c['cell_type']:8}  {len(body):6,} chars  first: {first[:70]}")

    expected_markers = [
        ("Cell 1 has %%writefile",       "%%writefile /kaggle/working/my_agent.py" in "".join(cells[2].get("source", []))),
        ("Cell 1 has class Misfit",      "class Misfit(Agent):" in "".join(cells[2].get("source", []))),
        ("Cell 1 has MyAgent alias",     "MyAgent = Misfit" in "".join(cells[2].get("source", []))),
        ("Cell 2 has KAGGLE_IS_COMPETITION_RERUN", "KAGGLE_IS_COMPETITION_RERUN" in "".join(cells[3].get("source", []))),
        ("Cell 2 has slim __init__.py rewrite",    "AVAILABLE_AGENTS" in "".join(cells[3].get("source", []))),
        ("Cell 3 writes submission.parquet",       "submission.parquet" in "".join(cells[4].get("source", []))),
        ("Internet disabled",                       nb["metadata"]["kaggle"]["isInternetEnabled"] is False),
    ]
    print("\nstructural assertions:")
    all_ok = True
    for label, ok in expected_markers:
        flag = "ok  " if ok else "FAIL"
        print(f"  [{flag}] {label}")
        if not ok:
            all_ok = False

    meta = json.loads(META.read_text(encoding="utf-8"))
    print(f"\nkernel-metadata.json:")
    print(f"  id = {meta['id']}")
    print(f"  competition_sources = {meta['competition_sources']}")
    print(f"  enable_internet = {meta['enable_internet']}")
    print(f"  enable_gpu = {meta['enable_gpu']}")

    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
