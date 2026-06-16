"""Verify the built submission.ipynb is structurally sound (v4).

Checks per the operator brief:
  (a) notebook AST parses (json.loads succeeds + minimum cell count + types)
  (b) all 12 misfit_agent .py modules appear inside the %%writefile cell
  (c) Misfit class IS in the AVAILABLE_AGENTS dict (slim init rewrite)
  (d) enable_internet=false in kernel-metadata.json AND in nb.metadata.kaggle

Exit code 0 = all green, 2 = at least one assertion failed.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
NB = REPO / "notebooks" / "submission.ipynb"
META = REPO / "notebooks" / "kernel-metadata.json"
SRC = REPO / "src" / "misfit_agent"

# Must match build_notebook.MODULE_ORDER (kept duplicated rather than imported
# so verify can run before/independent of build).
EXPECTED_MODULES = [
    "config.py",
    "perceptor.py",
    "episode.py",
    "fingerprint.py",
    "resonance.py",
    "rules/no_op.py",
    "rules/translate.py",
    "world_model.py",
    "click_quantizer.py",
    "tracker_hungarian.py",
    "goal_inducer.py",
    "abstain_policy.py",
    "action_search.py",
    "mcts_puct.py",
    "misfit_agent.py",
]


def _cell_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _parse_notebook() -> tuple[dict, list[str]]:
    """Return (notebook_json, [structural errors])."""
    errs: list[str] = []
    if not NB.exists():
        errs.append(f"notebook missing: {NB}")
        return {}, errs
    try:
        nb = json.loads(NB.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        errs.append(f"notebook JSON invalid: {e}")
        return {}, errs

    if "cells" not in nb or not isinstance(nb["cells"], list):
        errs.append("notebook missing 'cells' list")
        return nb, errs

    if len(nb["cells"]) < 5:
        errs.append(f"expected >=5 cells, got {len(nb['cells'])}")

    return nb, errs


def _check_writefile_has_all_modules(writefile_text: str) -> list[str]:
    """Check that every expected module appears in the %%writefile bundle.

    The build script tags each module with a `# MODULE: <path>` marker.
    """
    missing: list[str] = []
    for mod in EXPECTED_MODULES:
        marker = f"# MODULE: {mod}"
        if marker not in writefile_text:
            missing.append(mod)
    return missing


def _check_misfit_class_compiles(writefile_text: str) -> list[str]:
    """Strip the `%%writefile ...` line and parse the rest as Python AST.

    The bundled module imports `arcengine` and `agents.agent` which are not
    installed locally; AST parsing only validates syntax (no import side
    effects), so this is the right check for "notebook AST parses".
    """
    errs: list[str] = []
    lines = writefile_text.splitlines(keepends=True)
    if not lines or not lines[0].startswith("%%writefile"):
        errs.append("first line of cell 2 is not '%%writefile ...'")
        return errs
    py_source = "".join(lines[1:])
    try:
        tree = ast.parse(py_source)
    except SyntaxError as e:
        errs.append(f"bundled my_agent.py syntax error: {e}")
        return errs

    # Confirm `class Misfit(Agent):` is present at top level.
    has_misfit = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Misfit":
            has_misfit = True
            break
    if not has_misfit:
        errs.append("class Misfit not found in bundled my_agent.py")

    # Confirm `MyAgent = Misfit` alias exists.
    if "MyAgent = Misfit" not in py_source:
        errs.append("missing `MyAgent = Misfit` alias")

    return errs


def _check_available_agents_has_misfit(cell_text: str) -> list[str]:
    """Cell 3 must rewrite agents/__init__.py with Misfit registered.

    We look for the slim init body inside the cell — the build script embeds
    it via `repr()` so the body appears as a Python string literal inside the
    cell. We pattern-match for the AVAILABLE_AGENTS dict that maps to MyAgent
    (canonical) and/or misfit (alias).
    """
    errs: list[str] = []

    if "AVAILABLE_AGENTS" not in cell_text:
        errs.append("Cell 3 does not reference AVAILABLE_AGENTS")

    # Match either the `"myagent": ... MyAgent` or `"misfit": ... MyAgent`
    # binding so the slim-init body counts as registering Misfit. The build
    # script embeds the slim-init body via repr() so the inner double-quotes
    # appear as backslash-escaped `\"` inside the outer Python string literal,
    # AND the binding value is `cast(Type[Agent], MyAgent)` — note the comma
    # inside `cast(...)` defeats a naive `[^,}]*` rule. Match `MyAgent` within
    # 80 chars of the key, dot-all so the slim-init body's escaped newlines
    # don't break a one-line check.
    has_myagent_binding = bool(
        re.search(r"myagent.{0,80}MyAgent", cell_text, flags=re.DOTALL)
    )
    has_misfit_binding = bool(
        re.search(r"misfit.{0,80}MyAgent", cell_text, flags=re.DOTALL)
    )
    if not (has_myagent_binding or has_misfit_binding):
        errs.append("AVAILABLE_AGENTS dict in Cell 3 does not bind MyAgent")

    # Same tolerance for the import line — accept it whether the quotes are
    # raw or escaped inside the embedded slim-init body.
    if "from .templates.my_agent import MyAgent" not in cell_text:
        errs.append("Cell 3 slim init does not import MyAgent from templates")

    return errs


def _check_internet_disabled(nb: dict, meta: dict) -> list[str]:
    """Internet must be off in BOTH nb.metadata.kaggle and kernel-metadata.json."""
    errs: list[str] = []
    nb_kaggle = nb.get("metadata", {}).get("kaggle", {})
    if nb_kaggle.get("isInternetEnabled") is not False:
        errs.append(
            f"nb.metadata.kaggle.isInternetEnabled != false: {nb_kaggle.get('isInternetEnabled')!r}"
        )
    meta_internet = meta.get("enable_internet")
    if meta_internet not in (False, "false"):
        errs.append(
            f"kernel-metadata.json enable_internet != 'false': {meta_internet!r}"
        )
    return errs


def _check_gpu_disabled(meta: dict) -> list[str]:
    """GPU should be off — Tier-1 substrate is CPU-only."""
    errs: list[str] = []
    gpu = meta.get("enable_gpu")
    if gpu not in (False, "false"):
        errs.append(f"kernel-metadata.json enable_gpu != 'false': {gpu!r}")
    return errs


def _check_kernel_id(meta: dict) -> list[str]:
    expected = "atommccree/agi-in-a-video-shop-atom-eons-nostalgia"
    if meta.get("id") != expected:
        return [f"kernel id != {expected!r}: {meta.get('id')!r}"]
    return []


def _check_rerun_bootstrap_present(cell_text: str) -> list[str]:
    """Cell 3 must detect KAGGLE_IS_COMPETITION_RERUN and write the slim init."""
    errs: list[str] = []
    if "KAGGLE_IS_COMPETITION_RERUN" not in cell_text:
        errs.append("Cell 3 missing KAGGLE_IS_COMPETITION_RERUN env-var check")
    if "RECORDINGS_DIR" not in cell_text:
        errs.append("Cell 3 missing RECORDINGS_DIR in .env body")
    if "gateway:8001" not in cell_text:
        errs.append("Cell 3 missing gateway:8001 endpoint")
    return errs


def _check_run_trigger(cell_text: str) -> list[str]:
    """Cell 4 must contain both the dummy-parquet writer AND the run trigger."""
    errs: list[str] = []
    if "submission.parquet" not in cell_text:
        errs.append("Cell 4 missing submission.parquet write")
    if "python main.py" not in cell_text:
        errs.append("Cell 4 missing `python main.py` agent run trigger")
    if "--agent myagent" not in cell_text:
        errs.append("Cell 4 missing `--agent myagent` flag")
    return errs


def main() -> int:
    print(f"verifying {NB}")

    nb, parse_errs = _parse_notebook()
    if not META.exists():
        print(f"FAIL: kernel-metadata.json missing at {META}")
        return 2
    try:
        meta = json.loads(META.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"FAIL: kernel-metadata.json invalid: {e}")
        return 2

    if parse_errs:
        for e in parse_errs:
            print(f"  FAIL parse: {e}")
        return 2

    cells = nb["cells"]
    print(f"  size  = {NB.stat().st_size:,} bytes")
    print(f"  cells = {len(cells)}")
    for i, c in enumerate(cells):
        body = _cell_text(c)
        first = body.splitlines()[0] if body else "(empty)"
        print(f"    cell {i}  {c['cell_type']:8}  {len(body):7,} chars  first: {first[:70]}")

    # Cell layout: 0=md disclosure, 1=install, 2=%%writefile, 3=rerun bootstrap, 4=dummy+run
    cell0_text = _cell_text(cells[0])
    cell1_text = _cell_text(cells[1])
    cell2_text = _cell_text(cells[2])
    cell3_text = _cell_text(cells[3])
    cell4_text = _cell_text(cells[4])

    all_errs: list[str] = []

    # (a) notebook AST parses — bundled Python in Cell 2 must parse cleanly.
    print("\n(a) notebook AST parses:")
    ast_errs = _check_misfit_class_compiles(cell2_text)
    for e in ast_errs:
        print(f"  FAIL: {e}")
    if not ast_errs:
        print("  ok")
    all_errs.extend(ast_errs)

    # (b) all 12 misfit_agent .py files present in %%writefile cell
    print("\n(b) all expected modules present in %%writefile:")
    missing = _check_writefile_has_all_modules(cell2_text)
    if missing:
        for m in missing:
            print(f"  FAIL missing module: {m}")
    else:
        print(f"  ok  ({len(EXPECTED_MODULES)} modules)")
    all_errs.extend([f"missing module: {m}" for m in missing])

    # (c) Misfit class IS in AVAILABLE_AGENTS dict (slim init in Cell 3)
    print("\n(c) Misfit registered in AVAILABLE_AGENTS:")
    agents_errs = _check_available_agents_has_misfit(cell3_text)
    rerun_errs = _check_rerun_bootstrap_present(cell3_text)
    for e in agents_errs + rerun_errs:
        print(f"  FAIL: {e}")
    if not agents_errs and not rerun_errs:
        print("  ok")
    all_errs.extend(agents_errs + rerun_errs)

    # (d) enable_internet=false in both nb.metadata and kernel-metadata.json
    print("\n(d) enable_internet=false:")
    net_errs = _check_internet_disabled(nb, meta)
    gpu_errs = _check_gpu_disabled(meta)
    id_errs = _check_kernel_id(meta)
    for e in net_errs + gpu_errs + id_errs:
        print(f"  FAIL: {e}")
    if not (net_errs or gpu_errs or id_errs):
        print(f"  ok  (id={meta.get('id')!r}, gpu={meta.get('enable_gpu')!r}, internet={meta.get('enable_internet')!r})")
    all_errs.extend(net_errs + gpu_errs + id_errs)

    # Bonus: Cell 1 has offline install + Cell 4 has both branches.
    print("\nbonus checks:")
    install_ok = ("pip install" in cell1_text and "--no-index" in cell1_text)
    print(f"  cell 1 offline pip install: {'ok' if install_ok else 'FAIL'}")
    if not install_ok:
        all_errs.append("cell 1 missing offline pip install pattern")

    cell4_errs = _check_run_trigger(cell4_text)
    for e in cell4_errs:
        print(f"  FAIL: {e}")
    if not cell4_errs:
        print("  cell 4 dummy parquet + agent run trigger: ok")
    all_errs.extend(cell4_errs)

    cell0_disclosure_ok = ("Tier-1" in cell0_text) and ("Spelke" in cell0_text)
    print(f"  cell 0 Tier-1 disclosure embedded: {'ok' if cell0_disclosure_ok else 'FAIL'}")
    if not cell0_disclosure_ok:
        all_errs.append("cell 0 disclosure missing Tier-1 / Spelke text")

    print("\n=== SUMMARY ===")
    if all_errs:
        print(f"FAIL: {len(all_errs)} assertion(s) failed")
        return 2
    print("ok: all assertions green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
