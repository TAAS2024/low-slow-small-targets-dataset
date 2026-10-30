"""Run every code cell of notebooks/01_quickstart.ipynb in one namespace.

Used as a smoke test in CI-less environments: the notebook must import and run
without model weights. Cells that require assets are guarded inside the notebook
itself (RUN_GENERATION), so nothing here is skipped by the harness.

    python notebooks/smoke_run.py
"""
import contextlib
import io
import json
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def main() -> int:
    import os
    os.chdir(REPO)
    nb = json.loads((HERE / "01_quickstart.ipynb").read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    failed = 0
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        buf = io.StringIO()
        try:
            compiled = compile(source, f"cell{index}", "exec")
            with contextlib.redirect_stdout(buf):
                exec(compiled, namespace)          # noqa: S102 - local test harness
        except Exception as exc:                   # noqa: BLE001
            failed += 1
            print(f"cell {index}: FAIL {type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue
        lines = buf.getvalue().strip().splitlines()
        print(f"cell {index}: OK  ({len(lines)} lines of output)")
        for line in lines[:10]:
            print("   ", line)
    print("\nfailed cells:", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
