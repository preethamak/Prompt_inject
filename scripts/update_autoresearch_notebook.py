#!/usr/bin/env python
from __future__ import annotations

import contextlib
import io
import json
import traceback
from pathlib import Path


def display(obj) -> None:
    if hasattr(obj, "to_string"):
        print(obj.to_string(index=False))
    else:
        print(obj)


def main() -> None:
    path = Path("AutoResearch.ipynb")
    nb = json.loads(path.read_text())
    ns = {"__name__": "__main__", "display": display}
    exec_count = 1

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        cell["execution_count"] = exec_count
        cell["outputs"] = []

        stdout = io.StringIO()
        stderr = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compile(source, f"AutoResearch.ipynb cell {exec_count}", "exec"), ns)
        except Exception:
            text = stdout.getvalue() + stderr.getvalue()
            if text:
                cell["outputs"].append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": text,
                })
            cell["outputs"].append({
                "ename": "ExecutionError",
                "evalue": "See traceback",
                "output_type": "error",
                "traceback": traceback.format_exc().splitlines(),
            })
            path.write_text(json.dumps(nb, indent=1))
            raise

        text = stdout.getvalue() + stderr.getvalue()
        if text:
            cell["outputs"].append({
                "name": "stdout",
                "output_type": "stream",
                "text": text,
            })

        exec_count += 1
        path.write_text(json.dumps(nb, indent=1))

    print(f"Notebook updated: {path}")


if __name__ == "__main__":
    main()
