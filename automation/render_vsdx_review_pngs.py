"""Render VSDX files to PNGs for review without letting one file hang the batch.

Usage:
    python automation/render_vsdx_review_pngs.py \
        --input-dir output/reports/review_batch_20_readable \
        --output-dir output/reports/review_batch_20_readable/_rendered_png \
        --timeout 90

Requires Microsoft Visio + pywin32 on Windows. Each VSDX is rendered in a
separate child process and is terminated if it exceeds the timeout.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import sys
from pathlib import Path
from typing import List, Tuple


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "page"


def _render_one_worker(vsdx_path: str, output_dir: str, queue) -> None:
    try:
        import win32com.client  # type: ignore

        vsdx = Path(vsdx_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        app = win32com.client.DispatchEx("Visio.Application")
        app.Visible = False
        rendered: List[str] = []
        try:
            doc = app.Documents.Open(str(vsdx))
            try:
                for idx in range(1, doc.Pages.Count + 1):
                    page = doc.Pages.Item(idx)
                    page_name = _safe_name(str(page.Name))
                    png = out_dir / f"{vsdx.stem}__p{idx}_{page_name}.png"
                    page.Export(str(png))
                    rendered.append(str(png))
            finally:
                doc.Close()
        finally:
            app.Quit()
        queue.put(("ok", rendered))
    except Exception as exc:  # pragma: no cover - COM/environment dependent
        queue.put(("error", repr(exc)))


def render_with_timeout(vsdx: Path, output_dir: Path, timeout: int) -> Tuple[str, List[str]]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_render_one_worker,
        args=(str(vsdx), str(output_dir), queue),
        daemon=False,
    )
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        return "timeout", []
    if not queue.empty():
        status, payload = queue.get()
        if status == "ok":
            return "ok", payload
        return f"error: {payload}", []
    if proc.exitcode not in (0, None):
        return f"error: child exited {proc.exitcode}", []
    return "error: no render result", []


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely render VSDX review files to PNG.")
    parser.add_argument("--input-dir", required=True, help="Folder containing .vsdx files")
    parser.add_argument("--output-dir", required=True, help="Folder for rendered PNG files")
    parser.add_argument("--timeout", type=int, default=90, help="Seconds allowed per VSDX")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if sys.platform != "win32":
        print("ERROR: Visio COM rendering requires Windows.", file=sys.stderr)
        return 2

    files = sorted(input_dir.glob("*.vsdx"))
    print(f"VSDX_COUNT={len(files)}")
    failures = []
    for vsdx in files:
        print(f"RENDER {vsdx.name}", flush=True)
        status, rendered = render_with_timeout(vsdx, output_dir, args.timeout)
        if status == "ok":
            for item in rendered:
                print(f"  -> {Path(item).name}")
        else:
            failures.append({"file": vsdx.name, "status": status})
            print(f"  !! {status}")

    if failures:
        print("FAILURES:")
        for failure in failures:
            print(f"  {failure['file']}: {failure['status']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
