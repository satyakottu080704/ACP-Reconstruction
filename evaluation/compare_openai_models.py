#!/usr/bin/env python3
"""Compare OpenAI vision extraction models on Acorn plan images.

This runs the production layout extractor with two model names, writes one JSON
report and one CSV summary, and gives a practical recommendation based on
extraction completeness. It does not render Visio files.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

from utils.layout_extractor import extract_floor_plan_layout

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
GENERIC_ROOM_RE = re.compile(r"^(room|space|area|unknown|unnamed)(\s*\d+)?$", re.I)


def _project_from_name(path: Path) -> str:
    m = re.search(r"N-?\d{5,}", path.name, re.I)
    if not m:
        return path.stem
    v = m.group(0).upper()
    return v if "-" in v else f"N-{v[1:]}"


def _iter_images(input_dir: Path) -> Iterable[Path]:
    for p in sorted(input_dir.iterdir(), key=lambda x: x.name.lower()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and re.search(r"N-?\d{5,}", p.name, re.I):
            yield p


def _select_images(input_dir: Path, explicit: List[str], limit: int) -> List[Path]:
    if explicit:
        out = []
        for item in explicit:
            p = Path(item)
            if not p.is_absolute():
                p = input_dir / item
            if not p.is_file():
                raise FileNotFoundError(f"Image not found: {p}")
            out.append(p)
        return out[:limit]
    return list(_iter_images(input_dir))[:limit]


def _safe_num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _summarize(layout: Dict[str, Any], elapsed_s: float) -> Dict[str, Any]:
    rooms = layout.get("rooms", []) or []
    samples = layout.get("samples", []) or []
    walls = layout.get("walls", []) or []
    doors = layout.get("doors", []) or []
    windows = layout.get("windows", []) or []

    names = [str(r.get("name", "")).strip() for r in rooms]
    named = [n for n in names if n and not GENERIC_ROOM_RE.match(n)]
    generic = [n for n in names if not n or GENERIC_ROOM_RE.match(n)]
    numbered = [r for r in rooms if str(r.get("room_number") or r.get("number") or "").strip()]
    acm = [r for r in rooms if str(r.get("type", "")).lower() == "acm"]
    no_access = [r for r in rooms if str(r.get("type", "")).lower() == "no_access"]
    loft = [r for r in rooms if "loft" in str(r.get("name", "")).lower() or "attic" in str(r.get("name", "")).lower()]

    room_area = sum(max(0.0, _safe_num(r.get("w"))) * max(0.0, _safe_num(r.get("h"))) for r in rooms)
    duplicate_names = max(0, len(names) - len(set(n.lower() for n in names if n)))

    # Heuristic only. Real decision still needs visual review / ground truth.
    score = 0.0
    score += min(len(rooms), 15) * 2.0
    score += len(named) * 3.0
    score += len(numbered) * 1.5
    score += len(samples) * 4.0
    score += len(loft) * 2.0
    score += min(len(walls), 40) * 0.25
    score += min(len(doors), 15) * 0.5
    score -= len(generic) * 2.0
    score -= duplicate_names * 1.0
    if rooms and room_area < 10000:
        score -= 5.0

    return {
        "ok": True,
        "elapsed_s": round(elapsed_s, 2),
        "rooms": len(rooms),
        "named_rooms": len(named),
        "generic_rooms": len(generic),
        "numbered_rooms": len(numbered),
        "samples": len(samples),
        "acm_rooms": len(acm),
        "no_access_rooms": len(no_access),
        "loft_rooms": len(loft),
        "walls": len(walls),
        "doors": len(doors),
        "windows": len(windows),
        "duplicate_names": duplicate_names,
        "heuristic_score": round(score, 2),
    }


def _clear_dead_local_proxy() -> None:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        value = os.environ.get(name, "")
        if "127.0.0.1:9" in value or "localhost:9" in value:
            os.environ.pop(name, None)


def _run_one(image: Path, model: str) -> Dict[str, Any]:
    os.environ["OPENAI_VISION_MODEL"] = model
    started = time.time()
    try:
        layout = extract_floor_plan_layout(str(image))
        summary = _summarize(layout, time.time() - started)
        return {"summary": summary, "layout": layout, "error": ""}
    except Exception as exc:
        return {
            "summary": {
                "ok": False,
                "elapsed_s": round(time.time() - started, 2),
                "rooms": 0,
                "named_rooms": 0,
                "generic_rooms": 0,
                "numbered_rooms": 0,
                "samples": 0,
                "acm_rooms": 0,
                "no_access_rooms": 0,
                "loft_rooms": 0,
                "walls": 0,
                "doors": 0,
                "windows": 0,
                "duplicate_names": 0,
                "heuristic_score": 0.0,
            },
            "layout": {},
            "error": f"{type(exc).__name__}: {exc}",
        }


def _recommend(model_results: Dict[str, Dict[str, Any]]) -> str:
    scored = []
    for model, result in model_results.items():
        s = result["summary"]
        scored.append((float(s.get("heuristic_score", 0.0)), int(s.get("named_rooms", 0)), int(s.get("samples", 0)), model))
    scored.sort(reverse=True)
    if len(scored) < 2:
        return scored[0][3] if scored else "none"
    best, second = scored[0], scored[1]
    if best[0] - second[0] >= 5 or best[1] > second[1] or best[2] > second[2]:
        return best[3]
    return "tie"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare gpt-4o-mini and gpt-4o on plan extraction.")
    parser.add_argument("--input-dir", default="input")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--models", nargs="+", default=["gpt-4o-mini", "gpt-4o"])
    parser.add_argument("--images", nargs="*", default=[])
    parser.add_argument("--output-dir", default="evaluation/model_compare")
    parser.add_argument("--keep-dead-proxy", action="store_true", help="Do not clear known dead local proxy env values.")
    args = parser.parse_args()

    if not args.keep_dead_proxy:
        _clear_dead_local_proxy()

    images = _select_images(ROOT / args.input_dir, args.images, args.limit)
    if not images:
        raise SystemExit("No matching input images found")

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"openai_model_compare_{stamp}.json"
    csv_path = out_dir / f"openai_model_compare_{stamp}.csv"

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "models": args.models,
        "images": [str(p) for p in images],
        "results": [],
        "note": "Heuristic comparison only; final model choice needs visual review or ground truth.",
    }

    rows = []
    for image in images:
        print(f"\n=== {image.name} ===")
        model_results: Dict[str, Dict[str, Any]] = {}
        for model in args.models:
            print(f"  {model} ...", flush=True)
            result = _run_one(image, model)
            model_results[model] = result
            s = result["summary"]
            print(
                f"    ok={s['ok']} rooms={s['rooms']} named={s['named_rooms']} "
                f"samples={s['samples']} walls={s['walls']} doors={s['doors']} "
                f"score={s['heuristic_score']} time={s['elapsed_s']}s"
            )
            rows.append({
                "image": image.name,
                "project": _project_from_name(image),
                "model": model,
                **s,
                "error": result.get("error", ""),
            })
        rec = _recommend(model_results)
        print(f"  recommendation: {rec}")
        report["results"].append({
            "image": image.name,
            "project": _project_from_name(image),
            "recommendation": rec,
            "models": model_results,
        })

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    fieldnames = [
        "image", "project", "model", "ok", "elapsed_s", "rooms", "named_rooms",
        "generic_rooms", "numbered_rooms", "samples", "acm_rooms", "no_access_rooms",
        "loft_rooms", "walls", "doors", "windows", "duplicate_names",
        "heuristic_score", "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nJSON report: {json_path}")
    print(f"CSV summary: {csv_path}")


if __name__ == "__main__":
    main()