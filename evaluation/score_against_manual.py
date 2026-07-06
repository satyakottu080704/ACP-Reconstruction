#!/usr/bin/env python3
"""
Score the pipeline against the team's manual plans (structural accuracy).

Pairs an INPUT sketch (pulled from AlphaTracker into output/_at_sketches/) with
the manual-plan answer key (ground_truth/manual_plans_truth.json), runs the box
pipeline on the sketch, and compares the EXTRACTED rooms/samples to the answer
key. Uses STRUCTURAL metrics only (count, label, number, sample, floor) - the
manual plan and the sketch are different coordinate spaces, so bbox-IoU is N/A.

Usage:
    python evaluation/score_against_manual.py
    python evaluation/score_against_manual.py N-104621 N-105325
    python evaluation/score_against_manual.py --gate --max-projects 5
    python evaluation/score_against_manual.py --gate --json
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

TRUTH = json.load(open(ROOT / "ground_truth" / "manual_plans_truth.json", encoding="utf-8"))["projects"]
SKETCH_DIR = ROOT / "output" / "_at_sketches"


def _norm(s):
    return "".join(c for c in str(s or "").lower() if c.isalnum())


def _multiset_rate(pred, gt):
    """Fraction of gt items matched in pred (order-free, multiset)."""
    pc = Counter(_norm(x) for x in pred if _norm(x))
    gc = Counter(_norm(x) for x in gt if _norm(x))
    if not gc:
        return 1.0 if not pc else 0.0
    matched = sum(min(pc[k], gc[k]) for k in gc)
    return matched / sum(gc.values())


def available_projects():
    """Manual-truth projects that have a matching local input sketch."""
    return [p for p in TRUTH if (SKETCH_DIR / f"{p}_sketch.jpg").exists()]


def score_one(pn):
    sketch = SKETCH_DIR / f"{pn}_sketch.jpg"
    if not sketch.exists():
        return {"project": pn, "error": "no sketch"}
    truth = TRUTH.get(pn)
    if not truth:
        return {"project": pn, "error": "no answer key"}

    from pipeline import process_sketch

    out = ROOT / "output" / "visio" / f"{pn}_eval.vsdx"
    _, plan = process_sketch(str(sketch), output_path=str(out))

    pred_labels = [r.label for r in plan.rooms]
    pred_numbers = [r.number for r in plan.rooms]
    pred_samples = [s.id for s in plan.samples]
    gt_labels = [r["label"] for r in truth["rooms"]]
    gt_numbers = [r["room_number"] for r in truth["rooms"]]
    gt_samples = truth.get("samples", [])

    return {
        "project": pn,
        "rooms_pred": len(plan.rooms),
        "rooms_gt": len(truth["rooms"]),
        "label_rate": round(_multiset_rate(pred_labels, gt_labels), 2),
        "number_rate": round(_multiset_rate(pred_numbers, gt_numbers), 2),
        "samples_pred": len(pred_samples),
        "samples_gt": len(gt_samples),
        "sample_rate": round(_multiset_rate(pred_samples, gt_samples), 2),
    }


def score_many(projects):
    rows = []
    for pn in projects:
        print(f"\n--- scoring {pn} ---", file=sys.stderr)
        try:
            rows.append(score_one(pn))
        except Exception as e:
            rows.append({"project": pn, "error": f"{type(e).__name__}: {e}"})
    return rows


def gate_failures(rows, *, min_label, min_number, min_sample, max_room_delta):
    failures = []
    for row in rows:
        project = row.get("project", "?")
        if row.get("error"):
            failures.append(f"{project}: {row['error']}")
            continue
        room_delta = abs(int(row["rooms_pred"]) - int(row["rooms_gt"]))
        if room_delta > max_room_delta:
            failures.append(f"{project}: room delta {room_delta} > {max_room_delta}")
        if row["label_rate"] < min_label:
            failures.append(f"{project}: label_rate {row['label_rate']} < {min_label}")
        if row["number_rate"] < min_number:
            failures.append(f"{project}: number_rate {row['number_rate']} < {min_number}")
        if row["samples_gt"] and row["sample_rate"] < min_sample:
            failures.append(f"{project}: sample_rate {row['sample_rate']} < {min_sample}")
    return failures


def _print_table(rows, failures, gate_enabled):
    print("\n" + "=" * 78)
    print(f"{'Project':<11}{'rooms p/gt':<12}{'label':<8}{'number':<8}{'samp p/gt':<11}{'sample':<8}")
    print("-" * 78)
    for r in rows:
        if r.get("error"):
            print(f"{r['project']:<11}ERROR: {r['error']}")
            continue
        print(
            f"{r['project']:<11}{str(r['rooms_pred']) + '/' + str(r['rooms_gt']):<12}"
            f"{r['label_rate']:<8}{r['number_rate']:<8}"
            f"{str(r['samples_pred']) + '/' + str(r['samples_gt']):<11}{r['sample_rate']:<8}"
        )
    ok = [r for r in rows if not r.get("error")]
    if ok:
        print("-" * 78)
        print(
            f"{'MEAN':<11}{'':<12}{sum(r['label_rate'] for r in ok) / len(ok):<8.2f}"
            f"{sum(r['number_rate'] for r in ok) / len(ok):<8.2f}{'':<11}"
            f"{sum(r['sample_rate'] for r in ok) / len(ok):<8.2f}"
        )
    if gate_enabled:
        print("-" * 78)
        if failures:
            print("GATE: FAIL")
            for failure in failures:
                print(f"  - {failure}")
        else:
            print("GATE: PASS")
    print("=" * 78)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score generated plans against manual ground truth.")
    parser.add_argument("projects", nargs="*", help="Project numbers to score; defaults to all local sketch/truth pairs.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of the text table.")
    parser.add_argument("--gate", action="store_true", help="Exit non-zero if any project misses the configured thresholds.")
    parser.add_argument("--max-projects", type=int, default=0, help="Limit project count for smoke/regression runs.")
    parser.add_argument("--min-label-rate", type=float, default=float(os.getenv("ACORN_EVAL_MIN_LABEL_RATE", "0.70")))
    parser.add_argument("--min-number-rate", type=float, default=float(os.getenv("ACORN_EVAL_MIN_NUMBER_RATE", "0.70")))
    parser.add_argument("--min-sample-rate", type=float, default=float(os.getenv("ACORN_EVAL_MIN_SAMPLE_RATE", "0.50")))
    parser.add_argument("--max-room-delta", type=int, default=int(os.getenv("ACORN_EVAL_MAX_ROOM_DELTA", "2")))
    args = parser.parse_args(argv)

    projects = args.projects or available_projects()
    if args.max_projects > 0:
        projects = projects[: args.max_projects]

    rows = score_many(projects)
    failures = []
    if args.gate and not projects:
        failures.append("no local sketch/truth pairs available")
    failures.extend(
        gate_failures(
            rows,
            min_label=args.min_label_rate,
            min_number=args.min_number_rate,
            min_sample=args.min_sample_rate,
            max_room_delta=args.max_room_delta,
        )
    )

    if args.json:
        print(
            json.dumps(
                {
                    "projects": projects,
                    "rows": rows,
                    "gate": {
                        "enabled": args.gate,
                        "thresholds": {
                            "min_label_rate": args.min_label_rate,
                            "min_number_rate": args.min_number_rate,
                            "min_sample_rate": args.min_sample_rate,
                            "max_room_delta": args.max_room_delta,
                        },
                        "passed": not failures,
                        "failures": failures,
                    },
                },
                indent=2,
            )
        )
    else:
        _print_table(rows, failures, args.gate)

    return 2 if args.gate and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
