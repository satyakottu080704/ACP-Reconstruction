import os

import pytest

from evaluation import score_against_manual


def test_manual_truth_has_regression_projects():
    assert len(score_against_manual.TRUTH) >= 5


@pytest.mark.skipif(
    os.getenv("ACORN_RUN_EVAL_GATE") != "1",
    reason="set ACORN_RUN_EVAL_GATE=1 to run the expensive sketch/AI regression gate",
)
def test_manual_eval_gate_smoke():
    projects = score_against_manual.available_projects()
    if not projects:
        pytest.skip("no local output/_at_sketches/*_sketch.jpg files available")

    limit = int(os.getenv("ACORN_EVAL_MAX_PROJECTS", "3"))
    rows = score_against_manual.score_many(projects[:limit])
    failures = score_against_manual.gate_failures(
        rows,
        min_label=float(os.getenv("ACORN_EVAL_MIN_LABEL_RATE", "0.70")),
        min_number=float(os.getenv("ACORN_EVAL_MIN_NUMBER_RATE", "0.70")),
        min_sample=float(os.getenv("ACORN_EVAL_MIN_SAMPLE_RATE", "0.50")),
        max_room_delta=int(os.getenv("ACORN_EVAL_MAX_ROOM_DELTA", "2")),
    )
    assert not failures
