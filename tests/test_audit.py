from copy import deepcopy

import numpy as np
import pytest
from test_transcriber import make_transcriber

from specvoice.audit import audit_report
from specvoice.compare import run_comparison


def report_fixture():
    engine = make_transcriber()
    sample = {
        "id": "toy",
        "array": np.zeros(800, dtype=np.float32),
        "rate": 16000,
        "reference": "2 3 4",
    }
    report = run_comparison(engine, [sample], repetitions=2, warmup_runs=0)
    report.update(
        config={"modes": list(report["summary"]), "repetitions": 2},
        environment={"git_commit": "test"},
    )
    return report


def test_audit_checks_per_run_digests_and_metrics():
    result = audit_report(report_fixture())
    assert result["status"] == "passed"
    assert result["token_digest_checks"] == 6


@pytest.mark.parametrize("tamper", ["digest", "latency", "sample_count"])
def test_audit_rejects_inconsistent_evidence(tamper):
    report = deepcopy(report_fixture())
    if tamper == "digest":
        report["samples"][0]["runs"]["optimized"][0]["token_sha256"] = "wrong"
    elif tamper == "latency":
        report["summary"]["optimized"]["p50_request_ms"] += 20
    else:
        report["sample_count"] += 1
    with pytest.raises(ValueError):
        audit_report(report)
