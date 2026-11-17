"""Run independent, low-cost audit checks without modifying source artifacts."""
from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
OUT = AUDIT / "outputs"


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def metrics(path: Path):
    data = rows(path)
    y = [float(r["log10_ID_abs_A"]) for r in data]
    p = [float(r["log10_ID_predito_A"]) for r in data]
    residual = [a - b for a, b in zip(y, p)]
    mse = sum(e * e for e in residual) / len(residual)
    mae = sum(abs(e) for e in residual) / len(residual)
    denom = sum((v - sum(y) / len(y)) ** 2 for v in y)
    return {"n": len(data), "MSE_log10": mse, "RMSE_log10": math.sqrt(mse), "MAE_log10": mae,
            "R2": 1 - sum(e * e for e in residual) / denom if denom else None}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    suite = unittest.defaultTestLoader.discover(str(AUDIT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    checks = {
        "passed": result.testsRun - len(result.failures) - len(result.errors),
        "failed": len(result.failures), "errors": len(result.errors),
        "test_count": result.testsRun,
        "details": [f"FAIL: {t.id()} — {msg.splitlines()[0]}" for t, msg in result.failures + result.errors],
    }
    holdouts = {p.name: metrics(p) for p in sorted((ROOT / "results").glob("*_best_candidate_holdout.csv"))}
    splits = {}
    for org in ("org1", "org2", "org3", "org4"):
        splits[org] = {}
        for split in ("train", "test_id", "test_ood"):
            frame = rows(ROOT / "data" / f"synthetic_{org}" / f"{split}.csv")
            splits[org][split] = {"n_points": len(frame), "n_curves": len({r["curve_id"] for r in frame}),
                                  "n_duplicate_rows": len(frame) - len({tuple(r.values()) for r in frame}),
                                  "n_duplicate_xy": len(frame) - len({(r["VGS_abs_V"], r["VDS_abs_V"]) for r in frame})}
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "read-only checks of persisted CSV/JSON, Python source and candidate syntax; no model search rerun",
        "tests": checks,
        "synthetic_splits": splits,
        "recalculated_holdout_metrics_from_persisted_predictions": holdouts,
        "limitations": ["No LLM server, prompts, model identifier or inference parameters were persisted for the historical runs.",
                        "No independent physical simulator or experimental validation was run.",
                        "Metric recalculation covers exported best-candidate holdout predictions; full test_id prediction arrays are absent."],
    }
    (OUT / "audit_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    log = {"timestamp_utc": payload["generated_utc"], "command": "python OTFT-SR/audit/run_audit_checks.py",
           "exit_code": 0 if result.wasSuccessful() else 1, "tests_run": checks["test_count"],
           "passed": checks["passed"], "failed": checks["failed"], "errors": checks["errors"],
           "experiment_rerun": False, "production_code_modified": False}
    (OUT / "audit_execution_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
