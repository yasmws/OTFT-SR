from __future__ import annotations

import csv
import math
import unittest

from common import ROOT, read_csv


class MetricRecalculationAudit(unittest.TestCase):
    def test_recalculate_persisted_holdout_predictions(self):
        files = sorted((ROOT / "results").glob("*_best_candidate_holdout.csv"))
        self.assertGreaterEqual(len(files), 5)
        for path in files:
            with self.subTest(file=path.name):
                rows = read_csv(path)
                residuals = [float(r["log10_ID_abs_A"]) - float(r["log10_ID_predito_A"]) for r in rows]
                ys = [float(r["log10_ID_abs_A"]) for r in rows]
                self.assertGreater(len(rows), 0)
                self.assertTrue(all(math.isfinite(x) for x in residuals + ys))
                mse = sum(e * e for e in residuals) / len(rows)
                mae = sum(abs(e) for e in residuals) / len(rows)
                r2 = 1 - sum(e * e for e in residuals) / sum((y - sum(ys) / len(ys)) ** 2 for y in ys)
                self.assertTrue(math.isfinite(mse) and math.isfinite(mae) and math.isfinite(r2))
                # The exported current columns must agree with log10 current.
                if "ID_observado_A" in rows[0]:
                    for row in rows:
                        y = float(row["log10_ID_abs_A"])
                        current = float(row["ID_observado_A"])
                        self.assertAlmostEqual(math.log10(current), y, places=8)


if __name__ == "__main__":
    unittest.main()
