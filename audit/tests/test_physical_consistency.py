from __future__ import annotations

import math
import unittest

from common import ROOT, read_csv


class PhysicalConsistencyAudit(unittest.TestCase):
    def test_voltage_magnitude_and_log_current_contract(self):
        for org in ("org1", "org2", "org3", "org4"):
            for split in ("train", "test_id", "test_ood"):
                for row in read_csv(ROOT / "data" / f"synthetic_{org}" / f"{split}.csv"):
                    vgs, vds, log_id = (float(row[k]) for k in ("VGS_abs_V", "VDS_abs_V", "log10_ID_abs_A"))
                    self.assertGreaterEqual(vgs, 0)
                    self.assertGreaterEqual(vds, 0)
                    self.assertTrue(math.isfinite(log_id))
                    self.assertGreater(log_id, -20)
                    self.assertLess(log_id, 5)

    def test_exported_current_columns_match_log_scale(self):
        for path in (ROOT / "results").glob("*_best_candidate_holdout.csv"):
            rows = read_csv(path)
            if not rows or "ID_observado_A" not in rows[0]:
                continue
            for row in rows:
                self.assertAlmostEqual(math.log10(float(row["ID_observado_A"])), float(row["log10_ID_abs_A"]), places=8)
                self.assertAlmostEqual(math.log10(float(row["ID_predito_A"])), float(row["log10_ID_predito_A"]), places=8)


if __name__ == "__main__":
    unittest.main()
