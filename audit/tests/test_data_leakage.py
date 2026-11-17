from __future__ import annotations

import unittest

from common import ROOT, read_csv


class LeakageAudit(unittest.TestCase):
    def test_curve_ids_and_exact_input_pairs_are_disjoint(self):
        for org in ("org1", "org2", "org3", "org4"):
            with self.subTest(org=org):
                frames = [read_csv(ROOT / "data" / f"synthetic_{org}" / f"{s}.csv") for s in ("train", "test_id", "test_ood")]
                curve_sets = [{r["curve_id"] for r in f} for f in frames]
                xy_sets = [{(r["VGS_abs_V"], r["VDS_abs_V"]) for r in f} for f in frames]
                for i in range(3):
                    for j in range(i + 1, 3):
                        self.assertFalse(curve_sets[i] & curve_sets[j])
                        self.assertFalse(xy_sets[i] & xy_sets[j])

    def test_evaluation_splits_are_not_passed_to_search_runner(self):
        # Static check: runners construct the pipeline dataset from train.csv only.
        for name in ("run_synthetic_experiment.py", "run_synthetic_org2.py", "run_synthetic_org3.py", "run_synthetic_org4.py"):
            source = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn('"train"', source)
            self.assertIn('"data"', source)
            self.assertIn('"inputs"', source)
            self.assertIn('"train"', source)
            self.assertNotIn('"test_ood"', source)
            self.assertNotIn('"test_id"', source)


if __name__ == "__main__":
    unittest.main()
