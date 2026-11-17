from __future__ import annotations

import math
import unittest

from common import ROOT, read_csv, read_json


class DataIntegrityAudit(unittest.TestCase):
    def test_manifest_counts_and_numeric_integrity(self):
        for org in ("org1", "org2", "org3", "org4"):
            with self.subTest(org=org):
                base = ROOT / "data" / f"synthetic_{org}"
                manifest = read_json(base / "curve_split_manifest.json")
                for split in ("train", "test_id", "test_ood"):
                    rows = read_csv(base / f"{split}.csv")
                    declared = manifest["splits"][split]
                    self.assertEqual(len(rows), declared["n_points"])
                    self.assertEqual(len({r["curve_id"] for r in rows}), declared["n_curves"])
                    for row in rows:
                        vals = [float(row[k]) for k in ("VGS_abs_V", "VDS_abs_V", "log10_ID_abs_A")]
                        self.assertTrue(all(math.isfinite(v) for v in vals))
                        self.assertGreaterEqual(vals[0], 0)
                        self.assertGreaterEqual(vals[1], 0)
                # The manifest declares nonoverlapping curve allocation.
                ids = [set(r["curve_id"] for r in read_csv(base / f"{s}.csv")) for s in ("train", "test_id", "test_ood")]
                self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])

    def test_no_duplicate_rows_or_exact_xy_overlap_across_splits(self):
        for org in ("org1", "org2", "org3", "org4"):
            frames = [read_csv(ROOT / "data" / f"synthetic_{org}" / f"{s}.csv") for s in ("train", "test_id", "test_ood")]
            keys = []
            for frame in frames:
                full = [tuple(row.values()) for row in frame]
                self.assertEqual(len(full), len(set(full)))
                keys.append({(r["VGS_abs_V"], r["VDS_abs_V"]) for r in frame})
            self.assertFalse(keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])


if __name__ == "__main__":
    unittest.main()
