from __future__ import annotations

import ast
import unittest

import numpy as np

from common import ROOT, read_json


class EquationValidationAudit(unittest.TestCase):
    def test_saved_llm_candidate_is_valid_python_and_has_finite_holdout_predictions(self):
        for path in (ROOT / "results").glob("*_synthetic_notebook_curves/model_specifications.json"):
            spec = read_json(path)
            candidate = spec.get("equacao_candidata_LLM_SR", {})
            source = candidate.get("fonte")
            self.assertTrue(source, path)
            tree = ast.parse(source)
            self.assertTrue(any(isinstance(n, ast.Return) for n in ast.walk(tree)))
            params = candidate.get("parametros_reajustados_treino", [])
            self.assertTrue(all(isinstance(x, (float, int)) for x in params))
            self.assertEqual(candidate.get("status"), "executado")

    def test_saved_equation_has_only_expected_input_names(self):
        allowed = {"parameters", "params", "gate_voltage_abs", "drain_voltage_abs", "np", "equation", "ndarray", "float"}
        for path in (ROOT / "results").glob("*_synthetic_notebook_curves/model_specifications.json"):
            source = read_json(path).get("equacao_candidata_LLM_SR", {}).get("fonte", "")
            if not source:
                continue
            names = {n.id for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Name)}
            unexpected = {name for name in names - allowed if len(name) != 1 or name not in "abcdefghijklmno"}
            self.assertFalse(unexpected, (path, unexpected))

    def test_saved_equation_reproduces_exported_holdout_predictions(self):
        for spec_path in (ROOT / "results").glob("*_synthetic_notebook_curves/model_specifications.json"):
            with self.subTest(spec=spec_path.parent.name):
                spec = read_json(spec_path)
                entry = spec["equacao_candidata_LLM_SR"]
                tree = ast.parse(entry["fonte"])
                forbidden = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.With, ast.Try, ast.Lambda)
                self.assertFalse(any(isinstance(n, forbidden) for n in ast.walk(tree)))
                namespace = {"np": np, "__builtins__": {}}
                exec(compile(tree, str(spec_path), "exec"), namespace)
                rows = __import__("common").read_csv(
                    ROOT / "results" / f"{spec['execucao']['run_name']}_best_candidate_holdout.csv"
                )
                X = np.array([[float(r["VGS_abs_V"]), float(r["VDS_abs_V"])] for r in rows])
                p = np.asarray(entry["parametros_reajustados_treino"], dtype=float)
                predicted = np.asarray(namespace["equation"](X[:, 0], X[:, 1], p), dtype=float).reshape(-1)
                stored = np.array([float(r["log10_ID_predito_A"]) for r in rows])
                self.assertEqual(predicted.shape, stored.shape)
                self.assertTrue(np.isfinite(predicted).all())
                np.testing.assert_allclose(predicted, stored, rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
