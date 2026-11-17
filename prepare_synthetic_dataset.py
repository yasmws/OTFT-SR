"""Cria partições por curva para o corpus sintético p-OTFT da Org1."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_RUN_ROOT = WORKSPACE_ROOT / "Model_OTFT/synthetic_inference_runs/Org1"
SYNTHETIC_CURVE_ROOT = SYNTHETIC_RUN_ROOT / "Org1"
INDEX_PATH = SYNTHETIC_RUN_ROOT / "index_generated.json"
OUTPUT_ROOT = Path(__file__).resolve().parent / "data/synthetic_org1"
POINTS_PER_CURVE = 150

# O treino cobre as duas famílias de curva e as polarizações baixas e altas.
# test_id reserva polarizações intermediárias, mas ainda dentro da faixa treinada.
# test_ood reserva polarizações fixas de maior módulo (56–60 V), ausentes do treino.
CURVE_SPLIT_POLICY: dict[str, dict[str, tuple[float, ...]]] = {
    "train": {
        "transfer": (4.0, 8.0, 12.0, 52.0, 54.0),
        "output": (32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 54.0),
    },
    "test_id": {
        "transfer": (6.0, 10.0),
        "output": (34.0, 38.0, 42.0, 46.0, 50.0),
    },
    "test_ood": {
        "transfer": (56.0, 58.0, 60.0),
        "output": (56.0, 58.0, 60.0),
    },
}


def load_index(index_path: Path) -> list[dict[str, Any]]:
    with index_path.open("r", encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list):
        raise ValueError(f"O índice deve ser uma lista de registros: {index_path}")
    return records


def _indexed_curve_path(record: dict[str, Any], curve_root: Path) -> Path:
    """Rebaseia o caminho absoluto registrado no índice para o workspace atual."""
    indexed_path = Path(record["csv_path"])
    curve_family = indexed_path.parent.name
    if curve_family not in {"transfer", "output"}:
        raise ValueError(f"Família de curva não reconhecida no índice: {indexed_path}")
    return curve_root / "tipo_p" / curve_family / indexed_path.name


def _current_scale_to_ampere(csv_path: Path) -> tuple[float, str]:
    """Lê a escala da corrente sintética nos metadados NPZ da curva."""
    npz_path = csv_path.with_suffix(".npz")
    if not npz_path.is_file():
        raise FileNotFoundError(f"Arquivo NPZ com metadados de unidade ausente: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as archive:
        if "meta" not in archive.files:
            raise ValueError(f"Metadados ausentes em {npz_path}.")
        metadata = archive["meta"].item()
    settings = metadata.get("settings", {})
    current_unit = str(settings.get("current_typic", "")).strip()
    unit_to_ampere = {"A": 1.0, "mA": 1e-3, "uA": 1e-6, "nA": 1e-9, "pA": 1e-12}
    if current_unit not in unit_to_ampere:
        raise ValueError(f"Unidade current_typic ausente ou desconhecida em {npz_path}: {current_unit!r}")
    return unit_to_ampere[current_unit], current_unit


def _read_curve(
    csv_path: Path,
    points_per_curve: int,
    current_scale_to_ampere: float,
) -> list[dict[str, float]]:
    with csv_path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required_columns = {"VGS", "VDS", "ID"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"Cabeçalho inválido em {csv_path}; esperado VGS,VDS,ID.")
        source_rows = list(reader)

    if len(source_rows) < points_per_curve:
        raise ValueError(f"{csv_path} possui {len(source_rows)} linhas; esperado >= {points_per_curve}.")
    # Amostragem determinística e aproximadamente uniforme ao longo de cada curva.
    selected_indices = [
        round(index * (len(source_rows) - 1) / (points_per_curve - 1))
        for index in range(points_per_curve)
    ]
    selected_rows: list[dict[str, float]] = []
    for source_index in selected_indices:
        row = source_rows[source_index]
        try:
            gate_voltage_abs = abs(float(row["VGS"]))
            drain_voltage_abs = abs(float(row["VDS"]))
            drain_current_abs = abs(float(row["ID"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Dados inválidos em {csv_path}, linha {source_index + 2}.") from error
        if not all(math.isfinite(value) for value in (gate_voltage_abs, drain_voltage_abs, drain_current_abs)):
            raise ValueError(f"Valor não finito em {csv_path}, linha {source_index + 2}.")
        if drain_current_abs <= 0:
            raise ValueError(f"ID nula ou negativa após módulo em {csv_path}, linha {source_index + 2}.")
        selected_rows.append({
            "VGS_abs_V": gate_voltage_abs,
            "VDS_abs_V": drain_voltage_abs,
            "log10_ID_abs_A": math.log10(drain_current_abs * current_scale_to_ampere),
        })
    return selected_rows


def build_splits(
    index_path: Path = INDEX_PATH,
    curve_root: Path = SYNTHETIC_CURVE_ROOT,
    output_root: Path = OUTPUT_ROOT,
    points_per_curve: int = POINTS_PER_CURVE,
) -> dict[str, Path]:
    if points_per_curve < 2:
        raise ValueError("points_per_curve deve ser >= 2.")

    index_records = load_index(index_path)
    records_by_curve: dict[tuple[str, float], dict[str, Any]] = {}
    for record in index_records:
        indexed_path = Path(record["csv_path"])
        family = indexed_path.parent.name
        if family not in {"transfer", "output"}:
            continue
        fixed_voltage = abs(float(record["fixed_voltage_simulated"]))
        key = (family, fixed_voltage)
        if key in records_by_curve:
            raise ValueError(f"Polarização duplicada no índice: {key}")
        records_by_curve[key] = record

    assigned_keys: set[tuple[str, float]] = set()
    manifest: dict[str, Any] = {
        "source_index": str(index_path),
        "source_curve_root": str(curve_root),
        "target": "log10(|ID| / A)",
        "raw_current_unit": "current_typic registrado no NPZ de cada curva",
        "current_conversion": "ID_A = abs(ID_csv) * escala(current_typic); a escala vem dos metadados NPZ.",
        "inputs": ["|VGS| (V)", "|VDS| (V)"],
        "points_per_curve": points_per_curve,
        "sampling": "Índices aproximadamente uniformes em toda a curva, arredondados deterministicamente.",
        "split_rationale": {
            "train": "Curvas transfer e output selecionadas em polarizações baixas e altas; cobertura das duas famílias.",
            "test_id": "Polarizações intermediárias omitidas do treino e contidas no intervalo coberto pelo treino.",
            "test_ood": "Curvas em polarizações nominais de 56, 58 e 60 V, acima do maior módulo fixo usado no treino (54 V).",
            "leakage_control": "A curva é a unidade de partição; todos os pontos de uma curva pertencem a exatamente um split.",
        },
        "splits": {},
    }
    output_paths: dict[str, Path] = {}

    for split_name, family_selection in CURVE_SPLIT_POLICY.items():
        split_rows: list[dict[str, float]] = []
        selected_curves: list[dict[str, Any]] = []
        for family, fixed_voltages in family_selection.items():
            for fixed_voltage in fixed_voltages:
                key = (family, fixed_voltage)
                if key in assigned_keys:
                    raise ValueError(f"Curva atribuída a mais de um split: {key}")
                record = records_by_curve.get(key)
                if record is None:
                    raise FileNotFoundError(f"Curva ausente no índice index_generated.json: {key}")
                csv_path = _indexed_curve_path(record, curve_root)
                if not csv_path.is_file():
                    raise FileNotFoundError(f"CSV listado no índice não encontrado: {csv_path}")
                current_scale_to_ampere, current_unit = _current_scale_to_ampere(csv_path)
                curve_rows = _read_curve(csv_path, points_per_curve, current_scale_to_ampere)
                curve_id = f"{family}_fixed_{fixed_voltage:g}V"
                for curve_row in curve_rows:
                    curve_row["curve_id"] = curve_id
                    curve_row["curve_family"] = family
                    curve_row["fixed_voltage_abs_V"] = fixed_voltage
                    curve_row["sweep_voltage_abs_V"] = (
                        curve_row["VGS_abs_V"] if family == "transfer" else curve_row["VDS_abs_V"]
                    )
                split_rows.extend(curve_rows)
                selected_curves.append({
                    "family": family,
                    "fixed_voltage_simulated_V": float(record["fixed_voltage_simulated"]),
                    "csv_file": csv_path.name,
                    "curve_points_used": len(curve_rows),
                    "current_typic": current_unit,
                    "current_scale_to_ampere": current_scale_to_ampere,
                    "indexed_parameters": record.get("params", {}),
                })
                assigned_keys.add(key)

        output_root.mkdir(parents=True, exist_ok=True)
        output_path = output_root / f"{split_name}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "VGS_abs_V", "VDS_abs_V", "log10_ID_abs_A",
                    "curve_id", "curve_family", "fixed_voltage_abs_V", "sweep_voltage_abs_V",
                ],
            )
            writer.writeheader()
            writer.writerows(split_rows)
        output_paths[split_name] = output_path
        manifest["splits"][split_name] = {
            "csv_file": output_path.name,
            "n_curves": len(selected_curves),
            "n_points": len(split_rows),
            "curves": selected_curves,
        }

    expected_keys = {
        (family, voltage)
        for family_selection in CURVE_SPLIT_POLICY.values()
        for family, voltages in family_selection.items()
        for voltage in voltages
    }
    if assigned_keys != expected_keys:
        raise AssertionError("Atribuição de curvas incompleta ou divergente da política documentada.")

    manifest_path = output_root / "curve_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_paths["manifest"] = manifest_path
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--curve-root", type=Path, default=SYNTHETIC_CURVE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--points-per-curve", type=int, default=POINTS_PER_CURVE)
    arguments = parser.parse_args()
    output_paths = build_splits(
        index_path=arguments.index,
        curve_root=arguments.curve_root,
        output_root=arguments.output_root,
        points_per_curve=arguments.points_per_curve,
    )
    for split_name, output_path in output_paths.items():
        print(f"{split_name}: {output_path}")


if __name__ == "__main__":
    main()
