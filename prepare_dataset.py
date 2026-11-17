"""Adapta a curva de transferência Org1 p-OTFT ao formato LLM-SR."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = WORKSPACE_ROOT / "Model_OTFT/datas/Org1/tipo p/org1_50VDS.csv"
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "data"


def load_transfer_curve(source_csv: Path) -> list[tuple[float, float]]:
    """Retorna pares (|VGS|, log10(|ID|/A)) da varredura em VDS nominal fixo."""
    observations: list[tuple[float, float]] = []
    with source_csv.open("r", newline="", encoding="utf-8") as source_file:
        reader = csv.DictReader(source_file)
        if reader.fieldnames is None or not {"VGS", "VDS", "ID"}.issubset(reader.fieldnames):
            raise ValueError("Cabeçalho esperado: VGS,VDS,ID (ou VDS,VGS,ID).")
        for row_number, row in enumerate(reader, start=2):
            try:
                gate_voltage = abs(float(row["VGS"].strip()))
                drain_voltage = abs(float(row["VDS"].strip()))
                drain_current = abs(float(row["ID"].strip()))
            except (AttributeError, ValueError, TypeError):
                # O arquivo de referência contém a linha de unidades logo após o cabeçalho.
                if row_number == 2:
                    continue
                raise ValueError(f"Linha {row_number} inválida em {source_csv}.")
            if not all(map(math.isfinite, (gate_voltage, drain_voltage, drain_current))) or drain_current <= 0:
                raise ValueError(f"Observação não finita ou ID nula na linha {row_number}.")
            if not math.isclose(drain_voltage, 50.0, abs_tol=0.2):
                raise ValueError(f"VDS não corresponde à curva nominal de 50 V na linha {row_number}.")
            observations.append((gate_voltage, math.log10(drain_current)))
    observations.sort(key=lambda observation: observation[0])
    if len(observations) < 10:
        raise ValueError("A curva contém menos de 10 observações válidas.")
    return observations


def write_llmsr_csv(path: Path, observations: list[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["VGS_abs_V", "log10_ID_abs_A"])
        writer.writerows((f"{gate_voltage:.10g}", f"{log_current:.12g}") for gate_voltage, log_current in observations)


def prepare_dataset(source_csv: Path = SOURCE_CSV, output_directory: Path = OUTPUT_DIRECTORY) -> tuple[Path, Path]:
    observations = load_transfer_curve(source_csv)
    split_index = int(len(observations) * 0.8)
    if split_index <= 0 or split_index >= len(observations):
        raise ValueError("Partição treino/teste inválida.")
    train_path = output_directory / "train.csv"
    test_path = output_directory / "test_extrapolation.csv"
    write_llmsr_csv(train_path, observations[:split_index])
    write_llmsr_csv(test_path, observations[split_index:])
    return train_path, test_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_CSV)
    parser.add_argument("--output-directory", type=Path, default=OUTPUT_DIRECTORY)
    arguments = parser.parse_args()
    train_path, test_path = prepare_dataset(arguments.source, arguments.output_directory)
    print(f"Treino: {train_path}")
    print(f"Teste de extrapolação: {test_path}")


if __name__ == "__main__":
    main()
