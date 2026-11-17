"""Executa LLM-SR nas curvas sintéticas p-OTFT da Org1."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_ROOT = Path(__file__).resolve().parent
LLM_SR_ROOT = WORKSPACE_ROOT / "LLM-SR"
sys.path.insert(0, str(LLM_SR_ROOT))

from llmsr import config, evaluator, pipeline, sampler  # noqa: E402
from prepare_synthetic_dataset import OUTPUT_ROOT, build_splits  # noqa: E402


class DeterministicSmokeLLM(sampler.LLM):
    """Hipótese polinomial determinística para verificar a integração do pipeline."""

    def draw_samples(self, prompt: str, config: object | None = None) -> list[str]:
        del prompt, config
        return [
            "    return parameters[0] + parameters[1] * gate_voltage_abs + "
            "parameters[2] * drain_voltage_abs + parameters[3] * gate_voltage_abs * drain_voltage_abs"
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--smoke", action="store_true", help="Usa amostrador determinístico, sem servidor LLM.")
    parser.add_argument(
        "--log-directory",
        type=Path,
        default=IMPLEMENTATION_ROOT / "logs/synthetic_org1",
    )
    arguments = parser.parse_args()
    if arguments.max_samples < 2:
        raise ValueError("--max-samples deve ser pelo menos 2.")

    split_paths = build_splits()
    train_frame = pd.read_csv(split_paths["train"])
    input_columns = ["VGS_abs_V", "VDS_abs_V"]
    inputs = train_frame[input_columns].to_numpy(dtype=np.float64)
    outputs = train_frame["log10_ID_abs_A"].to_numpy(dtype=np.float64)
    dataset = {"data": {"inputs": inputs, "outputs": outputs}}

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    sampler.Sampler._global_samples_nums = 1
    class_config = config.ClassConfig(
        llm_class=DeterministicSmokeLLM if arguments.smoke else sampler.LocalLLM,
        sandbox_class=evaluator.LocalSandbox,
    )
    experiment_config = config.Config(problem_name="org1_otft_synthetic")
    specification_path = IMPLEMENTATION_ROOT / "specifications/otft_synthetic_numpy.txt"
    specification = specification_path.read_text(encoding="utf-8")
    arguments.log_directory.mkdir(parents=True, exist_ok=True)

    print(f"Manifest de partição: {split_paths['manifest']}")
    print(f"Pontos de treino: {len(train_frame)}; entrada bivariada: {inputs.shape}")
    pipeline.main(
        specification=specification,
        inputs=dataset,
        config=experiment_config,
        max_sample_nums=arguments.max_samples,
        class_config=class_config,
        log_dir=str(arguments.log_directory),
    )


if __name__ == "__main__":
    main()
