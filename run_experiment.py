"""Executa o pipeline LLM-SR oficial sobre a curva p-OTFT adaptada."""

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
from prepare_dataset import prepare_dataset  # noqa: E402


class DeterministicSmokeLLM(sampler.LLM):
    """Amostrador determinístico para verificar a integração sem serviço externo."""

    def draw_samples(self, prompt: str, config: object | None = None) -> list[str]:
        del prompt, config
        return ["    return parameters[0] + parameters[1] * gate_voltage_abs + parameters[2] * gate_voltage_abs**2"]


def main() -> None:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--max-samples", type=int, default=100)
    argument_parser.add_argument("--seed", type=int, default=7)
    argument_parser.add_argument("--smoke", action="store_true", help="Usa hipótese determinística, sem servidor LLM.")
    argument_parser.add_argument("--log-directory", type=Path, default=IMPLEMENTATION_ROOT / "logs/org1_tipo_p")
    arguments = argument_parser.parse_args()
    if arguments.max_samples < 2:
        raise ValueError("--max-samples deve ser pelo menos 2 para avaliar uma hipótese gerada.")

    train_path, _ = prepare_dataset()
    frame = pd.read_csv(train_path)
    inputs = frame[["VGS_abs_V"]].to_numpy(dtype=np.float64)
    outputs = frame["log10_ID_abs_A"].to_numpy(dtype=np.float64)
    dataset = {"data": {"inputs": inputs, "outputs": outputs}}

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    sampler.Sampler._global_samples_nums = 1
    class_config = config.ClassConfig(
        llm_class=DeterministicSmokeLLM if arguments.smoke else sampler.LocalLLM,
        sandbox_class=evaluator.LocalSandbox,
    )
    experiment_config = config.Config(problem_name="org1_otft_transfer")
    specification_path = IMPLEMENTATION_ROOT / "specifications/otft_transfer_numpy.txt"
    specification = specification_path.read_text(encoding="utf-8")
    arguments.log_directory.mkdir(parents=True, exist_ok=True)

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
