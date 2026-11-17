"""Runs LLM-SR on the Org3 synthetic p-OTFT curves."""
from __future__ import annotations
import argparse,random,sys
from pathlib import Path
import numpy as np,pandas as pd
WORKSPACE_ROOT=Path(__file__).resolve().parents[1]; IMPLEMENTATION_ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(WORKSPACE_ROOT/"LLM-SR"))
from llmsr import config,evaluator,pipeline,sampler
from prepare_synthetic_org3 import OUTPUT_ROOT,build_splits
class DeterministicSmokeLLM(sampler.LLM):
    def draw_samples(self,prompt:str,config:object|None=None)->list[str]:
        del prompt,config
        return ["    return parameters[0] + parameters[1] * gate_voltage_abs + parameters[2] * drain_voltage_abs + parameters[3] * gate_voltage_abs * drain_voltage_abs"]
def main()->None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--max-samples",type=int,default=100); parser.add_argument("--seed",type=int,default=7); parser.add_argument("--smoke",action="store_true"); parser.add_argument("--log-directory",type=Path,default=IMPLEMENTATION_ROOT/"logs/org3_synthetic"); args=parser.parse_args()
    if args.max_samples<2: raise ValueError("--max-samples must be >= 2")
    paths=build_splits(); frame=pd.read_csv(paths["train"]); columns=["VGS_abs_V","VDS_abs_V"]
    dataset={"data":{"inputs":frame[columns].to_numpy(dtype=np.float64),"outputs":frame["log10_ID_abs_A"].to_numpy(dtype=np.float64)}}
    random.seed(args.seed); np.random.seed(args.seed); sampler.Sampler._global_samples_nums=1
    classes=config.ClassConfig(llm_class=DeterministicSmokeLLM if args.smoke else sampler.LocalLLM,sandbox_class=evaluator.LocalSandbox)
    experiment=config.Config(problem_name="org3_otft_synthetic")
    specification=(IMPLEMENTATION_ROOT/"specifications/otft_synthetic_org3_numpy.txt").read_text(encoding="utf-8")
    args.log_directory.mkdir(parents=True,exist_ok=True); print(f"Split manifest: {paths['manifest']}; training shape: {dataset['data']['inputs'].shape}")
    pipeline.main(specification=specification,inputs=dataset,config=experiment,max_sample_nums=args.max_samples,class_config=classes,log_dir=str(args.log_directory))
if __name__=="__main__": main()
