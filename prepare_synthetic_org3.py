"""Builds curve-level train, test_id, and test_ood data for synthetic p-OTFT Org3."""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from typing import Any
import numpy as np
WORKSPACE_ROOT=Path(__file__).resolve().parents[1]
ORG_NAME="Org3"
SOURCE_ROOT=WORKSPACE_ROOT/"Model_OTFT"/"synthetic_inference_runs"/ORG_NAME
CURVE_ROOT=SOURCE_ROOT/ORG_NAME
INDEX_PATH=SOURCE_ROOT/"index_generated.json"
OUTPUT_ROOT=Path(__file__).resolve().parent/"data"/f"synthetic_{ORG_NAME.lower()}"
POINTS_PER_CURVE=150
CURVE_SPLIT_POLICY={'train': {'transfer': ['transfer-neg4V_synth_000.csv', 'transfer-neg8V_synth_002.csv'], 'output': ['output-neg10V_synth_000.csv', 'output-neg14V_synth_002.csv', 'output-neg18V_synth_000.csv', 'output-neg22V_synth_002.csv', 'output-neg26V_synth_002.csv']}, 'test_id': {'transfer': ['transfer-neg6V_synth_001.csv'], 'output': ['output-neg12V_synth_001.csv', 'output-neg16V_synth_003.csv', 'output-neg18V_synth_004.csv', 'output-neg20V_synth_001.csv', 'output-neg22V_synth_000.csv', 'output-neg24V_synth_001.csv', 'output-neg24V_synth_003.csv', 'output-neg26V_synth_004.csv']}, 'test_ood': {'transfer': ['transfer-neg10V_synth_003.csv', 'transfer-neg12V_synth_004.csv'], 'output': ['output-neg28V_synth_003.csv', 'output-neg30V_synth_004.csv']}}
SPLIT_RATIONALE={"train": "Polarizações output baixas e curvas transfer em 4/8 V são selecionadas como âncoras.", "test_id": "Curvas output intercaladas e transfer em 6 V validam faixas internas/intermediárias; tensões nominais repetidas são diferenciadas pelo nome do arquivo.", "test_ood": "Output em 28/30 V e transfer em 10/12 V excedem os limites de treino por família. Curvas repetidas são atribuídas pelo arquivo."}
def load_index(path:Path)->list[dict[str,Any]]:
    records=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records,list): raise ValueError(f"Invalid index: {path}")
    return records
def curve_path(record:dict[str,Any])->Path:
    source=Path(record["csv_path"]); family=source.parent.name
    if family not in {"transfer","output"}: raise ValueError(f"Unknown curve family: {source}")
    return CURVE_ROOT/"tipo_p"/family/source.name
def current_scale(csv_path:Path)->tuple[float,str]:
    npz_path=csv_path.with_suffix(".npz")
    with np.load(npz_path,allow_pickle=True) as archive: meta=archive["meta"].item()
    unit=str(meta.get("settings",{}).get("current_typic","")).strip()
    factors={"A":1.0,"mA":1e-3,"uA":1e-6,"nA":1e-9,"pA":1e-12}
    if unit not in factors: raise ValueError(f"Unknown current unit {unit!r} in {npz_path}")
    return factors[unit],unit
def read_curve(path:Path,scale:float)->list[dict[str,float]]:
    with path.open(newline="",encoding="utf-8") as stream: rows=list(csv.DictReader(stream))
    if not rows or not {"VGS","VDS","ID"}.issubset(rows[0]): raise ValueError(f"Invalid data columns: {path}")
    indices=[round(i*(len(rows)-1)/(POINTS_PER_CURVE-1)) for i in range(POINTS_PER_CURVE)]
    result=[]
    for index in indices:
        row=rows[index]; vgs=abs(float(row["VGS"])); vds=abs(float(row["VDS"])); current=abs(float(row["ID"]))
        if current<=0 or not all(map(math.isfinite,(vgs,vds,current))): raise ValueError(f"Invalid measurement in {path}, row {index}")
        result.append({"VGS_abs_V":vgs,"VDS_abs_V":vds,"log10_ID_abs_A":math.log10(current*scale)})
    return result
def build_splits(index_path:Path=INDEX_PATH,output_root:Path=OUTPUT_ROOT)->dict[str,Path]:
    records=load_index(index_path)
    records_by_file={(Path(r["csv_path"]).parent.name,Path(r["csv_path"]).name):r for r in records}
    all_selected=[(fam,name) for split in CURVE_SPLIT_POLICY.values() for fam,names in split.items() for name in names]
    selected=set(all_selected)
    if len(all_selected)!=len(selected): raise ValueError("A curve file is assigned to multiple splits.")
    indexed={key for key in records_by_file if key[0] in {"transfer","output"}}
    if selected!=indexed:
        raise ValueError(f"Policy/index mismatch; missing from policy={sorted(indexed-selected)}, absent from index={sorted(selected-indexed)}")
    output_root.mkdir(parents=True,exist_ok=True)
    manifest={"organization":ORG_NAME,"source_index":str(index_path),"source_curve_root":str(CURVE_ROOT),"target":"log10(|ID| / A)","inputs":["|VGS| (V)","|VDS| (V)"],"points_per_curve":POINTS_PER_CURVE,"sampling":"Deterministic approximately uniform sampling over each complete curve.","split_rationale":SPLIT_RATIONALE,"curve_level_policy":CURVE_SPLIT_POLICY,"splits":{}}
    outputs={}
    for split,selection in CURVE_SPLIT_POLICY.items():
        rows=[]; curves=[]
        for family,filenames in selection.items():
            for filename in filenames:
                record=records_by_file[(family,filename)]; path=curve_path(record)
                if not path.is_file(): raise FileNotFoundError(path)
                factor,unit=current_scale(path); curve_rows=read_curve(path,factor)
                fixed=abs(float(record["fixed_voltage_simulated"])); curve_id=path.stem
                for row in curve_rows:
                    row.update(curve_id=curve_id,curve_family=family,fixed_voltage_abs_V=fixed,sweep_voltage_abs_V=row["VGS_abs_V"] if family=="transfer" else row["VDS_abs_V"])
                rows.extend(curve_rows)
                curves.append({"curve_id":curve_id,"family":family,"fixed_voltage_abs_V":fixed,"csv_file":filename,"n_points":len(curve_rows),"current_typic":unit,"current_scale_to_ampere":factor,"indexed_parameters":record.get("params",{})})
        path=output_root/f"{split}.csv"
        with path.open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=["VGS_abs_V","VDS_abs_V","log10_ID_abs_A","curve_id","curve_family","fixed_voltage_abs_V","sweep_voltage_abs_V"]); writer.writeheader(); writer.writerows(rows)
        outputs[split]=path; manifest["splits"][split]={"csv_file":path.name,"n_curves":len(curves),"n_points":len(rows),"curves":curves}
    manifest_path=output_root/"curve_split_manifest.json"; manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"); outputs["manifest"]=manifest_path
    return outputs
def main()->None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--index",type=Path,default=INDEX_PATH); parser.add_argument("--output-root",type=Path,default=OUTPUT_ROOT); args=parser.parse_args()
    for name,path in build_splits(args.index,args.output_root).items(): print(f"{name}: {path}")
if __name__=="__main__": main()
