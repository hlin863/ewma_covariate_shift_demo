"""Run interpretable BCI decision-model comparisons and export evidence."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
import pandas as pd
from src.adaptation.experiments.decision_models_bci import run_bci_decision_model_experiment

def run_and_write(output_root=PROJECT_ROOT/"outputs", *, data_2a=PROJECT_ROOT/"data/raw/bci_competition_iv_2a", labels_2a=PROJECT_ROOT/"data/raw/bci_competition_iv_2a_labels", data_2b=PROJECT_ROOT/"data/raw/bci_competition_iv_2b", datasets=("2a","2b"), subjects=tuple(range(1,10)), random_state=42, tree_depth=4, min_samples_leaf=5, n_neighbors=5, rbf_sigma=1.0, explain_trials=12):
    root=Path(output_root); metrics_dir=root/"metrics"; detail_dir=root/"decision_models"; metrics_dir.mkdir(parents=True,exist_ok=True); detail_dir.mkdir(parents=True,exist_ok=True)
    frame,details=run_bci_decision_model_experiment(data_2a=data_2a,labels_2a=labels_2a,data_2b=data_2b,datasets=datasets,subjects=subjects,random_state=random_state,tree_depth=tree_depth,min_samples_leaf=min_samples_leaf,n_neighbors=n_neighbors,rbf_sigma=rbf_sigma,explain_trials=explain_trials)
    if frame.empty: raise ValueError("No BCI decision-model results were produced.")
    metrics_path=metrics_dir/"bci_decision_models.csv"; frame.to_csv(metrics_path,index=False)
    for key,detail in details.items(): (detail_dir/f"{key}.json").write_text(json.dumps(detail,indent=2,allow_nan=False),encoding="utf-8")
    return {"metrics":metrics_path,"details":detail_dir}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--data-2a",type=Path,default=Path("data/raw/bci_competition_iv_2a")); p.add_argument("--labels-2a",type=Path,default=Path("data/raw/bci_competition_iv_2a_labels")); p.add_argument("--data-2b",type=Path,default=Path("data/raw/bci_competition_iv_2b")); p.add_argument("--datasets",nargs="+",choices=("2a","2b"),default=("2a","2b")); p.add_argument("--subjects",nargs="+",type=int,default=list(range(1,10))); p.add_argument("--seed",type=int,default=42); p.add_argument("--tree-depth",type=int,default=4); p.add_argument("--min-samples-leaf",type=int,default=5); p.add_argument("--neighbors",type=int,default=5); p.add_argument("--rbf-sigma",type=float,default=1.0); p.add_argument("--explain-trials",type=int,default=12); p.add_argument("--output-root",type=Path,default=Path("outputs")); a=p.parse_args()
    paths=run_and_write(a.output_root,data_2a=a.data_2a,labels_2a=a.labels_2a,data_2b=a.data_2b,datasets=tuple(a.datasets),subjects=tuple(a.subjects),random_state=a.seed,tree_depth=a.tree_depth,min_samples_leaf=a.min_samples_leaf,n_neighbors=a.neighbors,rbf_sigma=a.rbf_sigma,explain_trials=a.explain_trials)
    frame=pd.read_csv(paths["metrics"]); print(frame.to_string(index=False)); print("\nMean accuracy by dataset/method:"); print(frame.groupby(["dataset","method"])["accuracy"].agg(["mean","std"]).to_string()); print(f"\nMetrics: {paths['metrics']}"); print(f"Evidence: {paths['details']}"); print("Explorer: http://127.0.0.1:5000/results/decision-models")
if __name__=="__main__": main()
