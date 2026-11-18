"""Comparison utilities for model result CSVs."""
from pathlib import Path
import pandas as pd

def compare(metric_paths, output="results/metrics/evaluation_results.csv"):
    frames=[]
    for p in metric_paths:
        path=Path(p)
        if path.exists(): frames.append(pd.read_csv(path))
    if not frames: raise FileNotFoundError("No metric CSVs found.")
    df=pd.concat(frames, ignore_index=True)
    Path(output).parent.mkdir(parents=True, exist_ok=True); df.to_csv(output,index=False)
    return df
