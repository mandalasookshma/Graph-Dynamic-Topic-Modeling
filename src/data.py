import os, json, glob
from typing import List, Dict, Any
import pandas as pd



def remap_times(df: pd.DataFrame):
    uniq = sorted(df["time"].unique())
    mapping = {t:i for i,t in enumerate(uniq)}
    df = df.copy()
    df["t_idx"] = df["time"].map(mapping)
    return df, mapping

def slice_by_time(df, max_docs_per_slice=None):
    slices = {}
    for t, sub in df.groupby("t_idx"):
        if max_docs_per_slice is not None:
            sub = sub.head(int(max_docs_per_slice))
        slices[int(t)] = sub.reset_index(drop=True)
    return slices


def load_jsonl_folder(data_dir: str, split: str = "train",
                      text_key: str = "text", time_key: str = "time"):
    """
    split: "train" or "test" (expects NeurIPS_train.jsonl / NeurIPS_test.jsonl)
    """
    fname = f"{split}.jsonl"
    files = sorted(glob.glob(os.path.join(data_dir, fname)))
    if not files:
        raise FileNotFoundError(f"Could not find {fname} inside {data_dir}")

    rows = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                rows.append({"text": obj[text_key], "time": obj[time_key]})
    return pd.DataFrame(rows)
