#!/usr/bin/env python3
"""
dataset_to_jsonl.py
Convert a dataset folder containing:
  train_texts.txt, train_times.txt, test_texts.txt, test_times.txt
or
  <dataset>_train_texts.txt, <dataset>_train_times.txt, <dataset>_test_texts.txt, <dataset>_test_times.txt
into JSONL files saved to:
  <dataset_dir>/<dataset_name>_jsonl/

Each JSONL line: {"text": "...", "time": <int or str>}

Usage:
  python -m scripts.dataset_to_jsonl --dataset_name NeurIPS --dataset_dir datasets/NeurIPS
"""
import os
import argparse
import json

def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

def write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _cast_time(tm):
    try:
        return int(tm)
    except:
        return tm

def convert_split(split, dataset_name, dataset_dir, out_dir):
    """
    Converts one split (train/test). Handles both prefixed and non-prefixed filenames.
    """
    # Try both filename patterns
    candidates = [
        (f"{dataset_name}_{split}_texts.txt", f"{dataset_name}_{split}_times.txt"),
        (f"{split}_texts.txt", f"{split}_times.txt"),
    ]
    texts_fp = times_fp = None
    for tfile, tfp in candidates:
        t_path, tm_path = os.path.join(dataset_dir, tfile), os.path.join(dataset_dir, tfp)
        if os.path.exists(t_path) and os.path.exists(tm_path):
            texts_fp, times_fp = t_path, tm_path
            break

    if not (texts_fp and times_fp):
        print(f"[!] Skipping {split}: files not found in {dataset_dir}")
        return

    texts = read_lines(texts_fp)
    with open(times_fp, "r", encoding="utf-8") as f:
        times = [line.strip() for line in f if line.strip()]

    if len(texts) != len(times):
        raise ValueError(f"{split} texts ({len(texts)}) and times ({len(times)}) length mismatch")

    records = [{"text": t, "time": _cast_time(tm)} for t, tm in zip(texts, times)]

    out_name = f"{dataset_name}_{split}.jsonl"
    out_path = os.path.join(out_dir, out_name)
    write_jsonl(out_path, records)

    print(f"✔ Wrote {split} to {out_path} ({len(records)} samples)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", required=True, help="Dataset name (e.g. acl, NeurIPS, reuters)")
    ap.add_argument("--dataset_dir", required=True, help="Path to dataset folder")
    args = ap.parse_args()

    # Automatically choose output directory inside dataset folder
    out_dir = os.path.join(args.dataset_dir, f"{args.dataset_name}_jsonl")

    convert_split("train", args.dataset_name, args.dataset_dir, out_dir)
    convert_split("test", args.dataset_name, args.dataset_dir, out_dir)

    print(f"\n✅ All JSONL files saved in: {out_dir}\n")

if __name__ == "__main__":
    main()
