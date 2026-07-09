
#!/usr/bin/env python3
"""
acl_to_jsonl.py
Convert the ACL dataset folder:
  train_texts.txt, train_times.txt, test_texts.txt, test_times.txt
into JSONL files that the scaffold expects:
  <out_dir>/train.jsonl
  <out_dir>/test.jsonl

Each JSONL line: {"text": "...", "time": <int or str>}

Usage:
  python acl_to_jsonl.py --acl_dir /path/to/ACL --out_dir ./acl_jsonl
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acl_dir", required=True, help="Path to ACL dataset folder")
    ap.add_argument("--out_dir", required=True, help="Output folder for JSONL files")
    args = ap.parse_args()

    # Train
    train_texts_fp = os.path.join(args.acl_dir, "train_texts.txt")
    train_times_fp = os.path.join(args.acl_dir, "train_times.txt")
    if not (os.path.exists(train_texts_fp) and os.path.exists(train_times_fp)):
        raise FileNotFoundError("train_texts.txt or train_times.txt not found in ACL dir")

    train_texts = read_lines(train_texts_fp)
    with open(train_times_fp, "r", encoding="utf-8") as f:
        train_times = [line.strip() for line in f if line.strip()]
    if len(train_texts) != len(train_times):
        raise ValueError(f"Train texts ({len(train_texts)}) and times ({len(train_times)}) length mismatch")

    def _cast_time(tm):
        try:
            return int(tm)
        except:
            return tm

    train = [{"text": t, "time": _cast_time(tm)} for t, tm in zip(train_texts, train_times)]
    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train)

    # Test (optional)
    test_texts_fp = os.path.join(args.acl_dir, "test_texts.txt")
    test_times_fp = os.path.join(args.acl_dir, "test_times.txt")
    if os.path.exists(test_texts_fp) and os.path.exists(test_times_fp):
        test_texts = read_lines(test_texts_fp)
        with open(test_times_fp, "r", encoding="utf-8") as f:
            test_times = [line.strip() for line in f if line.strip()]
        if len(test_texts) != len(test_times):
            raise ValueError(f"Test texts ({len(test_texts)}) and times ({len(test_times)}) length mismatch")
        test = [{"text": t, "time": _cast_time(tm)} for t, tm in zip(test_texts, test_times)]
        write_jsonl(os.path.join(args.out_dir, "test.jsonl"), test)

    print(f"Wrote JSONL to: {args.out_dir}")
    print("Files:", [p for p in ["train.jsonl", "test.jsonl"] if os.path.exists(os.path.join(args.out_dir, p))])

if __name__ == "__main__":
    main()
