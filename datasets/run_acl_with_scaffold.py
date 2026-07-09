
#!/usr/bin/env python3
"""
run_acl_with_scaffold.py
Run Phase 1–3 of the pipeline on the ACL dataset by:
  1) converting ACL files to JSONL,
  2) invoking the scaffold's demo on the produced folder.

Usage:
  python run_acl_with_scaffold.py --acl_dir /path/to/ACL --scaffold_root /path/to/scaffold --K 100 --encoder all-mpnet-base-v2
"""
import os
import argparse
import subprocess
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acl_dir", required=True)
    ap.add_argument("--scaffold_root", required=True, help="Root of the project scaffold you downloaded/unzipped")
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--encoder", type=str, default="all-mpnet-base-v2")
    ap.add_argument("--out_dir", type=str, default="acl_jsonl")
    args = ap.parse_args()

    # 1) Convert to JSONL
    conv = os.path.join(os.path.dirname(__file__), "acl_to_jsonl.py")
    cmd_conv = [sys.executable, conv, "--acl_dir", args.acl_dir, "--out_dir", args.out_dir]
    subprocess.check_call(cmd_conv)

    # 2) Call scaffold demo_minimal.py pointing to out_dir
    demo = os.path.join(args.scaffold_root, "demo_minimal.py")
    if not os.path.exists(demo):
        raise FileNotFoundError(f"demo_minimal.py not found at {demo}. Set --scaffold_root to the unzipped scaffold directory.")
    cmd_demo = [sys.executable, demo, "--data_dir", args.out_dir, "--K", str(args.K), "--encoder", args.encoder]
    subprocess.check_call(cmd_demo)

if __name__ == "__main__":
    main()
