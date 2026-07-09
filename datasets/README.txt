
ACL → JSONL conversion tools

Files:
- acl_to_jsonl.py           : Convert ACL folder (train_texts/times, test_texts/times) to JSONL.
- run_acl_with_scaffold.py  : Converts then runs the previously shared scaffold on the resulting JSONL.

Quick use:
1) Convert only:
   python acl_to_jsonl.py --acl_dir /path/to/ACL --out_dir ./acl_jsonl

2) End-to-end with scaffold:
   python run_acl_with_scaffold.py --acl_dir /path/to/ACL --scaffold_root /path/to/unzipped/scaffold --K 100 --encoder all-mpnet-base-v2
