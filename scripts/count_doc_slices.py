#!/usr/bin/env python3
"""
Count and visualize the number of documents per time slice
using only the official dataset files (train_times.txt, test_times.txt).
"""

import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "datasets/NeurIPS"  # adjust if your dataset is elsewhere

def load_times(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]

def summarize_split(split_name, times):
    counts = Counter(times)
    slices = sorted(counts.keys())

    print(f"\n📊 {split_name} split — documents per time slice:\n")
    for t in slices:
        print(f"Slice {t:02d}: {counts[t]} docs")

    vals = [counts[t] for t in slices]
    print(f"\nSummary ({split_name})")
    print(f"  Total docs:       {sum(vals)}")
    print(f"  Number of slices: {len(slices)}")
    print(f"  Min slice size:   {min(vals)}")
    print(f"  Max slice size:   {max(vals)}")
    print(f"  Slice IDs:        {slices}")

    return slices, vals


def plot_counts(train_counts, test_counts, train_vals, test_vals, out_path="docs_per_slice_all.png"):
    plt.figure(figsize=(10, 5))
    width = 0.4
    plt.bar(np.array(train_counts) - width/2, train_vals, width=width,
            color="skyblue", label="Train", edgecolor="k")
    plt.bar(np.array(test_counts) + width/2, test_vals, width=width,
            color="lightgreen", label="Test", edgecolor="k")

    plt.xlabel("Time Slice ID")
    plt.ylabel("Number of Documents")
    plt.title("Documents per Time Slice (Train vs Test)")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"\n✅ Saved bar chart as: {out_path}")


if __name__ == "__main__":
    train_times_path = os.path.join(DATA_DIR, "train_times.txt")
    test_times_path  = os.path.join(DATA_DIR, "test_times.txt")

    train_times = load_times(train_times_path)
    test_times  = load_times(test_times_path)

    train_slices, train_vals = summarize_split("Train", train_times)
    test_slices,  test_vals  = summarize_split("Test",  test_times)

    # combined plot
    plot_counts(train_slices, test_slices, train_vals, test_vals)
