#!/usr/bin/env python3
import argparse, os, numpy as np, torch
from sentence_transformers import SentenceTransformer
from src.utils import l2_normalize

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refined_dir", type=str, default="artifacts/refined")
    ap.add_argument("--vocab_file", type=str, required=True)
    ap.add_argument("--encoder", type=str, default="all-mpnet-base-v2")
    ap.add_argument("--topn", type=int, default=10)
    args = ap.parse_args()

    # Load vocab
    vocab = [w.strip() for w in open(args.vocab_file) if w.strip()]
    print(f"Loaded {len(vocab)} vocab words")

    # Encode vocab
    model = SentenceTransformer(args.encoder)
    word_emb = model.encode(vocab, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    # Iterate slices
    for fn in sorted(os.listdir(args.refined_dir)):
        if not fn.endswith(".npy"): continue
        Z = np.load(os.path.join(args.refined_dir, fn))
        Z = l2_normalize(Z)
        sims = Z @ word_emb.T  # (K, |V|)
        top_ids = np.argsort(-sims, axis=1)[:, :args.topn]

        print(f"\n=== {fn} ===")
        for k in range(Z.shape[0]):
            top_words = [vocab[i] for i in top_ids[k]]
            print(f"Topic {k:02d}: {' '.join(top_words)}")

if __name__ == "__main__":
    main()
