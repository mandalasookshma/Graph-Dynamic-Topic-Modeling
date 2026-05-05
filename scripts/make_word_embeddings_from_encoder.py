# make_word_embeddings_from_encoder.py
import argparse, numpy as np
from sentence_transformers import SentenceTransformer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_file", required=True)
    ap.add_argument("--encoder", default="all-mpnet-base-v2")
    ap.add_argument("--out_npz", required=True)
    args = ap.parse_args()

    vocab = [w.strip() for w in open(args.vocab_file, encoding="utf-8") if w.strip()]
    model = SentenceTransformer(args.encoder)
    E = model.encode(vocab, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)

    np.savez_compressed(args.out_npz, embeddings=E.astype("float32"), vocab=np.array(vocab, dtype=object))
    print(f"Saved word embeddings: {args.out_npz} | shape={E.shape}, vocab={len(vocab)}")

if __name__ == "__main__":
    main()
