#!/usr/bin/env python3
import argparse, os, collections
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer

# --- Utility ---
def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

# --- Main ---
def main():
    ap = argparse.ArgumentParser(description="Visualize topic-word associations over time (auto trend detection)")
    ap.add_argument("--refined_dir", type=str, default="artifacts/refined",
                    help="Directory containing per-slice .npy topic embeddings")
    ap.add_argument("--vocab_file", type=str, required=True,
                    help="Text file with one word per line")
    ap.add_argument("--encoder", type=str, default="all-mpnet-base-v2",
                    help="SentenceTransformer model name")
    ap.add_argument("--topn", type=int, default=10,
                    help="Number of top words per topic")
    ap.add_argument("--auto_trend_top", type=int, default=5,
                    help="Automatically track N most frequent top words")
    args = ap.parse_args()

    # Load vocab
    vocab = [w.strip() for w in open(args.vocab_file) if w.strip()]
    print(f"Loaded {len(vocab)} vocab words")

    # Encode vocab
    print(f"Encoding vocab with {args.encoder} ...")
    model = SentenceTransformer(args.encoder)
    word_emb = model.encode(vocab, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    os.makedirs("topic_wordclouds", exist_ok=True)
    os.makedirs("topic_trends", exist_ok=True)

    all_sims = []  # list of (slice_name, sims)
    word_counter = collections.Counter()

    # --- Process each slice ---
    for fn in sorted(os.listdir(args.refined_dir)):
        if not fn.endswith(".npy"): continue
        path = os.path.join(args.refined_dir, fn)
        Z = np.load(path)
        Z = l2_normalize(Z)

        sims = Z @ word_emb.T  # cosine similarity (K x |V|)
        all_sims.append((fn, sims))
        top_ids = np.argsort(-sims, axis=1)[:, :args.topn]

        print(f"\n=== {fn} ===")
        for k in range(Z.shape[0]):
            top_words = [vocab[i] for i in top_ids[k]]
            print(f"Topic {k:02d}: {' '.join(top_words)}")

            # Count words for auto-trend detection
            word_counter.update(top_words)

            # --- Word cloud ---
            word_scores = {vocab[i]: float(sims[k, i]) for i in top_ids[k]}
            wc = WordCloud(width=500, height=400, background_color="white", colormap="viridis")
            wc.generate_from_frequencies(word_scores)

            plt.figure()
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"{fn} | Topic {k}")
            plt.tight_layout()
            plt.savefig(f"topic_wordclouds/{fn}_topic{k:02d}.png")
            plt.close()

    # --- Determine top trend words automatically ---
    trend_words = [w for w, _ in word_counter.most_common(args.auto_trend_top)]
    print(f"\n📈 Automatically selected trend words: {trend_words}")

    # --- Trend plots ---
    print("\nGenerating trend plots...")
    slice_names = [fn for fn, _ in all_sims]
    num_topics = all_sims[0][1].shape[0] if all_sims else 0

    for word in trend_words:
        if word not in vocab:
            print(f"⚠️  Word '{word}' not in vocab, skipping.")
            continue
        wid = vocab.index(word)

        for topic_idx in range(num_topics):
            scores = [sims[topic_idx, wid] for _, sims in all_sims]
            plt.plot(slice_names, scores, marker="o", label=f"Topic {topic_idx:02d}")

        plt.title(f"'{word}' similarity over time across topics")
        plt.xlabel("Slice")
        plt.ylabel("Cosine similarity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"topic_trends/{word}_trend.png")
        plt.close()

    print("\n✅ Word clouds saved in 'topic_wordclouds/'")
    print("✅ Trend plots saved in 'topic_trends/'")

if __name__ == "__main__":
    main()
