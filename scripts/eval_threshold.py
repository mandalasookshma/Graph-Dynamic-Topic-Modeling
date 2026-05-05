#!/usr/bin/env python3
# scripts/eval_recompute_coherence.py
#
# Checkpoint-only evaluation (BoW-aligned, dynamic topics):
#   - Loads Z_by_t + tw_decoder from best_model.pt
#   - Loads CRF-exported phi_logits from ckpt["crf_export"]["phi_logits"] (required)
#   - Uses threshold-based filtering to select active topics per slice
#   - Extracts top-words via decoder β (TopicWordAttention)
#   - Computes dynamic coherence and dynamic topic diversity
#   - Reports K_active summaries
#
# Tokenization matches training BoW construction:
#   - lowercase
#   - whitespace split
#   - vocab filtering (keep only words in vocab.txt)

import os
import json
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn.functional as F

from src.bipartite_decoder import TopicWordAttention


# ----------------------------
# Tokenization (MATCH TRAINING)
# ----------------------------
def tokenize_with_vocab(texts: List[str], vocab_set: set) -> List[List[str]]:
    out: List[List[str]] = []
    for s in texts:
        toks = s.strip().lower().split()
        toks = [w for w in toks if w in vocab_set]
        out.append(toks)
    return out


def compute_topic_drift(Z_by_t):
    drift = []
    times = sorted(Z_by_t.keys())
    for i in range(1, len(times)):
        Z_prev = Z_by_t[times[i - 1]]
        Z_curr = Z_by_t[times[i]]
        if torch.is_tensor(Z_prev):
            Z_prev = Z_prev.detach().cpu().numpy()
        if torch.is_tensor(Z_curr):
            Z_curr = Z_curr.detach().cpu().numpy()
        drift.append(np.linalg.norm(Z_curr - Z_prev))
    return float(np.mean(drift)) if drift else 0.0


# ----------------------------
# Coherence helpers
# ----------------------------
def _coherence_from_tokenized(
    tokenized_ref: List[List[str]],
    top_words: List[List[str]],
    coherence_type: str = "c_v",
    topn: int = 20,
) -> float:
    dictionary = Dictionary(tokenized_ref)

    topics_filtered: List[List[str]] = []
    for topic in top_words:
        topic_f = [w for w in topic if w in dictionary.token2id]
        if topic_f:
            topics_filtered.append(topic_f)

    if not topics_filtered:
        return float("nan")

    cm = CoherenceModel(
        texts=tokenized_ref,
        dictionary=dictionary,
        topics=topics_filtered,
        topn=topn,
        coherence=coherence_type,
    )
    vals = cm.get_coherence_per_topic()
    if np.isnan(vals).any():
        vals = np.nan_to_num(vals, nan=0.0)
    return float(np.mean(vals))


def dynamic_coherence(
    train_texts: List[str],
    train_times: np.ndarray,
    top_words_list_by_t: Dict[int, List[List[str]]],
    vocab: List[str],
    coherence_type: str = "c_v",
    topn: int = 20,
    K_total_by_t: Optional[Dict[int, int]] = None,
) -> Tuple[float, Dict[int, Dict]]:
    """
    Computes coherence per time slice using slice-specific reference corpus.
    Works for dynamic K because topics list length can vary by t.
    """
    vocab_set = set(vocab)
    scores: List[float] = []
    by_t: Dict[int, Dict] = {}

    for t in sorted(top_words_list_by_t.keys()):
        idx = np.where(train_times == t)[0]
        ref_texts = [train_texts[i] for i in idx]
        tokenized_ref = tokenize_with_vocab(ref_texts, vocab_set)

        cv = _coherence_from_tokenized(
            tokenized_ref,
            top_words_list_by_t[t],
            coherence_type=coherence_type,
            topn=topn,
        )

        K_active = int(len(top_words_list_by_t[t]))
        K_total = int(K_total_by_t[t]) if (K_total_by_t is not None and t in K_total_by_t) else K_active
        K_inactive = int(K_total - K_active)

        by_t[int(t)] = {
            "mean": float(cv),
            "n_docs": int(len(ref_texts)),
            "K_active": K_active,
            "K_inactive": K_inactive,
            "K_total": K_total,
        }
        scores.append(cv)

    return float(np.mean(scores)) if scores else 0.0, by_t


# ----------------------------
# Dynamic topic diversity
# ----------------------------
def dynamic_topic_diversity(
    top_words_by_t: Dict[int, List[List[str]]],
    train_texts: List[str],
    train_times: np.ndarray,
    vocab: List[str],
    topn: int = 10,
    disable_tqdm: bool = True,
) -> Tuple[float, Dict[int, float]]:
    """
    TD_t = (# words that occur once across topics and exist in slice vocab) / (K_t * topn)
    Dynamic K supported because denominator uses K_t=len(topics after filtering).
    """
    td_by_t: Dict[int, float] = {}
    time_idx = np.sort(np.unique(train_times))
    vocab_set = set(vocab)

    slice_vocab: Dict[int, set] = {}
    for t in time_idx:
        doc_idx = np.where(train_times == t)[0]
        docs_tok = tokenize_with_vocab([train_texts[i] for i in doc_idx], vocab_set)
        slice_vocab[int(t)] = set([w for d in docs_tok for w in d])

    for t in tqdm(time_idx, desc="Computing dynamic TD", disable=disable_tqdm):
        t = int(t)
        if t not in top_words_by_t:
            continue

        topics = top_words_by_t[t]
        if not topics:
            td_by_t[t] = 0.0
            continue

        flatten = [w for topic in topics for w in topic[:topn]]
        counter = Counter(flatten)

        num_assoc = sum(1 for w in flatten if counter[w] == 1 and w in slice_vocab[t])
        total = len(topics) * topn
        td_by_t[t] = float(num_assoc / total) if total > 0 else 0.0

    mean_td = float(np.mean(list(td_by_t.values()))) if td_by_t else 0.0
    return mean_td, td_by_t


# ----------------------------
# Word embeddings
# ----------------------------
def l2_normalize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(X, axis=-1, keepdims=True) + eps
    return X / n


def load_word_embeddings_from_npz(word_emb_npz: str, vocab_size: int) -> np.ndarray:
    data = np.load(word_emb_npz)
    E = data.get("embeddings", data.get("E", data.get("arr_0")))
    if E is None:
        raise KeyError("word_embeddings.npz must contain 'embeddings' or 'E' or default 'arr_0'.")
    if E.shape[0] != vocab_size:
        raise ValueError(f"Embedding rows ({E.shape[0]}) != vocab size ({vocab_size})")
    return l2_normalize(E.astype("float32"))


# ----------------------------
# Threshold-based top words via decoder β
# ----------------------------
def top_words_from_decoder_threshold(
    Z_by_t,
    phi_by_t,
    vocab,
    E_words,
    decoder_state,
    topn=10,
    threshold=0.01,
    use_z_norm=False,
    device="cpu",
):
    """
    Threshold-based active topic selection.

    If use_z_norm=False:
        score = pi_t
    If use_z_norm=True:
        score = pi_t * ||Z_t||

    Keep topics with score >= threshold.
    If no topic survives, keep the top-1 topic.
    """
    V = len(vocab)
    d = int(E_words.shape[1])

    dec = TopicWordAttention(d=d).to(device)
    dec.load_state_dict(decoder_state)
    dec.eval()

    Ew = torch.tensor(E_words, dtype=torch.float32, device=device)
    Ew = F.normalize(Ew, dim=-1)

    top_words_by_t = {}
    K_active_by_t = {}
    score_by_t = {}

    for t in sorted(Z_by_t.keys()):
        Zt = Z_by_t[t]
        if not torch.is_tensor(Zt):
            Zt = torch.tensor(Zt, dtype=torch.float32)
        Zt = Zt.clone().detach().to(device)

        pi_t = phi_by_t[t]
        if not torch.is_tensor(pi_t):
            pi_t = torch.tensor(pi_t, dtype=torch.float32)
        pi_t = pi_t.clone().detach().to(device)
        pi_t = pi_t / (pi_t.sum() + 1e-12)

        if use_z_norm:
            Z_norm = torch.norm(Zt, dim=-1)
            score = pi_t * Z_norm
        else:
            score = pi_t

        keep_idx = torch.nonzero(score >= threshold, as_tuple=False).squeeze(-1)

        if keep_idx.numel() == 0:
            keep_idx = torch.argmax(score).view(1)

        K_active_by_t[int(t)] = int(keep_idx.numel())
        score_by_t[int(t)] = score.detach().cpu().numpy().tolist()

        Zt_active = Zt.index_select(0, keep_idx)
        beta = dec(Zt_active, Ew)

        if beta.size(1) != V:
            raise ValueError(f"beta vocab dim mismatch: {beta.size(1)} vs {V}")

        top_idx = torch.topk(beta, k=topn, dim=-1).indices.cpu().numpy()
        top_words_by_t[int(t)] = [[vocab[j] for j in row] for row in top_idx]

    return top_words_by_t, K_active_by_t, score_by_t


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Checkpoint-only dynamic eval with threshold-based topic selection. No per-slice printing."
    )
    ap.add_argument("--model_ckpt", type=str, required=True,
                    help="Path to best_model.pt (must include Z_by_t, tw_decoder, and crf_export.phi_logits).")
    ap.add_argument("--vocab_file", type=str, required=True)
    ap.add_argument("--word_emb_npz", type=str, required=True,
                    help="Precomputed word embeddings matching vocab.")
    ap.add_argument("--train_texts", type=str, required=True)
    ap.add_argument("--train_times", type=str, required=True)
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--coherence", type=str, default="c_v",
                    choices=["c_v", "c_npmi", "u_mass", "c_uci"])
    ap.add_argument("--save_dir", type=str, default="eval_out")
    ap.add_argument("--topic_threshold", type=float, default=0.01,
                    help="Threshold for selecting active topics.")
    ap.add_argument("--use_z_norm", action="store_true",
                    help="If set, use score = pi * ||Z||; otherwise use score = pi only.")

    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # 0) Load vocab
    with open(args.vocab_file, "r", encoding="utf-8") as f:
        vocab = [w.strip() for w in f if w.strip()]

    # 1) Load checkpoint contents
    ckpt = torch.load(args.model_ckpt, map_location="cpu", weights_only=False)

    if "Z_by_t" not in ckpt:
        raise RuntimeError("Checkpoint missing 'Z_by_t'.")
    if "tw_decoder" not in ckpt:
        raise RuntimeError("Checkpoint missing 'tw_decoder'.")
    if "crf_export" not in ckpt:
        raise RuntimeError("Checkpoint missing 'crf_export'.")

    Z_by_t = ckpt["Z_by_t"]
    tw_state = ckpt["tw_decoder"]
    crf_export = ckpt["crf_export"]

    if "phi_logits" not in crf_export:
        raise RuntimeError("CRF phi_logits not found in checkpoint.")

    phi_logits = torch.tensor(crf_export["phi_logits"], dtype=torch.float32)
    phi_by_t = torch.softmax(phi_logits, dim=-1)

    # 2) Load word embeddings
    E_words = load_word_embeddings_from_npz(args.word_emb_npz, vocab_size=len(vocab))

    # 3) Extract variable-length topics per slice using threshold filtering
    device = "cuda" if torch.cuda.is_available() else "cpu"
    score_type = "pi*||Z||" if args.use_z_norm else "pi"
    print(f"\n[info] Using threshold-based topic selection")
    print(f"[info] score type = {score_type}")
    print(f"[info] threshold = {args.topic_threshold}")

    top_words_by_t, K_active_by_t, score_by_t = top_words_from_decoder_threshold(
        Z_by_t=Z_by_t,
        phi_by_t=phi_by_t,
        vocab=vocab,
        E_words=E_words,
        decoder_state=tw_state,
        topn=args.topn,
        threshold=args.topic_threshold,
        use_z_norm=args.use_z_norm,
        device=device,
    )

    # 4) Load train texts/times
    train_texts = [line.rstrip("\n") for line in open(args.train_texts, encoding="utf-8")]
    train_times = np.loadtxt(args.train_times, dtype=int)

    # Total capacity per slice
    K_total_by_t = {}
    for t, Z in Z_by_t.items():
        if torch.is_tensor(Z):
            K_total_by_t[int(t)] = int(Z.shape[0])
        else:
            K_total_by_t[int(t)] = int(np.asarray(Z).shape[0])

    # 5) Compute metrics
    mean_tc, tc_by_t = dynamic_coherence(
        train_texts=train_texts,
        train_times=train_times,
        top_words_list_by_t=top_words_by_t,
        vocab=vocab,
        coherence_type=args.coherence,
        topn=args.topn,
        K_total_by_t=K_total_by_t,
    )

    mean_td, td_by_t = dynamic_topic_diversity(
        top_words_by_t=top_words_by_t,
        train_texts=train_texts,
        train_times=train_times,
        vocab=vocab,
        topn=args.topn,
        disable_tqdm=True,
    )

    mean_drift = compute_topic_drift(Z_by_t)

    kactive = np.array([K_active_by_t[t] for t in sorted(K_active_by_t.keys())], dtype=float)

    print(f"Average topic drift = {mean_drift:.4f}")

    print("\n[Dynamic Active Topic Cardinality]")
    print(f"[info] K_active (threshold-based): "
          f"mean={kactive.mean():.2f} "
          f"min={kactive.min():.2f} "
          f"max={kactive.max():.2f}")
    print(f"[info] K_max (capacity) = {len(next(iter(Z_by_t.values())))}")

    print(f"Dynamic topic coherence ({args.coherence}) = {mean_tc:.4f}")
    print(f"Dynamic topic diversity (refined) = {mean_td:.4f}")

    # 6) Save outputs
    with open(os.path.join(args.save_dir, "top_words_by_t.json"), "w", encoding="utf-8") as f:
        json.dump(top_words_by_t, f, ensure_ascii=False, indent=2)

    out_data = {
        "coherence": {
            "mean": float(mean_tc),
            "by_t": tc_by_t,
            "metric": args.coherence,
        },
        "diversity": {
            "mean": float(mean_td),
            "by_t": td_by_t,
        },
        "topic_cardinality": {
            "K_active_by_t_threshold": K_active_by_t,
            "K_active_summary": {
                "mean": float(kactive.mean()),
                "min": float(kactive.min()),
                "max": float(kactive.max()),
            },
            "threshold": float(args.topic_threshold),
            "score_type": "pi_times_z_norm" if args.use_z_norm else "pi_only",
            "K_max_capacity": int(len(next(iter(Z_by_t.values())))),
        },
        "settings": {
            "input_representation": "bow",
            "beta_source": "decoder",
            "model_ckpt": args.model_ckpt,
            "vocab_file": args.vocab_file,
            "word_emb_npz": args.word_emb_npz,
            "topn_words": int(args.topn),
            "topic_threshold": float(args.topic_threshold),
            "use_z_norm": bool(args.use_z_norm),
        },
        "topic_drift": mean_drift,
        "topic_scores_by_t": score_by_t,
    }

    with open(os.path.join(args.save_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
