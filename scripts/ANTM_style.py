
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
import matplotlib.pyplot as plt
import networkx as nx

def compute_birth_death_dynamic_topic_count(
    Z_by_t,
    phi_by_t,
    save_dir,
    sim_threshold=0.60,
    activity_ratio=0.01
):
    """
    Dynamic topic count using birth-death alignment.

    Active score:
        score_k = pi_t[k] * ||Z_t[k]||

    Active topic:
        score_k > activity_ratio * max(score)

    Survival:
        cosine(z_tk, z_t1j) > sim_threshold

    Recurrence:
        K[t+1] = K[t] + births - deaths
    """

    os.makedirs(save_dir, exist_ok=True)
    slices = sorted(Z_by_t.keys())

    # --------------------------------------------------------
    # STEP 1: ACTIVE TOPICS PER SLICE
    # --------------------------------------------------------
    active_topics = {}
    score_cache = {}

    for t in slices:

        Zt = Z_by_t[t]
        pi_t = phi_by_t[t]

        if torch.is_tensor(Zt):
            Zt = Zt.detach().cpu().numpy()

        if torch.is_tensor(pi_t):
            pi_t = pi_t.detach().cpu().numpy()

        znorm = np.linalg.norm(Zt, axis=1)

        score = pi_t * znorm
        score_cache[t] = score

        thr = activity_ratio * np.max(score)

        idx = np.where(score > thr)[0].tolist()

        active_topics[t] = idx

    # --------------------------------------------------------
    # STEP 2: INITIAL COUNT
    # --------------------------------------------------------
    K_dyn = {}
    births_by_t = {}
    deaths_by_t = {}

    t0 = slices[0]
    K_dyn[t0] = len(active_topics[t0])
    births_by_t[t0] = K_dyn[t0]
    deaths_by_t[t0] = 0

    # --------------------------------------------------------
    # STEP 3: BIRTH-DEATH ALIGNMENT
    # --------------------------------------------------------
    for i in range(len(slices) - 1):

        t = slices[i]
        t_next = slices[i + 1]

        curr_topics = active_topics[t]
        next_topics = active_topics[t_next]

        matched_next = set()
        deaths = 0

        for k1 in curr_topics:

            z1 = Z_by_t[t][k1]
            z1 = z1 / (np.linalg.norm(z1) + 1e-12)

            best_sim = -1
            best_k2 = None

            for k2 in next_topics:

                z2 = Z_by_t[t_next][k2]
                z2 = z2 / (np.linalg.norm(z2) + 1e-12)

                sim = float(np.dot(z1, z2))

                if sim > best_sim:
                    best_sim = sim
                    best_k2 = k2

            if best_sim >= sim_threshold:
                matched_next.add(best_k2)
            else:
                deaths += 1

        births = 0
        for k2 in next_topics:
            if k2 not in matched_next:
                births += 1

        births_by_t[t_next] = births
        deaths_by_t[t_next] = deaths

        K_dyn[t_next] = K_dyn[t] + births - deaths

    # --------------------------------------------------------
    # STEP 4: SAVE JSON
    # --------------------------------------------------------
    out = {}

    for t in slices:
        out[int(t)] = {
            "dynamic_topic_count": int(K_dyn[t]),
            "births": int(births_by_t[t]),
            "deaths": int(deaths_by_t[t]),
            "active_detected": int(len(active_topics[t]))
        }

    with open(os.path.join(save_dir, "birth_death_topic_counts.json"), "w") as f:
        json.dump(out, f, indent=2)

    # --------------------------------------------------------
    # STEP 5: PLOT
    # --------------------------------------------------------
    xs = slices
    ys = [K_dyn[t] for t in xs]

    plt.figure(figsize=(10,5))
    plt.plot(xs, ys, marker='o', linewidth=2)
    plt.title("Dynamic Topic Count via Birth-Death Alignment")
    plt.xlabel("Slice")
    plt.ylabel("Topic Count")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "birth_death_dynamic_topic_count.png"),
        dpi=300
    )
    plt.close()

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------
    print("\n[Birth-Death Dynamic Topic Count]")
    for t in slices:
        print(
            f"Slice {t}: "
            f"K={K_dyn[t]} | "
            f"B={births_by_t[t]} | "
            f"D={deaths_by_t[t]}"
        )

    return K_dyn



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
        Z_prev = Z_by_t[times[i-1]]
        Z_curr = Z_by_t[times[i]]
        drift.append(np.linalg.norm(Z_curr - Z_prev))
    return float(np.mean(drift)) if drift else 0.0

# ----------------------------
# CRF-based effective K (threshold-free)
# ----------------------------
def effective_K_from_probs(p: np.ndarray, eps: float = 1e-12) -> float:
    """
    p: (K,) simplex
    K_eff = exp(H(p))  (threshold-free)
    """
    p = np.clip(p, eps, 1.0)
    s = float(p.sum())
    if s <= 0:
        return 0.0
    p = p / s
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))



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

    # slice vocab sets
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
# Load checkpoint
# ----------------------------
def load_topics_and_decoder_from_ckpt(model_ckpt: str):
    ckpt = torch.load(model_ckpt, map_location="cpu", weights_only=False)

    if "Z_by_t" not in ckpt:
        raise KeyError("Checkpoint missing 'Z_by_t'.")
    if "tw_decoder" not in ckpt:
        raise KeyError("Checkpoint missing 'tw_decoder'.")

    Z_by_t: Dict[int, np.ndarray] = {}
    for t, z in ckpt["Z_by_t"].items():
        if torch.is_tensor(z):
            Z_by_t[int(t)] = z.detach().cpu().numpy().astype("float32")
        else:
            Z_by_t[int(t)] = np.asarray(z, dtype="float32")

    tw_state = ckpt["tw_decoder"]

    pi_by_t = None
    if "crf_export" in ckpt:
        crf_state = ckpt["crf_export"]

        if "phi" in crf_state:
            phi = crf_state["phi"]
            if torch.is_tensor(phi):
                phi = phi.detach().cpu().numpy()

            pi_by_t = phi / (phi.sum(axis=1, keepdims=True) + 1e-12)

    return Z_by_t, tw_state, pi_by_t


# ----------------------------
# Top words via decoder β 
# ----------------------------

def top_words_from_decoder_entropy(
    Z_by_t,
    phi_by_t,
    vocab,
    E_words,
    decoder_state,
    topn=10,
    device="cpu",
):

    V = len(vocab)
    d = int(E_words.shape[1])

    dec = TopicWordAttention(d=d).to(device)
    dec.load_state_dict(decoder_state)
    dec.eval()

    Ew = torch.tensor(E_words, dtype=torch.float32, device=device)
    Ew = F.normalize(Ew, dim=-1)

    top_words_by_t = {}
    K_eff_entropy = {}

    for t in sorted(Z_by_t.keys()):
        Zt = Z_by_t[t].clone().detach().to(device)
        phi_t = phi_by_t[t].clone().detach().to(device)
        phi_t = F.softplus(phi_t) + 1e-8
        pi_t = phi_t / phi_t.sum()

        Z_norm = torch.norm(Zt, dim=-1)

        score = pi_t * Z_norm
        score = score / (score.sum() + 1e-12)

        H = -(score * torch.log(score + 1e-12)).sum()
        K_eff = torch.exp(H)

        K_eff_entropy[int(t)] = float(K_eff.item())

        K_active = max(1, int(torch.round(K_eff).item()))

        order = torch.argsort(score, descending=True)
        keep_idx = order[:K_active]

        Zt_active = Zt.index_select(0, keep_idx)

        beta = dec(Zt_active, Ew)

        if beta.size(1) != V:
            raise ValueError(
                f"beta vocab dim mismatch: {beta.size(1)} vs {V}"
            )

        top_idx = torch.topk(beta, k=topn, dim=-1).indices.cpu().numpy()
        top_words_by_t[int(t)] = [[vocab[j] for j in row] for row in top_idx]

    return top_words_by_t, K_eff_entropy

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Checkpoint-only dynamic eval with π-mass topic selection + K_eff(entropy). No per-slice printing."
    )
    ap.add_argument("--model_ckpt", type=str, required=True,
                    help="Path to best_model.pt (must include Z_by_t, tw_decoder, and crf_export.pi_by_t).")
    ap.add_argument("--vocab_file", type=str, required=True)
    ap.add_argument("--word_emb_npz", type=str, required=True,
                    help="Precomputed word embeddings matching vocab.")
    ap.add_argument("--train_texts", type=str, required=True)
    ap.add_argument("--train_times", type=str, required=True)
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--coherence", type=str, default="c_v",
                    choices=["c_v", "c_npmi", "u_mass", "c_uci"])
    ap.add_argument("--save_dir", type=str, default="eval_out")


    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # 0) Load vocab
    with open(args.vocab_file, "r", encoding="utf-8") as f:
        vocab = [w.strip() for w in f if w.strip()]

    # 1) Load checkpoint contents
    ckpt = torch.load(args.model_ckpt, map_location="cpu")
    Z_by_t = ckpt["Z_by_t"]
    tw_state = ckpt["tw_decoder"]
    crf_export = ckpt["crf_export"]

    if "phi_logits" not in crf_export:
        raise RuntimeError("CRF phi_logits not found in checkpoint.")

    phi_logits = torch.tensor(crf_export["phi_logits"], dtype=torch.float32)
    phi_by_t = torch.softmax(phi_logits, dim=-1)
    E_words = load_word_embeddings_from_npz(args.word_emb_npz, vocab_size=len(vocab))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n[info] Using geometry-aware π×||Z|| topic selection")

    top_words_by_t, K_eff_by_t = top_words_from_decoder_entropy(
    Z_by_t=Z_by_t,
    phi_by_t=phi_by_t,
    vocab=vocab,
    E_words=E_words,
    decoder_state=tw_state,
    topn=args.topn,
    device=device,
)

    # 4) Load train texts/times
    train_texts = [line.rstrip("\n") for line in open(args.train_texts, encoding="utf-8")]
    train_times = np.loadtxt(args.train_times, dtype=int)
    K_total_by_t = {int(t): int(Z.shape[0]) for t, Z in Z_by_t.items()}

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
    print(f"Average topic drift = {mean_drift:.4f}")
    keff = np.array([K_eff_by_t[t] for t in sorted(K_eff_by_t.keys())], dtype=float)


    print(f"Dynamic topic coherence ({args.coherence}) = {mean_tc:.4f}")
    print(f"Dynamic topic diversity (refined) = {mean_td:.4f}")
    K_dyn = compute_birth_death_dynamic_topic_count(
    Z_by_t={
        t: Z_by_t[t].detach().cpu().numpy()
        if torch.is_tensor(Z_by_t[t]) else Z_by_t[t]
        for t in Z_by_t
    },
    phi_by_t={
        t: phi_by_t[t].detach().cpu().numpy()
        if torch.is_tensor(phi_by_t[t]) else phi_by_t[t]
        for t in range(phi_by_t.shape[0])
    },
    save_dir=args.save_dir,
    sim_threshold=0.55,   
    activity_ratio=0.01    
    )

    print("[saved] birth_death_dynamic_topic_count.png")

    # 7) Save outputs
    with open(os.path.join(args.save_dir, "top_words_by_t.json"), "w", encoding="utf-8") as f:
        json.dump(top_words_by_t, f, ensure_ascii=False, indent=2)

    out_data = {
        "coherence": {"mean": float(mean_tc), "by_t": tc_by_t, "metric": args.coherence},
        "diversity": {"mean": float(mean_td), "by_t": td_by_t},
        "settings": {
            "input_representation": "bow",
            "beta_source": "decoder",
            "model_ckpt": args.model_ckpt,
            "vocab_file": args.vocab_file,
            "word_emb_npz": args.word_emb_npz,
            "topn_words": int(args.topn),
        },
        "topic_drift": mean_drift,
    }

    with open(os.path.join(args.save_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
        


if __name__ == "__main__":
    main()
