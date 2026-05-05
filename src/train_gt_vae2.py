
import os, json, argparse
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
from src.graph_transformer import TopicGraphTransformer
from src.bipartite_decoder import TopicWordAttention
from src.data import load_jsonl_folder, remap_times, slice_by_time
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Sequence, Union
from src.theta_encoder import ThetaEncoderLN
from src.crf_weights import HDPTopicWeights
from torch.special import digamma
from src.kl_warmpu import linear_warmup  
import random
import subprocess


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("true", "1", "yes")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def gate_from_pi(pi, eps=1e-8):
    g = torch.sqrt(pi.clamp_min(eps))
    return g / (g.max() + eps)  # normalize max to 1

def bow_from_wids_batch(wids_batch, V: int, device):
    """
    wids_batch: List[List[int]] length B
    returns: X_bow (B, V) float32 counts
    """
    B = len(wids_batch)
    X = torch.zeros((B, V), device=device, dtype=torch.float32)
    for i, wids in enumerate(wids_batch):
        if not wids:
            continue
        idx = torch.tensor(wids, device=device, dtype=torch.long)
        ones = torch.ones((idx.numel(),), device=device, dtype=torch.float32)
        X[i].scatter_add_(0, idx, ones)
    return X

def split_train_val_indices_by_slice(slices, val_ratio=0.1, seed=42):
    rng = np.random.RandomState(seed)
    keys = sorted([int(k) for k in slices.keys()])
    train_idx_by_t, val_idx_by_t = {}, {}
    for t in keys:
        n = len(slices[t])
        perm = rng.permutation(n)
        n_val = int(round(val_ratio * n))
        val_idx_by_t[t] = perm[:n_val]
        train_idx_by_t[t] = perm[n_val:]
    return train_idx_by_t, val_idx_by_t


def build_competitive_edges(
    Z_t: torch.Tensor,
    Z_next: torch.Tensor,
    top_influence: int = 2,
    sim_threshold: float = 0.5,
    anti_sim_threshold: float = 0.2,
    g_t: Optional[torch.Tensor] = None,
    g_next: Optional[torch.Tensor] = None,
):
    """
    Build sparse, directional, anti-smoothing edges.
    """
    K = Z_t.size(0)
    device = Z_t.device

    Zt = F.normalize(Z_t, dim=-1)
    Zp = F.normalize(Z_next, dim=-1)

    edges = []

    # ----------------------------------
    # A) TEMPORAL INFLUENCE (t -> t+1)
    # ----------------------------------
    sim_tp = torch.matmul(Zp, Zt.T)   # (K,K)
    if g_t is not None:
        sim_tp = sim_tp * g_t.view(1, -1)   # downweight inactive sources
    if g_next is not None:
        sim_tp = sim_tp * g_next.view(-1, 1) # downweight inactive dsts

    for i in range(K):
        vals, idx = sim_tp[i].topk(top_influence)
        for j, v in zip(idx, vals):
            edges.append((
                j.item(),          # src (past)
                i + K,             # dst (future)
                float(v.item())
            ))

    # ----------------------------------
    # B) INTRA-SLICE COMPETITION
    # ----------------------------------
    sim_tt = torch.matmul(Zt, Zt.T)
    sim_tt.fill_diagonal_(0)
    if g_t is not None:
        sim_tt = sim_tt * (g_t.view(-1,1) * g_t.view(1,-1))
    for i in range(K):
        j = torch.argmax(sim_tt[i])
        if sim_tt[i, j] > sim_threshold:
            edges.append((
                j.item(),
                i,
                float(sim_tt[i, j].item())
            ))

    # ----------------------------------
    # C) ANTI-SIMILARITY EDGES
    # ----------------------------------
    for i in range(K):
        j = torch.argmin(sim_tt[i])
        if sim_tt[i, j] < anti_sim_threshold:
            edges.append((
                j.item(),
                i,
                float(sim_tt[i, j].item())
            ))

    return edges

class SliceTopicEncoder(nn.Module):
    """
    q(Z_t | X_t): X_t is BoW (B, V), but Z_t is (K, d_z) aligned with E_words.
    """
    def __init__(self, d_in: int, d_z: int, K: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.K = K
        self.d_in = d_in
        self.d_z = d_z

        self.slice_mlp = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

        self.topic_emb = nn.Embedding(K, hidden)

        self.to_mu = nn.Linear(hidden, d_z)
        self.to_logvar = nn.Linear(hidden, d_z)

    def forward(self, X_t: torch.Tensor):
        # X_t: (B, V)
        s = X_t.mean(dim=0)             # (V,)
        h = self.slice_mlp(s)           # (hidden,)

        topic_ids = torch.arange(self.K, device=X_t.device)
        e = self.topic_emb(topic_ids)   # (K, hidden)

        h_k = h.unsqueeze(0) + e        # (K, hidden) addition of embeddings randomly initialised

        mu = self.to_mu(h_k)            # (K, d_z)
        logvar = self.to_logvar(h_k)    # (K, d_z)
        return mu, logvar

    def sample(self, X_t: torch.Tensor):
        mu, logvar = self.forward(X_t)
        eps = torch.randn_like(mu)
        Z = mu + torch.exp(0.5 * logvar) * eps

        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        return Z, kl, mu, logvar
    

class SlicePriorNet(nn.Module):
    """
    m_t = g(Zt)  where Zt is (K,d)
    Output: (K,) prior mean for logistic-normal z.
    """
    def __init__(self, d: int, hidden: int = None):
        super().__init__()
        h = hidden if hidden is not None else d
        self.mlp = nn.Sequential(
            nn.Linear(d, h),
            nn.GELU(),
            nn.Linear(h, 1)  # per-topic scalar
        )

    def forward(self, Zt: torch.Tensor) -> torch.Tensor:
        # Zt: (K,d) -> (K,)
        return self.mlp(Zt).squeeze(-1)

def build_doc_word_ids_by_t(slices, vocab):
    """
    slices: dict[t] -> DataFrame with column 'text'
    vocab:  list[str]

    returns:
        dict[t] -> List[List[int]]
    """
    vocab_map = {w: i for i, w in enumerate(vocab)}
    doc_word_ids_by_t = {}

    for t, df_slice in slices.items():
        docs_ids = []
        for text in df_slice["text"]:
            # simple whitespace tokenization (same as eval scripts)
            #words = text.strip().split()
            words = text.strip().lower().split()
            wids = [vocab_map[w] for w in words if w in vocab_map]
            docs_ids.append(wids)

        doc_word_ids_by_t[int(t)] = docs_ids

    return doc_word_ids_by_t

def get_dataset_root(args):
    # Example: datasets/NYT_jsonl → NYT
    base = os.path.basename(args.data_dir)
    dataset_name = base.split("_")[0]        # "NYT_jsonl" → "NYT"
    return os.path.join("datasets", dataset_name)



def compute_epoch_loss_on_split_bow(
    idx_by_t,
    sorted_slices,
    args,
    device,
    V: int,
    E_words,
    doc_word_ids_by_t,
    tw_decoder,
    theta_encoder,
    gt,
    crf,
    slice_topic_encoder,
):
    """
    Validation loss for your GT+VAE model with HDP/CRF topic weights.
    """
    gt.eval()
    tw_decoder.eval()
    theta_encoder.eval()
    slice_topic_encoder.eval()
    if crf is not None:
        crf.eval()

    z_bs = getattr(args, "z_slice_batch", 512)
    bs   = getattr(args, "batch_size", 512)
    phi_floor = float(getattr(args, "phi_floor", 1e-3))

    with torch.no_grad():
        # ----------------------------------------
        # 1) Infer Z_t and accumulate z-KL
        # ----------------------------------------
        Z_sample = {}
        L_zkl = torch.zeros((), device=device)
        used_z_slices = 0

        Kmax = int(getattr(args, "K_max", getattr(args, "K")))
        d_z = int(E_words.shape[-1])

        for t in sorted_slices:
            idx = idx_by_t.get(t, None)
            if idx is None or len(idx) == 0:
                Z_sample[t] = torch.randn(Kmax, d_z, device=device) * 0.01
                continue

            pick = np.random.choice(idx, size=min(len(idx), z_bs), replace=False)
            wids_batch = [doc_word_ids_by_t[t][int(i)] for i in pick]
            X_t = bow_from_wids_batch(wids_batch, V=V, device=device)

            Zt, kl_z, _, _ = slice_topic_encoder.sample(X_t)  # (Kmax,d_z)
            Z_sample[t] = Zt
            L_zkl = L_zkl + kl_z
            used_z_slices += 1

        L_zkl = L_zkl / max(1, used_z_slices)
        # ----------------------------------------
        # 2.75) Cache pi ONCE (critical if sample_pi=True)
        # ----------------------------------------
        pi_by_t = None
        if crf is not None:
            # pi_all: (num_slices, Kmax)
            pi_all = crf.pi_all()
            pi_by_t = {}
            for t in sorted_slices:
                pi_t = pi_all[t]
                pi_t = pi_t / (pi_t.sum() + 1e-12)

                pi_by_t[t] = pi_t


        # ----------------------------------------
        # 2) GT refinement
        # ----------------------------------------
        Z_ref = {}
        if args.use_gt == 0:
            print("Validation : no graph transformer!!!")
            for t in sorted_slices:
                Z_ref[t] = Z_sample[t]
        else:
            print("Using graph transformer !!")
            t0 = sorted_slices[0]
            Z_ref[t0] = Z_sample[t0]
            Z_prev = Z_ref[t0]

            for i in range(1, len(sorted_slices)):
                t = sorted_slices[i]

                Z_curr = Z_sample[t]
                K_here = Z_prev.size(0)

                g_prev = gate_from_pi(pi_by_t[sorted_slices[i-1]])
                g_curr = gate_from_pi(pi_by_t[t])

                edges = build_competitive_edges(
                    Z_prev, Z_curr,
                    top_influence=getattr(args, "topk_intra", 4),
                    sim_threshold=getattr(args, "sim_threshold", 0.5),
                    anti_sim_threshold=getattr(args, "anti_sim_threshold", 0.2),
                    g_t=g_prev,
                    g_next=g_curr,
                )

                Z_pair = torch.cat([Z_prev, Z_curr], dim=0)
                node_gate = torch.cat([g_prev, g_curr], dim=0)  # (2K,)

                slice_ids = torch.cat([
                    torch.full((K_here,), sorted_slices[i - 1], device=device, dtype=torch.long),
                    torch.full((K_here,), t,                 device=device, dtype=torch.long),
                ])

                Z_out = gt(
                    Z_pair,
                    edges,
                    slice_ids=slice_ids,
                    node_gate=node_gate,
                )

                Z_prev = Z_out[K_here:K_here + K_here]
                Z_ref[t] = Z_prev


        # ----------------------------------------
        # 3) Doc likelihood + theta KL 
        # ----------------------------------------
        L_doc = torch.tensor(0.0, device=device)
        L_kl  = torch.tensor(0.0, device=device)

        total_docs  = 0
        used_slices = 0

        for t in sorted_slices:
            idx = idx_by_t.get(t, None)
            if idx is None or len(idx) == 0:
                continue

            Zt = Z_ref[t]  # (Kmax,d_z)
            if crf is not None:
                pi_t = pi_by_t[t]
            else:
                pi_t = torch.full((Zt.size(0),), 1.0 / Zt.size(0), device=device)

            # Gate topics for decoding
            Zt_gated = Zt * pi_t.unsqueeze(-1)
            beta = tw_decoder(Zt_gated, E_words)

            # ----- SAFE Dirichlet mean-log prior for theta encoder -----
            if crf is not None:
                # Start from intended Dir(alpha * pi_t)
                phi_t = crf.alpha * pi_t
                phi_t = phi_t.clamp_min(phi_floor)
                # Re-normalize to keep sum(phi_t)=alpha (true Dir strength)
                phi_t = phi_t / (phi_t.sum() + 1e-12) * crf.alpha
            else:
                # uniform Dirichlet prior (sum=1)
                phi_t = torch.full_like(pi_t, 1.0 / pi_t.numel()).clamp_min(phi_floor)
                phi_t = phi_t / (phi_t.sum() + 1e-12)

            m_t = digamma(phi_t) - digamma(phi_t.sum())

            slice_doc_loss = torch.tensor(0.0, device=device)
            slice_kl_sum   = torch.tensor(0.0, device=device)
            valid_docs = 0

            for b0 in range(0, len(idx), bs):
                batch_ids = idx[b0:b0 + bs]
                wids_batch = [doc_word_ids_by_t[t][int(i)] for i in batch_ids]
                X = bow_from_wids_batch(wids_batch, V=V, device=device)

                theta, kl_theta = theta_encoder(
                    X, prior_mean=m_t, n_samples=getattr(args, "mc_samples", 1)
                )
                slice_kl_sum = slice_kl_sum + kl_theta

                if theta.dim() == 2:
                    pw = theta @ beta
                else:
                    pw = torch.einsum("sbk,kv->sbv", theta, beta).mean(dim=0)

                log_pw = torch.log(pw + 1e-12)      # (B,V)
                doc_nll = -(X * log_pw).sum(dim=1)  # (B,)
                slice_doc_loss += doc_nll.sum()
                valid_docs += int(X.size(0))

            if valid_docs > 0:
                used_slices += 1
                total_docs  += valid_docs
                L_doc += slice_doc_loss
                L_kl  += slice_kl_sum / max(1, valid_docs)

        if total_docs == 0 or used_slices == 0:
            return float("inf")

        # Normalize like your original: doc by total docs, kl by used slices
        L_doc = L_doc / total_docs
        L_kl  = L_kl  / used_slices

        L = (
            args.lambda_doc * L_doc
            + getattr(args, "lambda_kl", 0.0)  * L_kl
            + getattr(args, "lambda_zkl", 0.0) * L_zkl
        )
        return float(L.item())
    

def evaluate_coherence_for_dir(epoch, args):


    dataset_root = get_dataset_root(args)
    vocab_file = os.path.join(dataset_root, "vocab.txt")
    word_emb_npz = os.path.join(dataset_root, "word_embeddings_encoder.npz")
    train_texts = os.path.join(dataset_root, "train_texts.txt")
    train_times = os.path.join(dataset_root, "train_times.txt")

    # NEW: full checkpoint that includes Z_by_t + tw_decoder
    model_ckpt = os.path.join(args.artifacts_dir, "best_model.pt")

    cmd = [
        "python", "-m", "scripts.ANTM_style",
        "--model_ckpt", model_ckpt,
        "--vocab_file", vocab_file,
        "--word_emb_npz", word_emb_npz,
        "--train_texts", train_texts,
        "--train_times", train_times,
        "--topn", "15",
        "--coherence", "c_v",
        "--save_dir", os.path.join(args.artifacts_dir, "eval_out"),
    ]

    out = subprocess.run(cmd, capture_output=True, text=True)
    print(out.stdout)
    if out.stderr:
        print("[eval-warning]", out.stderr)



def to_t(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def run_train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)

    with open(args.vocab_file, "r", encoding="utf-8") as f:
        vocab = [w.strip() for w in f if w.strip()]
    V = len(vocab)
    Kmax = args.K_max if args.K_max is not None else args.K
    print(f"[info] using K_max={Kmax}")

    df_train = load_jsonl_folder(args.data_dir, split="train")
    df_train, mapping = remap_times(df_train)
    slices = slice_by_time(df_train)

    num_slices = len(slices)
    sorted_slices = list(range(num_slices))

    doc_word_ids_by_t = build_doc_word_ids_by_t(slices, vocab)

    E_words = np.load(args.word_emb_npz)["embeddings"].astype("float32")
    E_words = torch.tensor(E_words, device=device)
    E_words = F.normalize(E_words, dim=-1)
    d_z = int(E_words.shape[1])

    print(f"[info] V={V} | d_z={d_z} | num_slices={num_slices}")

    val_ratio = getattr(args, "val_ratio", 0.1)
    val_seed  = getattr(args, "val_seed", 42)
    train_idx_by_t, val_idx_by_t = split_train_val_indices_by_slice(
        slices, val_ratio=val_ratio, seed=val_seed
    )
    print(f"[split] val_ratio={val_ratio} seed={val_seed}")
    disable_crf = bool(args.disable_crf)
    crf = HDPTopicWeights(
    K=Kmax,
    num_slices=num_slices,
    gamma=5,
    alpha=10,
    use_sparsemax=False,
    disable_crf = disable_crf, #for ablation "True"
).to(device)

    


    slice_topic_encoder = SliceTopicEncoder(
        d_in=V, d_z=d_z, K=Kmax,
        hidden=getattr(args, "z_enc_hidden", 512),
        dropout=getattr(args, "z_enc_dropout", 0.1),
            ).to(device)

    d_z = int(E_words.shape[1])   # topic space dim (must match Z_t and decoder)

    gt = TopicGraphTransformer(
        d=d_z,
        num_heads=args.heads,
        num_layers=args.layers,
        dropout=0.3,
        use_batchnorm=True,
        rtpe_clip=8,
        refine_alpha=getattr(args, "refine_alpha", 0.001),
        use_global_memory=True,
        num_slices=num_slices,    
    ).to(device)

    # -------- Optimizer --------
    tw_decoder = TopicWordAttention(d=d_z).to(device)
    theta_encoder = ThetaEncoderLN(
    d_in=V,
    K=Kmax,
    hidden=(args.enc_hidden if args.enc_hidden is not None else 512),
    dropout=args.enc_dropout,
    tau=getattr(args, "theta_tau", 1.0),
        ).to(device)

    lr_main = args.lr
    lr_gt   = args.lr * 0.1
    lr_z    = args.lr * 0.1
    lr_dec  = args.lr * 0.05
    
    opt = torch.optim.AdamW(
    [
        {"params": gt.parameters(),                "lr": lr_gt},
        {"params": slice_topic_encoder.parameters(),"lr": lr_z},
        {"params": tw_decoder.parameters(),        "lr": lr_dec},
        {"params": theta_encoder.parameters(),     "lr": lr_main},
        {"params": crf.parameters(),               "lr": 5e-3}, 
    ],
    weight_decay=args.weight_decay
)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt,
    mode="min",
    factor=0.5,     
    patience=15,    
    threshold=1e-6,  
    threshold_mode="abs",
    cooldown=0,     
    min_lr=1e-6,  
    verbose=True
)

    # -------- Early stopping state --------
    best_val = float("inf")

    # -------- training loop --------
    start_time_total = time.time()
    best_epoch = -1
    no_improve = 0
    patience = args.early_stop_patience if hasattr(args, "early_stop_patience") else 30
    lambda_crf_target = args.lambda_crf       
    crf_warmup_epochs = getattr(args, "crf_warmup_epochs", 30)  

    for ep in range(1, args.epochs + 1):
        print("\n \n \n[EPoch DIAGNOSTICS]",ep)
        gt.train()
        slice_topic_encoder.train()
        theta_encoder.train()
        tw_decoder.train()
        crf.train()
        opt.zero_grad(set_to_none=True)
        lambda_crf_ep = linear_warmup(ep, target=lambda_crf_target, warmup_epochs=crf_warmup_epochs)

        # =========================================
        # 1) Sample slice topics Z_t ~ q(Z_t | X_t)
        # =========================================
        Z_sample = {}
        L_zkl = torch.zeros((), device=device)
        z_bs = getattr(args, "z_slice_batch", 64)
        used = 0

        for t in sorted_slices:
            pool = train_idx_by_t.get(t, None)
            if pool is None or pool.size == 0:
                Zt_use = torch.randn(Kmax, d_z, device=device) * 0.01
                kl_z = torch.zeros((), device=device)
            else:
                if pool.shape[0] > z_bs:
                    pick = np.random.choice(pool, size=z_bs, replace=False)
                else:
                    pick = pool

                wids_batch = [doc_word_ids_by_t[t][int(i)] for i in pick]
                X_t = bow_from_wids_batch(wids_batch, V=V, device=device)
                Zt_use, kl_z, mu_t, logvar_t = slice_topic_encoder.sample(X_t)
                used += 1

            Z_sample[t] = Zt_use
            L_zkl = L_zkl + kl_z
        L_zkl = L_zkl / max(1, used)

        # ============================================================
        # 2.75) Cache pi ONCE per epoch (consistent even if sample_pi=True)
        # ============================================================
        pi_all = crf.pi_all()  # (num_slices, Kmax)
        pi_by_t = {}
        for t in sorted_slices:
            pi_t = pi_all[t]
            pi_t = pi_t.clamp_min(1e-8)
            pi_t = pi_t / pi_t.sum()
            pi_by_t[t] = pi_t


        # =========================================
        # 2) GT refinement using sampled Z
        # =========================================
        Z_ref = {}
        if args.use_gt == 0:
            print("no graph transformer")
            for t in sorted_slices:
                Z_ref[t] = Z_sample[t]

        else:
            # first slice: no past → keep as-is (or light self-refine)
            t0 = sorted_slices[0]
            Z_ref[t0] = Z_sample[t0]        
            Z_prev = Z_ref[t0]
            for i in range(1, len(sorted_slices)):
                t = sorted_slices[i]
                Z_curr = Z_sample[t]          # raw current (no future!)
                K_here = Z_prev.size(0)

                g_prev = gate_from_pi(pi_by_t[sorted_slices[i-1]])
                g_curr = gate_from_pi(pi_by_t[t])

                edges = build_competitive_edges(
                    Z_prev, Z_curr,
                    top_influence=getattr(args, "topk_intra", 4),
                    sim_threshold=getattr(args, "sim_threshold", 0.5),
                    anti_sim_threshold=getattr(args, "anti_sim_threshold", 0.2),
                    g_t=g_prev,
                    g_next=g_curr,
                )

                Z_pair = torch.cat([Z_prev, Z_curr], dim=0)   # (2K, d)
                node_gate = torch.cat([g_prev, g_curr], dim=0)  # (2K,)

                slice_ids = torch.cat([
                    torch.full((K_here,), sorted_slices[i-1], device=device, dtype=torch.long),
                    torch.full((K_here,), t,                 device=device, dtype=torch.long),
                ])

                Z_out = gt(
                        Z_pair,
                        edges,
                        slice_ids=slice_ids,
                        node_gate=node_gate,
                    )
                Z_prev = Z_out[K_here:K_here + K_here]   
                Z_ref[t] = Z_prev


        # =========================================
        # 3) Your existing doc VAE loss (L_doc + L_kl)
        # =========================================
        L_doc = torch.tensor(0.0, device=device)
        L_kl  = torch.tensor(0.0, device=device)
        Z_gated_by_t = {}          # {t: (K,d)}
        for t in sorted_slices:
            Zt = Z_ref[t]
            pi_t = pi_by_t[t]                    # (Kmax,)
            Zt_gated = Zt * pi_t.unsqueeze(-1)   # (Kmax,d)
            Z_gated_by_t[t] = Zt_gated

            beta = tw_decoder(Zt_gated, E_words)
            # Dirichlet params for E[log theta]  
            phi_t = crf.alpha * pi_t                       # sum ~= alpha
            phi_t = phi_t.clamp_min(1e-3)                 
            phi_t = phi_t / phi_t.sum() * crf.alpha        # re-normalize to keep total concentration = alpha
            m_t = digamma(phi_t) - digamma(phi_t.sum())
            pool = train_idx_by_t[t]
            if pool.size == 0:
                continue

            bs = getattr(args, "batch_size", 512)
            slice_doc_loss = torch.tensor(0.0, device=device)
            valid_docs = 0
            slice_kl_sum = torch.tensor(0.0, device=device)

            for b0 in range(0, pool.shape[0], bs):
                idx = pool[b0:b0+bs]

                wids_batch = [doc_word_ids_by_t[t][int(i)] for i in idx]
                X = bow_from_wids_batch(wids_batch, V=V, device=device)

                theta, kl = theta_encoder(
                    X, prior_mean=m_t, n_samples=getattr(args, "mc_samples", 1)
                )
                slice_kl_sum = slice_kl_sum + kl

                # Build word probabilities
                if theta.dim() == 2:
                    # theta: (B,K)
                    pw = theta @ beta                      # (B,V)
                else:
                    # theta: (S,B,K)
                    pw = torch.einsum("sbk,kv->sbv", theta, beta).mean(dim=0)  # (B,V)
                # Batch NLL using BoW counts (correct; no per-doc loop)
                log_pw = torch.log(pw + 1e-12)              # (B,V)
                doc_nll = -(X * log_pw).sum(dim=1)          # (B,)
                slice_doc_loss += doc_nll.sum()             # accumulate sum over docs in batch
                valid_docs += int(X.size(0))                # all docs in batch


            if valid_docs > 0:
                L_doc = L_doc +slice_doc_loss / valid_docs
                L_kl  = L_kl  +slice_kl_sum / max(1, valid_docs)

        T = len(sorted_slices)
        L_doc /= T
        L_kl  /= T
        L_crf = crf.loss()

        # =========================================
        # 4) Final loss: add topic KL (Z KL)
        # =========================================
        L = (
            args.lambda_doc * L_doc
            + args.lambda_kl  * L_kl
            + args.lambda_zkl * L_zkl
            + lambda_crf_ep * L_crf
        )
        L.backward()
        opt.step()
        if ep <= 5 or ep % 10 == 0:
            print(f"[ep {ep:03d}] lambda_crf={lambda_crf_ep:.4f} (target={lambda_crf_target})")

        gt.eval()
        val_loss = compute_epoch_loss_on_split_bow(
            idx_by_t=val_idx_by_t,
            sorted_slices=sorted_slices,
            args=args,
            device=device,
            V=V,
            E_words=E_words,
            doc_word_ids_by_t=doc_word_ids_by_t,
            tw_decoder=tw_decoder,
            theta_encoder=theta_encoder,
            gt=gt,
            crf=crf,
            slice_topic_encoder=slice_topic_encoder,
        )

        val_loss_value = float(val_loss)

        # update LR based on validation loss plateau
        scheduler.step(val_loss_value)
        lr_now = opt.param_groups[0]["lr"]
        print(
            f"[ep {ep:03d}] L={L.item():.4f} | doc={L_doc.item():.4f} | kl={L_kl.item():.4f} "
            f"| val={val_loss_value:.4f} | lr={lr_now:.2e} "
        )

        improved = val_loss_value < best_val - 1e-4
        if improved:
            best_val = val_loss_value
            best_epoch = ep
            no_improve = 0
            Z_save = {t: Z_gated_by_t[t].detach().cpu() for t in sorted_slices}
            
            best_ckpt = {
                "epoch": best_epoch,
                "val": best_val,
                "gt": gt.state_dict(),
                "tw_decoder": tw_decoder.state_dict(),
                "slice_topic_encoder": slice_topic_encoder.state_dict(),
                "Z_by_t": Z_save,
                "pi_by_t": {t: pi_by_t[t].detach().cpu() for t in sorted_slices},

                "crf": crf.state_dict(),
                "crf_export": crf.export_state(),
            }

            ckpt_path = os.path.join(args.artifacts_dir, "best_model.pt")
            torch.save(best_ckpt, ckpt_path)
            print(f"[save] best full checkpoint -> {ckpt_path}")
        else:
            no_improve += 1
            print(f"[early-stop] best epoch is at {best_epoch} and early stop triggered ({no_improve}/{patience}) \n")


        if no_improve >= patience:
            print(f"[early-stop] STOP: no improvement for {patience} epochs. best_val={best_val:.6f} at epoch={best_epoch}")
            print("Calcuting coherence after early stopped Triggered !!!")
            break

    #evaluate_coherence_for_dir(best_epoch, args)
    total_time = time.time() - start_time_total
    print(f"[done] training finished in {total_time/60:.2f} minutes | best_val={best_val:.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_gt", type=int, default=1,
                help="1=use graph transformer, 0=disable")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda_crf", type=float, default=0.2,
                help="Weight for CRF/HDP coupling regularizer KL(pi_t || beta).")
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--cache_dir", type=str, default="cache")
    ap.add_argument("--artifacts_dir", type=str, default="artifacts")
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--topk_intra", type=int, default=3)
    ap.add_argument("--topm_temporal", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.28)
    ap.add_argument("--delta", type=float, default=0.2)
    ap.add_argument("--lambda_doc", type=float, default=0.5)
    ap.add_argument("--lambda_div", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--val_seed", type=int, default=42)
    ap.add_argument("--early_stop_patience", type=int, default=30)
    ap.add_argument("--cache_test_dir", type=str, default="cache_test")
    ap.add_argument("--word_emb_npz", type=str, required=True)
    ap.add_argument("--vocab_file", type=str, required=True)
    ap.add_argument("--use_global_memory", action="store_true")
    ap.add_argument("--lambda_kl", type=float, default=0.2,help="Weight for KL(q(z|x)||N(0,I)) in logistic-normal VAE")
    ap.add_argument("--mc_samples", type=int, default=1)
    ap.add_argument("--enc_hidden", type=int, default=None)
    ap.add_argument("--enc_dropout", type=float, default=0.0)
    ap.add_argument("--theta_tau", type=float, default=2.0,help="Temperature for sparsemax(theta) mapping; <1 makes theta peakier.")
    ap.add_argument("--model_ckpt", type=str, default=None)
    ap.add_argument("--lambda_zkl", type=float, default=0.05,help="KL weight for variational slice topics Z_t.")
    ap.add_argument("--K_max", type=int, default=None,help="Maximum number of topics (overcomplete). If None, uses --K.")
    ap.add_argument("--disable_crf",type=str2bool,default=False,help="True = disable CRF, False = enable CRF")
    ap.add_argument("--lambda_gate", type=float, default=0.01,help="Sparsity penalty weight for topic gates (mean gate).") #not_used
    ap.add_argument("--gate_tau", type=float, default=0.5, help="Threshold for reporting active topics K_t (for logging).") #not_used


    args = ap.parse_args()
    set_seed(args.seed)
    run_train(args)

if __name__ == "__main__":
    main()
