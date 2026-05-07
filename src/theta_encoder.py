# src/theta_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.sparsemax import sparsemax

def kl_normal_to_normal_diag(mu, logvar, prior_mu, prior_logvar):
    """
    KL( N(mu, diag(exp(logvar))) || N(prior_mu, diag(exp(prior_logvar))) )
    returns: scalar (sum over batch and dims) OR (batch,) if reduce=False
    """
    # all shapes: (B,K)
    var = torch.exp(logvar)
    prior_var = torch.exp(prior_logvar)

    kl = 0.5 * (
        prior_logvar - logvar
        + (var + (mu - prior_mu) ** 2) / prior_var
        - 1.0
    )
    return kl.sum(dim=-1)  # (B,)

class ThetaEncoderLN(nn.Module):
    """
    Logistic-Normal VAE encoder for document-topic proportions.
    q(z|x) = N(mu(x), diag(sigma^2(x)))
    p(z|t) = N(m_t, I)   (slice-conditioned prior mean)
    theta  = sparsemax(z / tau)   (sparse simplex map)
    """

    def __init__(self, d_in: int, K: int, hidden: int = None, dropout: float = 0.0, tau: float = 1.0):
        super().__init__()
        self.d_in = d_in
        self.K = K
        self.hidden = hidden if hidden is not None else d_in
        self.tau = float(tau)

        self.net = nn.Sequential(
            nn.Linear(d_in, self.hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(self.hidden, K)
        self.logvar = nn.Linear(self.hidden, K)

    def forward(self, X: torch.Tensor, prior_mean: torch.Tensor, n_samples: int = 1):
        """
        X: (B,d)
        prior_mean: (K,) or (B,K)  slice-conditioned mean m_t
        n_samples: Monte Carlo samples
        returns:
          theta: (B,K) if n_samples=1 else (S,B,K)
          kl: scalar (sum over batch)  (you can average outside)
        """
        B = X.size(0)
        h = self.net(X)
        mu = self.mu(h)                  # (B,K)
        logvar = self.logvar(h).clamp(-8.0, 4.0)  # stabilize

        # broadcast prior_mean to (B,K)
        if prior_mean.dim() == 1:
            prior_mu = prior_mean.view(1, -1).expand(B, -1)
        else:
            prior_mu = prior_mean

        # p(z|t) = N(m_t, I)  => prior_logvar = 0
        prior_logvar = torch.zeros_like(mu)

        # reparameterize
        std = torch.exp(0.5 * logvar)
        if n_samples == 1:
            eps = torch.randn_like(std)
            z = mu + eps * std           # (B,K)
            theta = sparsemax(z / self.tau, dim=-1)  # (B,K)
        else:
            eps = torch.randn((n_samples, B, self.K), device=X.device, dtype=X.dtype)
            z = mu.unsqueeze(0) + eps * std.unsqueeze(0)   # (S,B,K)
            theta = sparsemax(z / self.tau, dim=-1)         # (S,B,K)

        # KL per doc, then sum
        kl_per_doc = kl_normal_to_normal_diag(mu, logvar, prior_mu, prior_logvar)  # (B,)
        kl = kl_per_doc.sum()  # scalar
        return theta, kl
