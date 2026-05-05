# crf_weights.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import digamma, gammaln


class Sparsemax(nn.Module):
    def forward(self, input):
        # input: (..., K)
        input = input - input.max(dim=-1, keepdim=True)[0]
        sorted_input, _ = torch.sort(input, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_input, dim=-1)

        k = torch.arange(1, input.size(-1) + 1, device=input.device)
        k = k.view((1,) * (input.dim() - 1) + (-1,))

        support = sorted_input + (1 - cumsum) / k > 0
        k_z = support.sum(dim=-1, keepdim=True).clamp_min(1)

        tau = (cumsum.gather(-1, k_z - 1) - 1) / k_z
        output = torch.clamp(input - tau, min=0)
        return output


class HDPTopicWeights(nn.Module):
    """
    Variational HDP topic weights.

    disable_crf=False:
        Normal HDP behavior

    disable_crf=True:
        Neutral ablation mode:
            - uniform beta
            - uniform pi_t
            - zero loss
            - zero entropy regularizer
    """

    def __init__(
        self,
        K,
        num_slices,
        gamma=0.5,
        alpha=0.5,
        tau_q=10.0,
        eps=1e-8,
        use_sparsemax=False,
        bias_with_log_beta=True,
        sample_pi=False,
        beta_log_clamp=1e-6,
        disable_crf=False,          # NEW
    ):
        super().__init__()

        self.K = int(K)
        self.num_slices = int(num_slices)

        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.tau_q = float(tau_q)
        self.eps = float(eps)

        self.use_sparsemax = bool(use_sparsemax)
        self.bias_with_log_beta = bool(bias_with_log_beta)
        self.sample_pi = bool(sample_pi)
        self.beta_log_clamp = float(beta_log_clamp)

        self.disable_crf = bool(disable_crf)

        # parameters still created so state_dict works
        self.log_a = nn.Parameter(torch.zeros(self.K))
        self.log_b = nn.Parameter(torch.zeros(self.K))
        self.phi_logits = nn.Parameter(torch.zeros(self.num_slices, self.K))

        self.sparsemax = Sparsemax()

    # =====================================================
    # Utilities
    # =====================================================
    def _uniform_vec(self, device):
        return torch.ones(self.K, device=device) / self.K

    def _uniform_mat(self, device):
        return torch.ones(self.num_slices, self.K, device=device) / self.K

    # =====================================================
    # Global sticks
    # =====================================================
    def expected_log_sticks(self):
        a = F.softplus(self.log_a) + self.eps
        b = F.softplus(self.log_b) + self.eps

        E_log_v = digamma(a) - digamma(a + b)
        E_log_1mv = digamma(b) - digamma(a + b)

        log_beta = []
        for k in range(self.K):
            if k == 0:
                val = E_log_v[k]
            else:
                val = E_log_v[k] + torch.sum(E_log_1mv[:k])
            log_beta.append(val)

        return torch.stack(log_beta)

    def beta_mean(self):
        if self.disable_crf:
            return self._uniform_vec(self.log_a.device)

        log_beta = self.expected_log_sticks()
        beta = torch.exp(log_beta)
        return beta / (beta.sum() + self.eps)

    # =====================================================
    # KL
    # =====================================================
    def stick_kl(self):
        a = F.softplus(self.log_a) + self.eps
        b = F.softplus(self.log_b) + self.eps
        gamma = torch.tensor(self.gamma, device=a.device)

        logB_prior = (
            gammaln(torch.tensor(1.0, device=a.device))
            + gammaln(gamma)
            - gammaln(1.0 + gamma)
        )

        logB_q = gammaln(a) + gammaln(b) - gammaln(a + b)

        KL = (
            logB_prior - logB_q
            + (a - 1) * (digamma(a) - digamma(a + b))
            + (b - gamma) * (digamma(b) - digamma(a + b))
        )

        return KL.sum()

    # =====================================================
    # Gate
    # =====================================================
    def _gate(self, logits_TK):
        if self.use_sparsemax:
            g = self.sparsemax(logits_TK)
            g = g.clamp_min(self.eps)
            g = g / (g.sum(dim=-1, keepdim=True) + self.eps)
            return g
        else:
            return torch.softmax(logits_TK, dim=-1)

    # =====================================================
    # Posterior phi
    # =====================================================
    def phi_posterior(self):
        if self.disable_crf:
            return self._uniform_mat(self.log_a.device)

        beta = self.beta_mean()

        beta_safe = beta.clamp_min(self.beta_log_clamp)
        beta_safe = beta_safe / (beta_safe.sum() + self.eps)

        logits = self.phi_logits

        if self.bias_with_log_beta:
            logits = logits + torch.log(beta_safe.unsqueeze(0))

        pi_mean = self._gate(logits)

        phi = self.tau_q * pi_mean
        return phi.clamp_min(self.eps)

    # =====================================================
    # pi
    # =====================================================
    def pi_all(self):
        if self.disable_crf:
            return self._uniform_mat(self.log_a.device)

        phi = self.phi_posterior()

        if not self.sample_pi:
            return phi / (phi.sum(dim=-1, keepdim=True) + self.eps)

        g = torch.distributions.Gamma(phi, torch.ones_like(phi)).rsample()
        return g / (g.sum(dim=-1, keepdim=True) + self.eps)

    def pi(self, t):
        if self.disable_crf:
            print("No CRF !!")
            return self._uniform_vec(self.log_a.device)

        phi = self.phi_posterior()[t]

        if not self.sample_pi:
            return phi / (phi.sum() + self.eps)

        g = torch.distributions.Gamma(phi, torch.ones_like(phi)).rsample()
        return g / (g.sum() + self.eps)

    # =====================================================
    # Slice KL
    # =====================================================
    def slice_kl(self):
        if self.disable_crf:
            return torch.tensor(0.0, device=self.log_a.device)

        beta = self.beta_mean().clamp_min(self.beta_log_clamp)
        beta = beta / (beta.sum() + self.eps)

        alpha_beta = (self.alpha * beta).clamp_min(self.eps)
        phi = self.phi_posterior()

        logB_q = torch.sum(gammaln(phi), dim=1) - gammaln(torch.sum(phi, dim=1))
        logB_p = torch.sum(gammaln(alpha_beta)) - gammaln(torch.sum(alpha_beta))

        dig_phi = digamma(phi)
        dig_sum = digamma(torch.sum(phi, dim=1, keepdim=True))

        term = (phi - alpha_beta.unsqueeze(0)) * (dig_phi - dig_sum)

        KL = logB_p - logB_q + term.sum(dim=1)

        return KL.sum()

    # =====================================================
    # Total loss
    # =====================================================
    def loss(self):
        if self.disable_crf:
            print("No CRF !!")
            return torch.tensor(0.0, device=self.log_a.device)

        return (self.stick_kl() / self.K) + (self.slice_kl() / self.num_slices)

    # =====================================================
    # Diagnostics
    # =====================================================
    def posterior_score(self, t, Zt, pi_t=None):
        if pi_t is None:
            pi_t = self.pi(t)

        Z_norm = torch.norm(Zt, dim=-1)
        score = pi_t * Z_norm

        return score / (score.sum() + 1e-12)

    def effective_K_slice(self, t, Zt, pi_t=None):
        score = self.posterior_score(t, Zt, pi_t=pi_t)
        H = -(score * torch.log(score + 1e-12)).sum()
        return torch.exp(H)

    def slice_entropy(self):
        if self.disable_crf:
            return torch.tensor(0.0, device=self.log_a.device)

        phi = self.phi_posterior()
        pi = phi / (phi.sum(dim=-1, keepdim=True) + self.eps)

        return (-(pi * torch.log(pi + 1e-12)).sum(dim=1)).sum()

    # =====================================================
    # Export
    # =====================================================
    @torch.no_grad()
    def export_state(self):
        return {
            "beta_global": self.beta_mean().detach().cpu(),
            "phi_logits": self.phi_logits.detach().cpu(),
            "a": F.softplus(self.log_a).detach().cpu(),
            "b": F.softplus(self.log_b).detach().cpu(),
            "alpha": self.alpha,
            "tau_q": self.tau_q,
            "gamma": self.gamma,
            "use_sparsemax": self.use_sparsemax,
            "bias_with_log_beta": self.bias_with_log_beta,
            "sample_pi": self.sample_pi,
            "beta_log_clamp": self.beta_log_clamp,
            "disable_crf": self.disable_crf,
        }