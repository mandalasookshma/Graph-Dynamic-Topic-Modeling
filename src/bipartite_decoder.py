import torch
import torch.nn as nn
import math

class TopicWordAttention(nn.Module):
    """
    Bipartite Topic–Word Graph Decoder
    """
    def __init__(self, d):
        super().__init__()
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)

    def forward(self, Z, E_words):
        """
        Z:        (K, d)   refined topics
        E_words:  (V, d)   word embeddings

        returns:
        beta:     (K, V)   topic-word distributions
        """
        Q = self.Wq(Z)                    # (K, d)
        K = self.Wk(E_words)              # (V, d)
        logits = Q @ K.T / math.sqrt(Z.size(1))
        beta = torch.softmax(logits, dim=-1)
        return beta

