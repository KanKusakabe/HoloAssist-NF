"""Sequential conditional density over a step's timing (log_dur, log_gap).

ConditionEncoder: GRU over the history of previous steps (action/noun embeddings
+ their timing) plus the CURRENT step's action identity -> conditioning vector.
Two interchangeable heads share the same interface so we can honestly ask
"does the Flow beat a Gaussian mixture?" (the lesson from Experiment A):

  * MDNHead  -- diagonal Gaussian mixture (the baseline).
  * FlowHead -- zuko conditional Neural Spline Flow (the model of interest).

SURPRISE = -log p( timing | history, action ).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import zuko

DIM = 2  # (log_dur, log_gap)


class ConditionEncoder(nn.Module):
    def __init__(self, n_a, n_n, a_emb=16, n_emb=16, gru_hidden=64, out_dim=64):
        super().__init__()
        self.a_emb = nn.Embedding(n_a, a_emb, padding_idx=0)
        self.n_emb = nn.Embedding(n_n, n_emb, padding_idx=0)
        self.gru = nn.GRU(a_emb + n_emb + DIM, gru_hidden, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(gru_hidden + a_emb + n_emb, out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim), nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, hist_a, hist_n, hist_c, cur_a, cur_n):
        seq = torch.cat([self.a_emb(hist_a), self.n_emb(hist_n), hist_c], dim=-1)
        _, h = self.gru(seq)
        h = h.squeeze(0)
        cur = torch.cat([self.a_emb(cur_a), self.n_emb(cur_n)], dim=-1)
        return self.mlp(torch.cat([h, cur], dim=-1))


class FlowHead(nn.Module):
    def __init__(self, ctx_dim, transforms=3, hidden=(64, 64)):
        super().__init__()
        self.flow = zuko.flows.NSF(features=DIM, context=ctx_dim,
                                   transforms=transforms, hidden_features=hidden)

    def log_prob(self, c, y):
        return self.flow(c).log_prob(y)

    def sample(self, c, n=1):
        return self.flow(c).sample((n,)).permute(1, 0, 2)


class MDNHead(nn.Module):
    """Diagonal Gaussian mixture baseline."""

    def __init__(self, ctx_dim, k=5):
        super().__init__()
        self.k = k
        self.net = nn.Linear(ctx_dim, k * (1 + 2 * DIM))

    def _params(self, c):
        o = self.net(c)
        logit = o[:, :self.k]
        mu = o[:, self.k:self.k + self.k * DIM].view(-1, self.k, DIM)
        log_sd = o[:, self.k + self.k * DIM:].view(-1, self.k, DIM).clamp(-6, 3)
        return logit, mu, log_sd

    def log_prob(self, c, y):
        logit, mu, log_sd = self._params(c)
        y = y.unsqueeze(1)
        comp = -0.5 * (((y - mu) / log_sd.exp()) ** 2 + 2 * log_sd
                       + torch.log(torch.tensor(2 * torch.pi))).sum(-1)
        return torch.logsumexp(F.log_softmax(logit, -1) + comp, dim=-1)

    def sample(self, c, n=1):
        logit, mu, log_sd = self._params(c)
        idx = torch.multinomial(F.softmax(logit, -1), n, replacement=True)  # [B,n]
        b = torch.arange(c.shape[0]).unsqueeze(1)
        m = mu[b, idx]
        s = log_sd[b, idx].exp()
        return m + s * torch.randn_like(m)


class StepModel(nn.Module):
    def __init__(self, n_a, n_n, head="flow", **head_kw):
        super().__init__()
        self.encoder = ConditionEncoder(n_a, n_n)
        self.head = (FlowHead(self.encoder.out_dim, **head_kw) if head == "flow"
                     else MDNHead(self.encoder.out_dim, **head_kw))

    def condition(self, b):
        return self.encoder(b["hist_a"], b["hist_n"], b["hist_c"], b["cur_a"], b["cur_n"])

    def log_prob(self, b):
        return self.head.log_prob(self.condition(b), b["y"])

    def nll(self, b):
        return -self.log_prob(b).mean()
