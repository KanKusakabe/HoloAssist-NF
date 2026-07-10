"""Build fixed-length history windows for the sequential step model."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from . import config as C


def load_windows():
    df = pd.read_parquet(C.SEQ_PARQUET).sort_values(["clip", "order"])
    vocab = json.loads(C.VOCAB_JSON.read_text())
    norm = json.loads(C.NORM_JSON.read_text())
    H = C.HISTORY
    md, mg = norm["log_dur"]
    gd, gg = norm["log_gap"]

    hist_a, hist_n, hist_c, cur_a, cur_n, y = [], [], [], [], [], []
    clips, orders, splits, mis, itv = [], [], [], [], []

    for clip, g in df.groupby("clip", sort=False):
        a = g["a_code"].to_numpy() + 1          # +1: reserve 0 for padding
        n = g["n_code"].to_numpy() + 1
        cd = (g["log_dur"].to_numpy() - md) / mg
        cg = (g["log_gap"].to_numpy() - gd) / gg
        c = np.stack([cd, cg], axis=1)
        L = len(g)
        sp = g["split"].to_numpy(); mm = g["mistake"].to_numpy(); ii = g["intervention"].to_numpy()
        for t in range(L):
            lo = max(0, t - H)
            ha, hn, hc = a[lo:t], n[lo:t], c[lo:t]
            pad = H - len(ha)
            hist_a.append(np.concatenate([np.zeros(pad, int), ha]))
            hist_n.append(np.concatenate([np.zeros(pad, int), hn]))
            hist_c.append(np.concatenate([np.zeros((pad, 2)), hc]))
            cur_a.append(a[t]); cur_n.append(n[t]); y.append(c[t])
            clips.append(clip); orders.append(t); splits.append(sp[t])
            mis.append(mm[t]); itv.append(ii[t])

    tens = dict(
        hist_a=torch.tensor(np.array(hist_a), dtype=torch.long),
        hist_n=torch.tensor(np.array(hist_n), dtype=torch.long),
        hist_c=torch.tensor(np.array(hist_c), dtype=torch.float32),
        cur_a=torch.tensor(np.array(cur_a), dtype=torch.long),
        cur_n=torch.tensor(np.array(cur_n), dtype=torch.long),
        y=torch.tensor(np.array(y), dtype=torch.float32),
    )
    meta = pd.DataFrame(dict(clip=clips, order=orders, split=splits,
                             mistake=mis, intervention=itv))
    return tens, meta, vocab


def to_device(tens, dev, idx=None):
    if idx is None:
        return {k: v.to(dev) for k, v in tens.items()}
    return {k: v[idx].to(dev) for k, v in tens.items()}
