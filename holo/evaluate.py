"""What does the timing SURPRISE buy us? Three questions on held-out clips:

  1. mistake detection     -- AUC(surprise -> this step is a "Wrong Action")
  2. intervention detection -- AUC(surprise -> instructor intervenes now)
  3. anticipation          -- AUC(surprise -> intervention within the next W steps)

Each is reported for the Flow AND the MDN baseline (Experiment-A lesson), plus a
trivial duration-only baseline.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from . import config as C
from .data import load_windows
from .model import StepModel

W_ANTICIPATE = 3


def auc(scores, labels):
    labels = np.asarray(labels).astype(int)
    n_pos, n_neg = int(labels.sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def _surprise(model, tens, bs=8192):
    out = []
    n = tens["y"].shape[0]
    for i in range(0, n, bs):
        b = {k: v[i:i + bs] for k, v in tens.items()}
        out.append((-model.log_prob(b)).numpy())
    return np.concatenate(out)


def _anticipation_labels(meta):
    lab = np.zeros(len(meta), int)
    itv = meta["intervention"].to_numpy()
    order = meta["order"].to_numpy()
    clip = meta["clip"].to_numpy()
    start = 0
    for i in range(1, len(meta) + 1):
        if i == len(meta) or clip[i] != clip[start]:
            seg = slice(start, i)
            iv = itv[seg]
            for j in range(i - start):
                lab[start + j] = int(iv[j + 1:j + 1 + W_ANTICIPATE].any())
            start = i
    return lab


def main() -> None:
    tens, meta, vocab = load_windows()
    ck = torch.load(C.MODEL_PT, map_location="cpu")
    flow = StepModel(ck["n_a"], ck["n_n"], head="flow"); flow.load_state_dict(ck["state"]); flow.eval()
    mdn = StepModel(ck["n_a"], ck["n_n"], head="mdn"); mdn.load_state_dict(ck["mdn_state"]); mdn.eval()

    s_flow = _surprise(flow, tens)
    s_mdn = _surprise(mdn, tens)
    dur_only = np.abs(tens["y"][:, 0].numpy())      # |z(log_dur)| trivial baseline

    va = (meta["split"].values == "val")
    antic = _anticipation_labels(meta)
    mis = meta["mistake"].to_numpy(); itv = meta["intervention"].to_numpy()

    def block(mask_lab, name):
        return {
            f"{name}_auc_flow": round(auc(s_flow[va], mask_lab[va]), 4),
            f"{name}_auc_mdn": round(auc(s_mdn[va], mask_lab[va]), 4),
            f"{name}_auc_duration_only": round(auc(dur_only[va], mask_lab[va]), 4),
        }

    res = {}
    res.update(block(mis, "mistake"))
    res.update(block(itv, "intervention"))
    res.update(block(antic, "anticipation"))
    res["n_val"] = int(va.sum())
    res["val_mistake_rate"] = round(float(mis[va].mean()), 4)
    res["val_intervention_rate"] = round(float(itv[va].mean()), 4)
    # lead: mean surprise in the 3 steps before an intervention vs elsewhere (val)
    pre = np.zeros(len(meta), bool)
    idx = np.where(itv == 1)[0]
    for k in (1, 2, 3):
        pre[np.clip(idx - k, 0, len(meta) - 1)] = True
    res["surprise_pre_intervention_mean"] = round(float(s_flow[va & pre].mean()), 3)
    res["surprise_elsewhere_mean"] = round(float(s_flow[va & ~pre].mean()), 3)

    prev = json.loads(C.METRICS_JSON.read_text()) if C.METRICS_JSON.exists() else {}
    prev["evaluate"] = res
    C.METRICS_JSON.write_text(json.dumps(prev, indent=1))
    for k, v in res.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
