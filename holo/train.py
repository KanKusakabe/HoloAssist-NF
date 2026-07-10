"""Train the sequential timing density; held-out = official val clips.

Trains both the Flow and the MDN baseline so the report can honestly compare
them (the Experiment-A lesson: a Flow that only ties a Gaussian mixture has not
earned its keep)."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from . import config as C
from .data import load_windows, to_device
from .model import StepModel


def device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _train_one(head, tens, meta, vocab, dev, epochs, batch, lr):
    tr_idx = np.where(meta["split"].values == "train")[0]
    va = to_device(tens, dev, np.where(meta["split"].values == "val")[0])
    model = StepModel(vocab["n_a"], vocab["n_n"], head=head).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr_t = to_device(tens, dev)
    n = len(tr_idx); tr_idx = torch.tensor(tr_idx, device=dev)
    hist, best, best_state = [], float("inf"), None
    for ep in range(epochs):
        model.train(); perm = tr_idx[torch.randperm(n, device=dev)]
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            b = {k: v[idx] for k, v in tr_t.items()}
            loss = model.nll(b)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx)
        model.eval()
        with torch.no_grad():
            vnll = float(model.nll(va))
        hist.append({"epoch": ep, "train_nll": tot / n, "val_nll": vnll})
        if vnll < best:
            best = vnll
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 4 == 0 or ep == epochs - 1:
            print(f"  [{head}] ep {ep:3d}  train {tot/n:7.3f}  val {vnll:7.3f}")
    model.load_state_dict(best_state)
    return model, {"history": hist, "best_val_nll": best}


def main(epochs=40, batch=4096, lr=1e-3, fast=False):
    dev = device(); print("device:", dev)
    if fast:
        epochs = 12
    tens, meta, vocab = load_windows()
    print(f"windows: {len(meta):,}  (train {int((meta['split']=='train').sum()):,} / "
          f"val {int((meta['split']=='val').sum()):,})")

    flow, m_flow = _train_one("flow", tens, meta, vocab, dev, epochs, batch, lr)
    mdn, m_mdn = _train_one("mdn", tens, meta, vocab, dev, epochs, batch, lr)

    torch.save({"state": flow.state_dict(), "mdn_state": mdn.state_dict(),
                "n_a": vocab["n_a"], "n_n": vocab["n_n"]}, C.MODEL_PT)
    prev = json.loads(C.METRICS_JSON.read_text()) if C.METRICS_JSON.exists() else {}
    prev["train"] = {"flow": m_flow, "mdn_baseline": m_mdn,
                     "n_train": int((meta["split"] == "train").sum()),
                     "n_val": int((meta["split"] == "val").sum())}
    C.METRICS_JSON.write_text(json.dumps(prev, indent=1))
    print(f"Flow held-out NLL {m_flow['best_val_nll']:.3f} vs MDN {m_mdn['best_val_nll']:.3f} "
          f"-> saved {C.MODEL_PT.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    main(epochs=a.epochs, fast=a.fast)
