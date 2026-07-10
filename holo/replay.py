"""Replay one held-out clip: watch the timing SURPRISE build up over the task,
while the Flow *generates* the expected next-step timing (a fan of samples) and
we overlay what actually happened. Interventions are marked in red.

This uses the two NF strengths Experiment A left out: a sequential exact-
likelihood surprise curve and generation. Output: GIF (+ mp4 if ffmpeg).
"""
from __future__ import annotations

import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from . import config as C
from .data import load_windows
from .model import StepModel


def _pick_clip(meta):
    va = meta[meta["split"] == "val"]
    best, bn = None, -1
    for clip, g in va.groupby("clip"):
        if 20 <= len(g) <= 90 and g["intervention"].sum() >= 2:
            if g["intervention"].sum() > bn:
                best, bn = clip, g["intervention"].sum()
    if best is None:  # relax
        for clip, g in va.groupby("clip"):
            if 15 <= len(g) <= 120 and g["intervention"].sum() > bn:
                best, bn = clip, g["intervention"].sum()
    return best


@torch.no_grad()
def main():
    tens, meta, vocab = load_windows()
    norm = json.loads(C.NORM_JSON.read_text())
    ck = torch.load(C.MODEL_PT, map_location="cpu")
    flow = StepModel(ck["n_a"], ck["n_n"], head="flow"); flow.load_state_dict(ck["state"]); flow.eval()

    clip = _pick_clip(meta)
    idx = np.where((meta["clip"].values == clip))[0]
    idx = idx[np.argsort(meta["order"].values[idx])]
    b = {k: v[idx] for k, v in tens.items()}
    cond = flow.condition(b)
    surprise = (-flow.head.log_prob(cond, b["y"])).numpy()
    itv = meta["intervention"].values[idx].astype(bool)
    mis = meta["mistake"].values[idx].astype(bool)
    steps = np.arange(len(idx))

    md, mg = norm["log_dur"]; gd, gg = norm["log_gap"]

    def to_sec(z, m, s):
        return np.exp(m + s * z)

    samp = flow.head.sample(cond, n=150).numpy()          # [T,150,2]
    actual = b["y"].numpy()                                 # [T,2]

    fig, (axT, axR) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 2]})
    axT.set_xlim(0, len(idx) - 1); axT.set_ylim(surprise.min() - 0.3, surprise.max() + 0.5)
    axT.set_xlabel("procedure step"); axT.set_ylabel("SURPRISE = -log p (timing)")
    axT.set_title(f"{clip}: timing surprise over the task")
    line, = axT.plot([], [], "-", color="#d97757", lw=1.8)
    cur = axT.scatter([], [], s=60, color="#d97757", zorder=5)
    for s in steps[itv]:
        axT.axvline(s, color="red", alpha=0.25, lw=2)
    axT.plot([], [], color="red", alpha=0.4, lw=2, label="intervention")
    axT.legend(loc="upper left", fontsize=8)

    axR.set_xlabel("duration [s]"); axR.set_ylabel("gap before step [s]")
    axR.set_title("Flow's expected timing (fan) vs actual")
    fan = axR.scatter([], [], s=8, alpha=0.25, color="#8b93a1", label="expected (Flow samples)")
    act = axR.scatter([], [], s=140, color="#d97757", edgecolor="k", zorder=5, label="actual")
    axR.legend(loc="upper right", fontsize=8)

    def update(t):
        line.set_data(steps[:t + 1], surprise[:t + 1])
        cur.set_offsets([[steps[t], surprise[t]]])
        d = to_sec(samp[t, :, 0], md, mg); g = to_sec(samp[t, :, 1], gd, gg)
        fan.set_offsets(np.column_stack([d, g]))
        ad = to_sec(actual[t, 0], md, mg); ag = to_sec(actual[t, 1], gd, gg)
        act.set_offsets([[ad, ag]])
        act.set_color("red" if itv[t] else ("#c2410c" if mis[t] else "#d97757"))
        axR.set_xlim(0, np.percentile(np.concatenate([d, [ad]]), 99) * 1.1 + 0.1)
        axR.set_ylim(0, np.percentile(np.concatenate([g, [ag]]), 99) * 1.1 + 0.1)
        lab = "  ← intervention" if itv[t] else ("  ← mistake" if mis[t] else "")
        axR.set_title(f"step {t}: expected timing (fan) vs actual{lab}")
        return line, cur, fan, act

    anim = FuncAnimation(fig, update, frames=len(idx), interval=180, blit=False)
    fig.tight_layout()
    anim.save(C.FIGS / "replay.gif", writer=PillowWriter(fps=6), dpi=90)
    print("wrote replay.gif", f"({clip}, {len(idx)} steps, {int(itv.sum())} interventions)")
    try:
        anim.save(C.RESULTS / "replay.mp4", writer="ffmpeg", fps=6, dpi=110)
        print("wrote replay.mp4")
    except Exception as e:
        print("mp4 skipped:", e)
    plt.close(fig)

    # static timeline for the README
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(steps, surprise, "-o", ms=3, color="#d97757")
    for s in steps[itv]:
        ax.axvline(s, color="red", alpha=0.3, lw=2)
    ax.set_xlabel("procedure step"); ax.set_ylabel("SURPRISE")
    ax.set_title(f"{clip}: timing surprise (red = intervention)")
    fig.tight_layout(); fig.savefig(C.FIGS / "timeline.png", dpi=110); plt.close(fig)
    print("wrote timeline.png")


if __name__ == "__main__":
    main()
