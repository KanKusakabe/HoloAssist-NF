"""HoloAssist annotations -> tidy per-step sequence parquet.

One row per fine-grained action step, ordered within a clip, with:
  log_dur   = log duration of the step
  log_gap   = log gap since the previous step ended (hesitation signal)
  a_code    = fine action code, n_code = noun code (consistent int codes -> embeddings)
  mistake   = 1 if the step overlaps a "Wrong Action" mistake segment
  intervention = 1 if it overlaps an instructor intervention segment
  split     = train/val/test (official clip split)
"""
from __future__ import annotations

import json
import math

import pandas as pd

from . import config as C


def _intervals(seg_list):
    """[start,end,[..,flag]] segments with flag==1 -> list of (start,end)."""
    out = []
    for s in seg_list or []:
        if s[2] and s[2][-1] == 1:
            out.append((s[0], s[1]))
    return out


def _hit(mid, intervals):
    return int(any(a <= mid <= b for a, b in intervals))


def main() -> None:
    labels = json.loads(C.LABELS_JSON.read_text())
    action = labels[C.K_ACTION]
    mistake = labels.get(C.K_MISTAKE, {})
    interv = labels.get(C.K_INTERV, {})

    split_of = {}
    for s in ("train", "val", "test"):
        for line in (C.RAW / f"{s}.txt").read_text().splitlines():
            if line.strip():
                split_of[line.strip()] = s

    rows = []
    for clip, segs in action.items():
        split = split_of.get(clip)
        if split is None:
            continue
        segs = sorted(segs, key=lambda x: x[0])
        m_iv = _intervals(mistake.get(clip))
        i_iv = _intervals(interv.get(clip))
        prev_end = None
        for order, s in enumerate(segs):
            start, end, vec = s[0], s[1], s[2]
            dur = max(end - start, C.MIN_DT)
            gap = 0.0 if prev_end is None else max(start - prev_end, 0.0)
            mid = 0.5 * (start + end)
            rows.append(dict(
                clip=clip, split=split, order=order, start=start, end=end,
                log_dur=math.log(dur), log_gap=math.log(gap + C.MIN_DT),
                a_code=int(vec[1]), n_code=int(vec[2]),
                mistake=_hit(mid, m_iv), intervention=_hit(mid, i_iv)))
            prev_end = end

    df = pd.DataFrame(rows)
    df.to_parquet(C.SEQ_PARQUET)
    n = len(df)
    print(f"rows={n}  clips={df['clip'].nunique()}  "
          f"splits={df['split'].value_counts().to_dict()}")
    print(f"mistake={int(df['mistake'].sum())} ({100*df['mistake'].mean():.1f}%)  "
          f"intervention={int(df['intervention'].sum())} ({100*df['intervention'].mean():.1f}%)")
    print("wrote", C.SEQ_PARQUET)


if __name__ == "__main__":
    main()
