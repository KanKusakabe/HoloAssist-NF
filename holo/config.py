"""Paths, constants, and the HoloAssist open-annotation URLs."""
from __future__ import annotations

from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
RESULTS = BASE / "results"
FIGS = RESULTS / "figures"
for _d in (RAW, PROC, RESULTS, FIGS):
    _d.mkdir(parents=True, exist_ok=True)

# HoloAssist annotations are fully open on GitHub (no login, no video needed).
GH = ("https://raw.githubusercontent.com/Ember-HoloAssist/holoassist-release/"
      "main/src/data_2221/")
LABELS_URL = GH + "labels_20240719_2221_classes.json"
LABEL2IDX_URL = GH + "labels_20240719_2221_label2idx.json"
SPLIT_URLS = {s: GH + f"{s}_0724.txt" for s in ("train", "val", "test")}

# annotation keys we use
K_ACTION = "fine_grained_action"        # [start, end, [_, action_code, noun_code, _, _]]
K_MISTAKE = "mistake_prediction"        # [start, end, [_, wrong(0/1)]]
K_INTERV = "intervention_detection_1"   # [start, end, [_, flag(0/1)]]

TARGET = ["log_dur", "log_gap"]         # 2-D density the flow models
HISTORY = 8                             # GRU history length (steps)
MIN_DT = 1e-2                           # seconds floor before log

LABELS_JSON = RAW / "labels.json"
LABEL2IDX_JSON = RAW / "label2idx.json"
SEQ_PARQUET = PROC / "segments.parquet"
VOCAB_JSON = PROC / "vocab.json"
NORM_JSON = PROC / "norm_stats.json"
MODEL_PT = RESULTS / "model.pt"
METRICS_JSON = RESULTS / "metrics.json"
