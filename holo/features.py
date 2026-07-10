"""Vocab sizes for the action/noun codes + timing normalisation (train split only)."""
from __future__ import annotations

import json

import pandas as pd

from . import config as C


def main() -> None:
    df = pd.read_parquet(C.SEQ_PARQUET)
    tr = df[df["split"] == "train"]
    # index 0 is reserved as a padding slot; real codes are shifted +1 at load time.
    vocab = {
        "n_a": int(df["a_code"].max()) + 2,
        "n_n": int(df["n_code"].max()) + 2,
    }
    C.VOCAB_JSON.write_text(json.dumps(vocab, indent=1))
    norm = {k: [float(tr[k].mean()), float(tr[k].std() + 1e-6)] for k in C.TARGET}
    C.NORM_JSON.write_text(json.dumps(norm, indent=1))
    print("vocab", vocab)
    print("norm", norm)


if __name__ == "__main__":
    main()
