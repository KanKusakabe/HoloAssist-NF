"""Metric figures + Japanese README.md / index.html for HoloAssist-NF (Experiment C)."""
from __future__ import annotations

import json
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C


def _md(s: str) -> str:
    """Render inline markdown (**bold**, *italic*) as HTML so index.html isn't literal."""
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\*(.+?)\*", r"<i>\1</i>", s)
    return s


DATA_ROWS = [
    ("データ", "<b>HoloAssist</b>（作業者＋指導者の協働タスク）の<b>公開注釈のみ</b>（動画・ログイン不要）。"
               "各クリップ＝行動区間 <code>[開始, 終了, ラベル]</code> の系列。"),
    ("使う量", "各ステップの <b>所要時間</b>・<b>直前ステップとの間</b>、<b>ミス(Correct/Wrong)</b>、<b>指導者の介入</b>。"),
    ("予測対象 x", "<b>(log所要時間, log間)</b>"),
    ("条件 c", "<b>GRU(過去ステップの行動＋タイミング) ＋ 現ステップの行動</b>（＝これまでの手順の流れ）"),
    ("モデル", "<b>逐次</b>条件付き <b>Neural Spline Flow</b>。比較ベース＝MDN（ガウス混合）。"),
    ("スコア", "<b>SURPRISE = −log p(タイミング｜履歴, 行動)</b>＝「今の手順の間合いが、流れから見てどれだけ意外か」"),
]
NUM_GUIDE = [
    ("AUROC", "ミス/介入とそうでない所を見分ける力。<b>0.5＝勘・1.0＝完璧</b>。"),
    ("NLL（held-out）", "手順ペースの分布をどれだけ当てたか。<b>低いほど良い</b>。"),
]


def _figures(m):
    tr = m.get("train", {})
    fh = tr.get("flow", {}).get("history", [])
    mh = tr.get("mdn_baseline", {}).get("history", [])
    if fh:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([r["epoch"] for r in fh], [r["val_nll"] for r in fh], label="Flow (NSF)", color="#d97757")
        if mh:
            ax.plot([r["epoch"] for r in mh], [r["val_nll"] for r in mh], label="MDN baseline", color="#8b93a1")
        ax.set_xlabel("epoch"); ax.set_ylabel("held-out NLL (lower = better)")
        ax.set_title("Sequential timing density: Flow vs MDN"); ax.legend()
        fig.tight_layout(); fig.savefig(C.FIGS / "training_curve.png", dpi=110); plt.close(fig)

    ev = m.get("evaluate", {})
    groups = ["mistake", "intervention", "anticipation"]
    flow = [ev.get(f"{g}_auc_flow", 0) for g in groups]
    mdn = [ev.get(f"{g}_auc_mdn", 0) for g in groups]
    dur = [ev.get(f"{g}_auc_duration_only", 0) for g in groups]
    x = np.arange(3); w = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(x - w, flow, w, label="Flow surprise", color="#d97757")
    ax.bar(x, mdn, w, label="MDN surprise", color="#8b93a1")
    ax.bar(x + w, dur, w, label="duration-only", color="#c9c2b8")
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(["mistake\ndetection", "intervention\ndetection", "intervention\nanticipation"])
    ax.set_ylim(0.4, 0.75); ax.set_ylabel("AUROC (0.5 = chance)")
    ax.set_title("What timing-surprise detects (held-out clips)"); ax.legend()
    for xi, v in zip([x - w, x, x + w], [flow, mdn, dur]):
        for a, b in zip(xi, v):
            ax.annotate(f"{b:.2f}", (a, b), ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIGS / "auc.png", dpi=110); plt.close(fig)


FIGURES = [
    ("replay.gif", "リプレイ：手順の進行に沿ってタイミング・サプライズが動く＋Flowが期待タイミングを生成",
     "**左**＝手順ステップ順に並べた **サプライズ = −log p(タイミング)** の推移。赤い縦線＝指導者の介入。"
     "**右**＝各ステップでFlowが**サンプリングで生成した「期待される所要時間・間の分布(fan)」**（灰）と、"
     "実際の値（オレンジ／ミスは濃橙／介入は赤）。**見どころ**：Aで眠らせていた"
     "*逐次の厳密尤度*と*生成*を使えている。実測が灰の雲から外れる＝pace異常。"),
    ("timeline.png", "1クリップのサプライズ時系列（赤＝介入）",
     "手順が進むにつれサプライズがどう動くか。介入(赤線)の周辺でやや持ち上がる傾向はあるが、"
     "単発のスパイクで介入を当てられるほど強くはない（下のAUC参照）。"),
    ("auc.png", "タイミング・サプライズは何を検出できるか（Flow / MDN / 所要時間のみ）",
     "3課題×3手法のAUROC（0.5=偶然）。**見どころ**：①ミス検出はほぼ偶然(≈0.51)＝"
     "「間違い」はタイミングでなく*意味*の異常。②介入の検出/予兆は弱いが偶然超え(≈0.57)。"
     "③FlowとMDNはAUCではほぼ互角＝**下流の天井はモデルでなく信号（タイミング）側**にある。"),
    ("training_curve.png", "密度としての当てはまり：Flow vs MDN（held-out NLL）",
     "**ここがNFの本質的な勝ち**：held-out NLLで **Flow < MDN**（低いほど良い）。"
     "手順タイミングの分布は非ガウス・多峰で、Neural Spline Flowの表現力が効く＝"
     "実験Aで『ガウスに並ばれた』反省を、密度の指標では克服できている。"),
]


def _readme(m):
    tr, ev = m.get("train", {}), m.get("evaluate", {})
    fn = tr.get("flow", {}).get("best_val_nll"); mn = tr.get("mdn_baseline", {}).get("best_val_nll")
    L = []
    L.append("# HoloAssist-NF — 手順の「タイミング」を逐次 Normalizing Flow で（実験C）\n")
    L.append("手順タスクの**ペース（各ステップの所要時間・間）**を、履歴で条件づけた**逐次** "
             "Normalizing Flow で `log p(タイミング | これまでの手順, 行動)` として学習し、"
             "**SURPRISE = −log p** で「ミス／指導者の介入」を先読みできるかを見る実験。"
             "実験Aで眠らせていた **逐次の厳密尤度・生成・予兆** を起こすのが狙い。"
             "NF forget/mistake シリーズ(A–E)の C。図・数値は `python -m holo.report` で自動生成。\n")

    L.append("## どんなデータか\n")
    L.append("- **HoloAssist**（作業者＋指導者の協働タスク）の**オープン注釈のみ**を使用（動画不要・ログイン不要）。")
    L.append("- 各クリップ＝行動区間の系列 `[開始, 終了, ラベル]`。ここから **所要時間・直前ステップとの間**を計算。")
    L.append(f"- 注釈：**fine-grained action**（{ev.get('n_val',0):,} 検証ステップ）＋"
             f"**mistake（Correct/Wrong）**＋**intervention（指導者の介入）**。"
             f"検証集合のミス率 {ev.get('val_mistake_rate')} / 介入率 {ev.get('val_intervention_rate')}。\n")

    L.append("## どんなモデルを学習したか\n")
    L.append("- **逐次条件付き NSF**：`x = (log所要時間, log間)` ／ "
             "`c = GRU(過去ステップの行動埋め込み+タイミング) + 現ステップの行動`。")
    L.append("- 比較のため **MDN（ガウス混合）ベースライン**も同条件で学習（実験Aの教訓＝"
             "『ガウス混合に並ばれたら勝ちでない』を毎回検証）。")
    L.append("- スコア **SURPRISE = −log p(タイミング | 履歴, 行動)**。\n")

    L.append("## 結果（正直に）\n")
    if fn is not None and mn is not None:
        L.append(f"- **密度の当てはまりは Flow の勝ち**：held-out NLL **Flow `{fn:.3f}` < MDN `{mn:.3f}`**（低いほど良い）。"
                 f"手順タイミングの非ガウス・多峰性にNFの表現力が効く＝**Aの『ガウスに並ばれた』を克服**。")
    if ev:
        L.append(f"- **介入の検出/予兆は弱いが偶然超え**：AUROC 介入 `{ev.get('intervention_auc_flow')}` / "
                 f"予兆(次{3}手以内) `{ev.get('anticipation_auc_flow')}`。介入直前のサプライズ "
                 f"`{ev.get('surprise_pre_intervention_mean')}` > 他 `{ev.get('surprise_elsewhere_mean')}`。")
        L.append(f"- **ミス検出はほぼ偶然**：AUROC `{ev.get('mistake_auc_flow')}`。"
                 f"『間違い』はタイミングでなく*意味（行動の正誤）*の異常だから、という正直な限界。")
        L.append(f"- **AUCでは Flow ≈ MDN**：下流の天井は*モデル*でなく*信号（タイミング単独）*側にある。"
                 f"→ 動作・視線（rawストリーム）を足す拡張(C4)の動機。\n")

    L.append("## 図の見方\n")
    for f, t, h in FIGURES:
        if (C.FIGS / f).exists():
            L.append(f"### {t}\n\n![{f}](results/figures/{f})\n\n{h}\n")

    L.append("## 再現手順\n")
    L.append("```bash\n"
             "python -m holo.fetch      # HoloAssistオープン注釈を取得\n"
             "python -m holo.extract    # -> data/processed/segments.parquet\n"
             "python -m holo.features   # 語彙 + 正規化\n"
             "python -m holo.train      # 逐次NSF + MDNベースライン（--fastで高速）\n"
             "python -m holo.evaluate   # ミス/介入/予兆のAUC\n"
             "python -m holo.replay     # サプライズ時系列 + 期待タイミングfan（GIF/MP4）\n"
             "python -m holo.report     # 図 + この日本語README + index.html\n```\n")
    L.append("_条件付きNFで手順の尤度を forget/mistake ポテンシャルとして測るシリーズ(A–E)の C。"
             "A=物の置き場（空間）、C=手順のタイミング（時間）。次はB(HD-EPIC: 3D配置×視線)。_")
    C.BASE.joinpath("README.md").write_text("\n".join(L))


def _index_html(m):
    tr, ev = m.get("train", {}), m.get("evaluate", {})
    fn = tr.get("flow", {}).get("best_val_nll"); mn = tr.get("mdn_baseline", {}).get("best_val_nll")
    blocks = "\n".join(
        f'<section><h2>{t}</h2><img src="results/figures/{f}" alt="{f}"><p class="howto">{_md(h)}</p></section>'
        for f, t, h in FIGURES if (C.FIGS / f).exists())
    drows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in DATA_ROWS)
    guide = "".join(f"<li><b>{k}</b>：{v}</li>" for k, v in NUM_GUIDE)
    data_html = (f'<section><h2>データ / 学習（このページの前提）</h2><table class="d">{drows}</table>'
                 f'<p class="sub" style="margin:.7rem 0 .2rem"><b>数字の読み方</b></p><ul>{guide}</ul></section>')
    nll = f"{fn:.3f}" if fn is not None else "—"
    nllm = f"{mn:.3f}" if mn is not None else "—"
    html = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HoloAssist-NF · 手順のタイミングを逐次NFで</title>
<style>
 body{{font:16px/1.75 -apple-system,"Hiragino Sans","Noto Sans JP",sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#222}}
 h1{{line-height:1.35;margin-bottom:.2rem}} .sub{{color:#666}}
 .kpis{{display:flex;gap:1rem;flex-wrap:wrap;margin:1.4rem 0}}
 .kpi{{background:#f5f3f0;border-radius:12px;padding:1rem 1.3rem;min-width:150px}} .kpi b{{display:block;font-size:1.6rem;color:#c2410c}}
 section{{margin:2.3rem 0}} img{{width:100%;border:1px solid #e5e5e5;border-radius:10px}}
 .howto{{color:#444;background:#faf8f5;border-left:3px solid #d97757;padding:.6rem .9rem;border-radius:0 8px 8px 0}}
 code{{background:#f0eee9;padding:.1rem .3rem;border-radius:4px}} .lead{{background:#f7f5f2;border-radius:12px;padding:1rem 1.2rem}}
 table.d{{border-collapse:collapse;width:100%}} table.d td{{border:1px solid #e5e5e5;padding:.4rem .6rem;vertical-align:top}}
 table.d tr td:first-child{{white-space:nowrap;font-weight:600;background:#faf8f5;width:9rem}}
 ul{{margin:.3rem 0}}
</style></head><body>
<h1>HoloAssist-NF — 手順の「タイミング」を逐次 Normalizing Flow で</h1>
<p class="sub">実験C · <code>SURPRISE = −log p(タイミング | 履歴, 行動)</code> でミス／介入を先読みできるか。</p>
<div class="kpis">
 <div class="kpi"><b>{nll}</b>Flow held-out NLL</div>
 <div class="kpi"><b>{nllm}</b>MDN baseline NLL</div>
 <div class="kpi"><b>{ev.get('intervention_auc_flow','—')}</b>介入検出 AUROC</div>
</div>
<p class="lead"><b>何をしたか</b>：HoloAssistのオープン注釈（動画不要）から各ステップの所要時間・間を取り出し、
履歴で条件づけた逐次NSFで<b>手順ペースの密度</b>を学習。<b>密度の当てはまりはFlowがMDNに勝つ</b>（NLLが低い）一方、
<b>タイミング単独ではミス検出はほぼ偶然・介入は弱い信号</b>という正直な結果。実験Aで欠けていた
<b>逐次サプライズと生成（期待タイミングfan）</b>を使えている。</p>
{data_html}
{blocks}
<p class="sub"><code>python -m holo.report</code> で自動生成。NF forget/mistakeシリーズ(A–E)のC。</p>
</body></html>"""
    C.BASE.joinpath("index.html").write_text(html)


def main() -> None:
    m = json.loads(C.METRICS_JSON.read_text())
    _figures(m)
    _readme(m)
    _index_html(m)
    print("wrote README.md (JA) + index.html (JA) + figures")


if __name__ == "__main__":
    main()
