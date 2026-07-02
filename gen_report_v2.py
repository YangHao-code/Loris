#!/usr/bin/env python3
"""LORIS paper report generator v2 — 7 datasets + the full Exp 1-5 matrix.

Reads the experiment outputs written by run_main_loris / run_phase3(_loris) /
run_ksweep_loris / run_msweep_loris / run_humancost / run_scalability /
run_ablations / run_influence_mrr, plus baseline_results.csv, and emits
paper/results_auto.tex (+ pdflatex). Idempotent and robust to missing data:
absent experiments render a "\\emph{(pending)}" placeholder so the report can be
regenerated at any point as runs complete.

Baseline substitutions are DISCLOSED (never mislabeled): RAL→CoMAL, standalone
LocalBoost→RuleCleaner (KDD'25; LocalBoost KDD'23 released no code), RulePrompt→
documented reimpl (original cannot run on the sm_120 GPU). The Group-C in-pipeline
'localboost' pattern ranker is a distinct LocalBoost reimpl.
"""
import csv, glob, json, os, subprocess

ROOT = "/root/autodl-tmp/Loris"
os.chdir(ROOT)

# 7 datasets (rcv1 kept where applicable; it is hashed-TFIDF so text/LLM baselines
# and router MRR are N/A there — disclosed in footnotes).
COLS = ["reuters21578", "aapd", "rcv1", "bgc", "arxiv", "pubmed", "hupd"]
DSP = {"reuters21578": "reuters", "aapd": "aapd", "rcv1": "rcv1", "bgc": "bgc",
       "arxiv": "arxiv", "pubmed": "pubmed", "hupd": "hupd"}


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def first(g):
    fs = sorted(glob.glob(g))
    return load(fs[0]) if fs else None


def fmt(x, d=3):
    return f"{x:.{d}f}" if isinstance(x, (int, float)) else "--"


def esc(s):
    return str(s).replace("_", "\\_")


# ══════════════════════════════════════════════════════════════════════════════
# Exp-1: main table (per-model reference rows + LORIS base / pool+LoRA)
# ══════════════════════════════════════════════════════════════════════════════
MODELS = ["tfidf_svm_unigram", "tfidf_svm_bigram", "tfidf_lr_bigram",
          "textcnn", "bilstm", "encoder_mlp", "lora_slm", "lora_mistral", "lora_llama3"]


def _main_cell(ds):
    """Prefer the new main_7ds/ output; fall back to legacy main_0619_enc/."""
    j = load(f"experiments/main_7ds/main_base__{ds}.json")
    lora = load(f"experiments/main_7ds/main_lora__{ds}.json")
    if j:
        return {"pm": j.get("per_model_test") or {},
                "loris": (j.get("final_micro_f1"), j.get("final_macro_f1")),
                "rules": j.get("n_rules"),
                "lora": ((lora or {}).get("final_micro_f1"), (lora or {}).get("final_macro_f1")) if lora else None}
    legacy = {"reuters21578": "enc_reuters", "aapd": "enc_aapd", "rcv1": "enc_rcv1", "bgc": "enc_bgc"}
    d = legacy.get(ds)
    m = first(f"experiments/main_0619_enc/{d}/*/metrics.json") if d else None
    if not m:
        return None
    return {"pm": m.get("per_model_test") or {},
            "loris": (m.get("final_micro_f1"), m.get("final_macro_f1")),
            "rules": m.get("n_rules"), "lora": None}


def main_table():
    data = {c: _main_cell(c) for c in COLS}
    have = [c for c in COLS if data[c]]
    if not have:
        return "\\emph{(main table pending — run run\\_main\\_loris.py)}\n"
    ncol = len(have)
    header = " & ".join("\\multicolumn{2}{c}{\\textbf{%s}}" % DSP[c] for c in have)
    cmid = "".join("\\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i) for i in range(ncol))
    mima = " & ".join(["mi & ma"] * ncol)
    R = []
    present = [m for m in MODELS if any(m in (data[c]["pm"] or {}) for c in have)]
    for mo in present:
        cells = []
        for c in have:
            v = (data[c]["pm"] or {}).get(mo)
            cells += [fmt(v["micro_f1"]), fmt(v["macro_f1"])] if v else ["--", "--"]
        R.append("\\quad " + esc(mo) + " & " + " & ".join(cells) + " \\\\")
    R.append("\\midrule")
    lcells = []
    for c in have:
        mi, ma = data[c]["loris"]
        lcells += ["\\bld{%s}" % fmt(mi), "\\bld{%s}" % fmt(ma)]
    R.append("\\rowcolor{blue!8}\\bld{\\quad LORIS (base pool)} & " + " & ".join(lcells) + " \\\\")
    if any(data[c].get("lora") for c in have):
        lcells = []
        for c in have:
            lv = data[c].get("lora")
            lcells += (["\\bld{%s}" % fmt(lv[0]), "\\bld{%s}" % fmt(lv[1])] if lv and lv[0] else ["--", "--"])
        R.append("\\rowcolor{blue!8}\\bld{\\quad LORIS (pool + LoRA)} & " + " & ".join(lcells) + " \\\\")
    R.append("\\quad \\#rules & " + " & ".join("\\multicolumn{2}{c}{%s}" % (data[c]["rules"] or "--") for c in have) + " \\\\")
    colspec = "l" + "cc" * ncol
    return (r"""\begin{table}[h]\centering\footnotesize\setlength{\tabcolsep}{4pt}
\caption{\textbf{Main evaluation (full supervision, from scratch).} Per-model rows are
\emph{reference} accuracies (each model alone), never LORIS's input; LORIS labels from
scratch (\texttt{--track1\_baseline blank}), a model enters only as a rule predicate
$M(x,\tau)$. ``pool + LoRA'' adds Mistral-7B + Llama-3-8B (QLoRA). rcv1 ships as hashed
TF-IDF (no raw text) so subword/LLM models are N/A there.}
\label{tab:big}
\begin{tabular}{""" + colspec + r"""}
\toprule
 & """ + header + r""" \\
""" + cmid + "\n & " + mima + r""" \\
\midrule
""" + "\n".join(R) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Exp-1: research-method baselines vs LORIS (from baseline_results.csv)
# ══════════════════════════════════════════════════════════════════════════════
SUBS = {  # disclosed substitutions
    "comal": "CoMAL$^{a}$", "localboost": "RuleCleaner$^{b}$",
    "ruleprompt": "RulePrompt$^{c}$",
}


def baselines_table():
    p = "baseline_results.csv"
    if not os.path.exists(p):
        return "\\emph{(baselines table pending)}\n"
    rows = list(csv.DictReader(open(p)))
    # keep macro_f1 rows only
    macro = [r for r in rows if r.get("metric") == "macro_f1"]
    if not macro:
        return "\\emph{(baselines table pending)}\n"
    cols = [c for c in COLS if any(c in r and r[c] not in ("", None) for r in macro)]
    out = []
    for r in macro:
        name = r["baseline"]
        disp = SUBS.get(name, esc(name))
        cells = [fmt(float(r[c])) if r.get(c) not in ("", None) else "--" for c in cols]
        out.append(f"\\quad {disp} & " + " & ".join(cells) + " \\\\")
    colspec = "l" + "c" * len(cols)
    header = " & ".join("\\textbf{%s}" % DSP[c] for c in cols)
    return (r"""\begin{table}[h]\centering\footnotesize\setlength{\tabcolsep}{4pt}
\caption{\textbf{Research-method baselines --- macro-F1} (standalone, per method).
$^{a}$RAL released no code $\Rightarrow$ substituted by \textbf{CoMAL} (KDD'24), reported
under its own name. $^{b}$LocalBoost (KDD'23) released no code $\Rightarrow$ standalone slot
uses \textbf{RuleCleaner} (KDD'25). $^{c}$RulePrompt is a documented first-pass reimpl
(the original pins torch1.12/transformers4.20, which lack sm\_120 kernels for the GPU).
gpt4/ruleprompt run in mock mode without an API key.}
\label{tab:baselines}
\begin{tabular}{""" + colspec + r"""}
\toprule
Method & """ + header + r""" \\
\midrule
""" + "\n".join(out) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Exp-4: model selection (Group A gate: router vs baseline selectors)
# ══════════════════════════════════════════════════════════════════════════════
def _gate_macro(ds, name):
    for d in ("experiments/baselines_loris_rework", "experiments/baselines_loris"):
        j = load(f"{d}/{name}__{ds}__seed0.json")
        if j and j.get("macro_f1") is not None:
            return j["macro_f1"]
    return None


def modelselect_table():
    sels = [("A_router", "router (LORIS)"), ("A_random_ms", "random\\_ms"),
            ("A_indiv_ms", "indiv\\_ms"), ("A_hybrid_llm", "hybrid\\_llm"),
            ("A_caas", "caas")]
    cols = [c for c in COLS if any(_gate_macro(c, s) is not None for s, _ in sels)]
    if not cols:
        return "\\emph{(model-selection gate pending — run run\\_phase3\\_loris.py)}\n"
    R = []
    for key, disp in sels:
        cells = [fmt(_gate_macro(c, key)) for c in cols]
        pref = "\\rowcolor{blue!8}" if key == "A_router" else ""
        R.append(pref + "\\quad " + disp + " & " + " & ".join(cells) + " \\\\")
    colspec = "l" + "c" * len(cols)
    header = " & ".join("\\textbf{%s}" % DSP[c] for c in cols)
    return (r"""\begin{table}[h]\centering\footnotesize\setlength{\tabcolsep}{4pt}
\caption{\textbf{Model selection (Exp-4): LORIS final macro-F1 under a component swap.}
Each cell runs the FULL LORIS pipeline with the model-selection component replaced by the
named baseline at the same K; the router row is per-cluster + oracle-tuned. Varying-K and
Varying-$|M|$ sweeps in Tables~\ref{tab:ksweep},~\ref{tab:msweep}.}
\label{tab:select}
\begin{tabular}{""" + colspec + r"""}
\toprule
Selector & """ + header + r""" \\
\midrule
""" + "\n".join(R) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def _sweep_table(subdir, keyfmt, axis_name, label, caption):
    files = glob.glob(f"experiments/baselines_loris/{subdir}/*.json")
    if not files:
        return "\\emph{(%s pending)}\n" % axis_name
    rows = {}
    for f in files:
        j = load(f)
        if not j or j.get("macro_f1") is None:
            continue
        sel = j.get("selector") or j.get("baseline")
        ax = j.get(keyfmt)
        rows.setdefault(sel, {})[ax] = j["macro_f1"]
    if not rows:
        return "\\emph{(%s pending)}\n" % axis_name
    axes = sorted({a for d in rows.values() for a in d})
    R = []
    for sel in sorted(rows):
        cells = [fmt(rows[sel].get(a)) for a in axes]
        R.append("\\quad " + esc(sel) + " & " + " & ".join(cells) + " \\\\")
    colspec = "l" + "c" * len(axes)
    header = " & ".join("%s=%s" % (axis_name, a) for a in axes)
    return (r"""\begin{table}[h]\centering\footnotesize
\caption{""" + caption + r"""}
\label{""" + label + r"""}
\begin{tabular}{""" + colspec + r"""}
\toprule
Selector & """ + header + r""" \\
\midrule
""" + "\n".join(R) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def ksweep_table():
    return _sweep_table("ksweep", "k_models", "K", "tab:ksweep",
                        "\\textbf{Varying-K (Exp-4).} LORIS macro-F1 vs \\# selected models K, per selector.")


def msweep_table():
    return _sweep_table("msweep", "M", "M", "tab:msweep",
                        "\\textbf{Varying-$|M|$ (Exp-4).} LORIS macro-F1 vs candidate-pool size $|M|$, per selector.")


# ══════════════════════════════════════════════════════════════════════════════
# Exp-2: human cost (Γ training sweep + RILL LLM-vs-noLLM)
# ══════════════════════════════════════════════════════════════════════════════
def humancost_table():
    files = sorted(glob.glob("experiments/humancost/gamma_*.json"))
    if not files:
        return "\\emph{(human-cost pending — run run\\_humancost.py)}\n"
    R = []
    gammas = None
    for f in files:
        j = load(f)
        if not j:
            continue
        ds = j["dataset"]
        rows = {r["gamma"]: r for r in j.get("rows", [])}
        if gammas is None:
            gammas = sorted(rows)
        cells = [fmt(rows[g]["macro_f1"]) if g in rows else "--" for g in gammas]
        R.append("\\quad " + DSP.get(ds, ds) + " & " + " & ".join(cells) + " \\\\")
    if not R or not gammas:
        return "\\emph{(human-cost pending)}\n"
    colspec = "l" + "c" * len(gammas)
    header = " & ".join("$\\Gamma{=}%d$" % g for g in gammas)
    return (r"""\begin{table}[h]\centering\footnotesize
\caption{\textbf{Human cost (Exp-2): macro-F1 vs training human budget $\Gamma$.} LORIS
learns rules from $\Gamma$ coverage-seeded human-labeled docs (from scratch) and labels the
rest. The LLM-pre-annotator vs human-only saving (fewer human queries at equal F1) and the
LORIS-vs-BESRA/CoMAL comparison are in the RILL sweep JSONs (experiments/humancost/rill\_*).}
\label{tab:human}
\begin{tabular}{""" + colspec + r"""}
\toprule
Dataset & """ + header + r""" \\
\midrule
""" + "\n".join(R) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Exp-3: scalability (runtime) + incremental vs noInc
# ══════════════════════════════════════════════════════════════════════════════
def scalability_table():
    files = sorted(glob.glob("experiments/scalability/*.json"))
    files = [f for f in files if "/_runs/" not in f]
    if not files:
        return "\\emph{(scalability pending — run run\\_scalability.py)}\n"
    R = []
    for f in files:
        j = load(f)
        if not j:
            continue
        ds = j["dataset"]
        inc = {r["mode"]: r for r in j.get("incremental", [])}
        i_s = inc.get("incremental", {}).get("chase_wall_sec")
        n_s = inc.get("noInc", {}).get("chase_wall_sec")
        speed = (fmt(n_s / i_s, 2) + "$\\times$") if (i_s and n_s) else "--"
        R.append("\\quad %s & %s & %s & %s \\\\" % (DSP.get(ds, ds), fmt(i_s, 2), fmt(n_s, 2), speed))
    if not R:
        return "\\emph{(scalability pending)}\n"
    return (r"""\begin{table}[h]\centering\footnotesize
\caption{\textbf{Scalability (Exp-3): incremental vs non-incremental chase wall-clock (s).}
Identical rules and labels; the incremental chase re-evaluates only affected docs, the noInc
ablation re-scans all rules over all docs each round. Full \\mbox{$|D|$/$|\Sigma|$} runtime
sweeps in experiments/scalability/<ds>.json.}
\label{tab:scale}
\begin{tabular}{lccc}
\toprule
Dataset & incremental & noInc & speedup \\
\midrule
""" + "\n".join(R) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Exp-5: architecture + staged ablations
# ══════════════════════════════════════════════════════════════════════════════
def ablation_table():
    files = sorted(glob.glob("experiments/ablations/*.json"))
    files = [f for f in files if "/_runs/" not in f]
    if not files:
        return "\\emph{(ablations pending — run run\\_ablations.py)}\n"
    data = {}
    arms_order = []
    for f in files:
        j = load(f)
        if not j:
            continue
        ds = j["dataset"]
        data[ds] = {a["arm"]: a.get("macro_f1") for a in j.get("arms", [])}
        for a in j.get("arms", []):
            if a["arm"] not in arms_order:
                arms_order.append(a["arm"])
    cols = [c for c in COLS if c in data]
    R = []
    for arm in arms_order:
        cells = [fmt(data[c].get(arm)) for c in cols]
        R.append("\\quad " + esc(arm) + " & " + " & ".join(cells) + " \\\\")
    colspec = "l" + "c" * len(cols)
    header = " & ".join("\\textbf{%s}" % DSP[c] for c in cols)
    return (r"""\begin{table}[h]\centering\footnotesize
\caption{\textbf{Ablations (Exp-5): LORIS macro-F1 per variant.} Architecture: noS
(Gumbel-Softmax vs stochastic smoothing), noL (task-loss-only router), noInc
(non-incremental chase; F1 identical to full, see Table~\ref{tab:scale} for its runtime
cost). Staged: leave-one-out of stages 1/2/3; pool=embedding-only; gt-leak oracle bound.
Pattern-selection alternatives (filter\_mi/chi2/weshap/localboost) are the Group-C swap in
Table~\ref{tab:select}'s companion gate.}
\label{tab:abl}
\begin{tabular}{""" + colspec + r"""}
\toprule
Variant & """ + header + r""" \\
\midrule
""" + "\n".join(R) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Exp-1c: influence estimation (MRR + RDG time saving)
# ══════════════════════════════════════════════════════════════════════════════
def influence_table():
    q = load("experiments/influence/quality.json")
    t = load("experiments/influence/rdg_timing.json")
    if not q:
        return "\\emph{(influence pending — run run\\_influence\\_mrr.py)}\n"
    R = [f"\\quad {DSP.get(r['dataset'], r['dataset'])} & {fmt(r['hit_at_k'])} & {fmt(r['mrr'])} \\\\"
         for r in q.get("rows", [])]
    tnote = ""
    if t and t.get("rows"):
        best = t["rows"][-1]
        tnote = (" RDG bounded-BFS influence vs exact transitive closure saves "
                 "\\textbf{%.0f\\%%} time at %d labels (%.1f$\\times$)." %
                 (best["time_saved_pct"], best["n_labels"], best["speedup"]))
    return (r"""\begin{table}[h]\centering\footnotesize
\caption{\textbf{Influence estimation (Exp-1c): RDG quality + time saving.} Oracle hit@k and
MRR of the RDG/router influence ranking (held-out; from router tuning). rcv1 omitted
(hashed-TFIDF).""" + tnote + r"""}
\label{tab:mrr}
\begin{tabular}{lcc}
\toprule
Dataset & hit@k & MRR \\
\midrule
""" + "\n".join(R) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


# ══════════════════════════════════════════════════════════════════════════════
PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage{booktabs}\usepackage{multirow}\usepackage{amsmath}\usepackage{amssymb}
\usepackage{xcolor}\usepackage{colortbl}\usepackage{underscore}\usepackage{pifont}
\newcommand{\bld}[1]{\textbf{#1}}\newcommand{\gn}[1]{\textcolor{blue}{#1}}
\title{LORIS --- Experimental Results (v2, 7 datasets)}\date{}
\begin{document}\maketitle
\section{Setup}
Seven multi-label corpora: reuters-21578, aapd, rcv1, bgc, arxiv, PubMed-MeSH, HUPD.
All LORIS runs label \emph{from scratch} (\texttt{--track1\_baseline blank},
\texttt{--pattern\_mode full}, per-cluster oracle-tuned router). Baseline substitutions
(RAL$\to$CoMAL, LocalBoost$\to$RuleCleaner, RulePrompt reimpl) are disclosed in
Table~\ref{tab:baselines}. Seed 0.
"""


def build():
    tex = (PREAMBLE
           + "\\section{Main Evaluation (Exp-1)}\n" + main_table() + baselines_table()
           + "\\section{Model Selection (Exp-4)}\n" + modelselect_table() + ksweep_table() + msweep_table()
           + "\\section{Human Cost (Exp-2)}\n" + humancost_table()
           + "\\section{Scalability (Exp-3)}\n" + scalability_table()
           + "\\section{Ablations (Exp-5)}\n" + ablation_table()
           + "\\section{Influence Estimation (Exp-1c)}\n" + influence_table()
           + "\\end{document}\n")
    # Write to results_auto_v2.tex until the 7-dataset runs complete (keeps the
    # legacy 4-dataset results_auto.tex intact). Swap the name to results_auto.tex
    # once the full matrix is in.
    out_name = os.environ.get("LORIS_REPORT_TEX", "results_auto_v2.tex")
    with open(f"paper/{out_name}", "w") as f:
        f.write(tex)
    ok = False
    try:
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", out_name],
                           cwd="paper", capture_output=True, timeout=180)
        ok = os.path.exists(f"paper/{out_name.replace('.tex', '.pdf')}")
    except Exception:
        ok = False
    print(f"[gen_report_v2] wrote paper/{out_name}  pdf={ok}")


if __name__ == "__main__":
    build()
