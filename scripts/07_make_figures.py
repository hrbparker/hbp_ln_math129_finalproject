"""Phase 4: build the report's figures and LaTeX tables from the CSVs in reports/.

    pip3 install matplotlib      (once)
    python3 scripts/07_make_figures.py

Reads (all written by earlier scripts; missing files are skipped with a warning):
    reports/confound_cv.csv, confound_paired.csv, confound_batch_predictability.csv,
    reports/confound_temporal.csv, within_month.csv, within_month_paired.csv

Writes (aggregate numbers only, safe to commit):
    reports/figures/fig1_paper_critique.{png,pdf}   the paper's confusion matrix vs a majority baseline
    reports/figures/fig2_baselines.{png,pdf}        age / metadata / speech AUC, 5x5 CV, with 95% CIs
    reports/figures/fig3_matched_pairs.{png,pdf}    AUC when pairs are restricted to same month / similar age
    reports/figures/fig4_temporal.{png,pdf}         train-early / test-late vs cross-validation
    reports/tables/*.tex                             booktabs tables (\\input them in the report)

Figure 1 uses the confusion matrix read off Figure 1(b) of Ugwu & Oyeleke (Slovak,
full model): TN=735, FP=128, FN=18, TP=26. Check it against the paper before citing.
Every other number comes from the CSVs, so the report matches your runs.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REP = Path("reports")
FIG, TAB = REP / "figures", REP / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e3e2dd", "#ffffff"
COL = {  # fixed colour per feature set, used in every figure
    "age only": "#8a8985",
    "all metadata": "#52514e",
    "speech": "#2a78d6",
    "speech + all metadata": "#eb6834",
    "speech (age/sex removed)": "#1baf7a",
    "speech (age/sex/month removed)": "#c98500",
}
MARK = {"age only": "o", "all metadata": "s", "speech": "D", "speech + all metadata": "^",
        "speech (age/sex removed)": "v", "speech (age/sex/month removed)": "P"}
TASKS = ["AD/MCI vs HC", "PD vs HC"]
TASK_NAME = {"AD/MCI vs HC": "AD/MCI vs. healthy", "PD vs HC": "Parkinson's vs. healthy"}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "legend.frameon": False,
    "font.family": "DejaVu Sans",
})


def load(name):
    p = REP / name
    if not p.exists():
        print(f"WARNING: {p} not found, skipping what needs it")
        return None
    return pd.read_csv(p)


def save(fig, stem):
    fig.savefig(FIG / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / f"{stem}.png")


def fmt(x, nd=3):
    return "--" if pd.isna(x) else f"{x:.{nd}f}"


def ci(lo, hi):
    return f"[{fmt(lo, 2)}, {fmt(hi, 2)}]"


def tex_escape(s):
    return str(s).replace("%", r"\%").replace("&", r"\&").replace("_", r"\_").replace("R^2", "$R^2$")


def write_tex(name, header, rows, caption, label, align=None):
    align = align or "l" + "r" * (len(header) - 1)
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\resizebox{\linewidth}{!}{%", rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}}",
              rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print("wrote", TAB / f"{name}.tex")


# ===================================================================== Fig 1: the paper's headline number
def fig1():
    tn, fp, fn, tp = 735, 128, 18, 26
    n = tn + fp + fn + tp
    acc = (tn + tp) / n
    maj = (tn + fp) / n
    sens, spec, prec = tp / (tp + fn), tn / (tn + fp), tp / (tp + fp)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
    cm = np.array([[tn, fp], [fn, tp]])
    a.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.25)
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=15, fontweight="bold",
                   color="white" if cm[i, j] > 400 else INK)
    a.set_xticks([0, 1], ["Healthy", "AD/MCI"])
    a.set_yticks([0, 1], ["Healthy", "AD/MCI"])
    a.set_xlabel("Predicted")
    a.set_ylabel("True")
    a.grid(False)
    a.set_title("Paper's Slovak confusion matrix\n(full model)", loc="left")
    labels = ["Accuracy\n(model)", "Accuracy if\nalways\n\"healthy\"", "Sensitivity\n(patients\nfound)",
              "AD/MCI\nprecision"]
    vals = [acc, maj, sens, prec]
    cols = ["#2a78d6", "#8a8985", "#2a78d6", "#2a78d6"]
    bars = b.bar(range(4), vals, color=cols, width=0.62)
    for r, v in zip(bars, vals):
        b.text(r.get_x() + r.get_width() / 2, v + 0.015, f"{v:.1%}", ha="center", color=INK, fontsize=10)
    b.set_xticks(range(4), labels, fontsize=9)
    b.set_ylim(0, 1.1)
    b.set_ylabel("Share")
    b.grid(axis="x", visible=False)
    b.set_title(f"A do-nothing rule beats the model on accuracy\n({tn + fp} healthy, {tp + fn} AD/MCI speakers)",
                loc="left")
    save(fig, "fig1_paper_critique")
    write_tex("tab_paper_matrix",
              ["", "Pred. healthy", "Pred. AD/MCI"],
              [["True healthy", str(tn), str(fp)], ["True AD/MCI", str(fn), str(tp)]],
              rf"Confusion matrix read from Fig.~1(b) of the paper (Slovak, full model). "
              rf"Accuracy {acc:.1%}; always predicting healthy gives {maj:.1%}; AD/MCI precision {prec:.1%}.",
              "tab:paper_matrix", "lrr")


# ===================================================================== Fig 2: baselines vs speech (CV)
def fig2(cv):
    order = ["age only", "all metadata", "speech", "speech + all metadata",
             "speech (age/sex removed)", "speech (age/sex/month removed)"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax, task in zip(axes, TASKS):
        d = cv[cv.task == task].set_index("features")
        for k, f in enumerate(order):
            if f not in d.index:
                continue
            r = d.loc[f]
            y = len(order) - 1 - k
            ax.plot([r.auc_boot_lo, r.auc_boot_hi], [y, y], color=COL[f], lw=2, solid_capstyle="round")
            ax.plot(r.auc_mean_over_repeats, y, MARK[f], color=COL[f], ms=8, mec=SURF, mew=1.5)
            ax.text(min(r.auc_boot_hi + 0.012, 0.99), y, f"{r.auc_mean_over_repeats:.2f}", va="center",
                    fontsize=9, color=INK)
        ax.axvline(0.5, color=INK2, lw=1, ls=(0, (3, 3)))
        ax.set_xlim(0.45, 1.0)
        ax.set_yticks(range(len(order)), order[::-1])
        ax.set_xlabel("AUC (5x5-fold cross-validation, 95% CI over speakers)")
        n_pos = int(d.n_pos.iloc[0])
        ax.set_title(f"{TASK_NAME[task]} ({n_pos} patients)", loc="left")
        ax.grid(axis="y", visible=False)
    for ax in axes:
        ax.text(0.455, len(order) - 1.5, "chance", fontsize=8, color=INK2, va="center")
    fig.suptitle("Speech outperforms the age and metadata baselines in cross-validation",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", y=1.02)
    save(fig, "fig2_baselines")

    rows = []
    for task in TASKS:
        d = cv[cv.task == task].set_index("features")
        for f in order:
            if f in d.index:
                r = d.loc[f]
                rows.append([TASK_NAME[task] if f == "age only" else "", tex_escape(f),
                             fmt(r.auc_mean_over_repeats), ci(r.auc_boot_lo, r.auc_boot_hi),
                             fmt(r.auc_70plus), ci(r.auc_70plus_lo, r.auc_70plus_hi)])
    write_tex("tab_cv_auc", ["Task", "Features", "AUC", "95\\% CI", "AUC (70+)", "95\\% CI (70+)"], rows,
              "Out-of-fold AUC from 5x5-fold cross-validation over all speakers. Bootstrap CIs resample speakers.",
              "tab:cv_auc", "llrcrc")


def table_paired(pr):
    rows = []
    for _, r in pr.iterrows():
        rows.append([TASK_NAME.get(r.task.split(" (")[0], r.task) + (" (70+)" if "70+" in r.task else ""),
                     tex_escape(r.candidate), tex_escape(r.reference), f"{r.auc_diff:+.3f}",
                     ci(r.lo, r.hi), "yes" if r.ci_excludes_zero else "no"])
    write_tex("tab_paired", ["Task", "Candidate", "Reference", "AUC diff.", "95\\% CI", "CI excl. 0"], rows,
              "Paired AUC differences on the same out-of-fold scores (bootstrap over speakers).",
              "tab:paired", "lllrcc")


# ===================================================================== Fig 3: matched-pair AUC
def fig3(wm):
    conds = ["all pairs", "same month", "same 14-day window", "age within 5y", "same month + age within 5y"]
    clab = ["all\npairs", "same\nmonth", "same 14-day\nwindow", "age within\n5 years", "same month +\nage within 5y"]
    feats = ["age only", "all metadata", "speech", "speech + all metadata"]
    subsets = ["all ages", "aged 70+"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6), sharey=True)
    for i, subset in enumerate(subsets):
        for j, task in enumerate(TASKS):
            ax = axes[i, j]
            d = wm[(wm.task == task) & (wm.subset == subset)]
            for k, f in enumerate(feats):
                dd = d[d.features == f].set_index("condition").reindex(conds)
                x = np.arange(len(conds)) + (k - 1.5) * 0.15
                ok = dd.auc.notna().to_numpy()
                ax.vlines(x[ok], dd.lo[ok], dd.hi[ok], color=COL[f], lw=1.6)
                ax.plot(x[ok], dd.auc[ok], MARK[f], color=COL[f], ms=6.5, mec=SURF, mew=1.2,
                        label=f if (i == 0 and j == 0) else None)
            ax.axhline(0.5, color=INK2, lw=1, ls=(0, (3, 3)))
            ax.set_xticks(range(len(conds)), clab, fontsize=8.5)
            ax.set_ylim(0.3, 1.0)
            ax.grid(axis="x", visible=False)
            npat = int(d.n_patients.iloc[0]) if len(d) else 0
            ax.set_title(f"{TASK_NAME[task]}, {subset} ({npat} patients)", loc="left")
            if j == 0:
                ax.set_ylabel("AUC over patient-control pairs")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Speech largely keeps its AUC when patient-control pairs share recording month and age",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    save(fig, "fig3_matched_pairs")

    rows = []
    for task in TASKS:
        for cond in ["all pairs", "same month + age within 5y"]:
            d = wm[(wm.task == task) & (wm.subset == "all ages") & (wm.condition == cond)].set_index("features")
            if d.empty:
                continue
            rows.append([TASK_NAME[task], tex_escape(cond), f"{int(d.n_pairs.iloc[0]):,}"]
                        + [f"{d.loc[f].auc:.3f} {ci(d.loc[f].lo, d.loc[f].hi)}" for f in feats])
    write_tex("tab_matched", ["Task", "Pairs", "\\#pairs"] + [tex_escape(f) for f in feats], rows,
              "Conditional AUC: the share of patient-control pairs, restricted as shown, in which the patient scores higher. "
              "Brackets: 95\\% bootstrap CI over speakers.", "tab:matched", "lllrrrr")


# ===================================================================== Fig 4: temporal split
def fig4(tp, cv):
    feats = ["age only", "all metadata", "speech", "speech (age/sex removed)"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), sharey=True)
    for ax, task in zip(axes, TASKS):
        d = tp[(tp.task == task) & tp.auc.notna()].set_index("features")
        c = cv[cv.task == task].set_index("features") if cv is not None else None
        for k, f in enumerate(feats):
            if f not in d.index:
                continue
            y = len(feats) - 1 - k
            r = d.loc[f]
            ax.plot([r.lo, r.hi], [y + 0.1, y + 0.1], color=COL[f], lw=2, solid_capstyle="round")
            ax.plot(r.auc, y + 0.1, MARK[f], color=COL[f], ms=8, mec=SURF, mew=1.5,
                    label="train before 2023, test 2023+" if k == 0 else None)
            if c is not None and f in c.index:
                ax.plot(c.loc[f].auc_mean_over_repeats, y - 0.18, MARK[f], mfc="none", mec=COL[f], mew=1.6, ms=8,
                        label="cross-validation (all years)" if k == 0 else None)
        ax.axvline(0.5, color=INK2, lw=1, ls=(0, (3, 3)))
        ax.set_xlim(0.45, 1.0)
        ax.set_yticks(range(len(feats)), feats[::-1])
        ax.set_xlabel("AUC")
        npos = int(d.pos_test.iloc[0]) if len(d) else 0
        ax.set_title(f"{TASK_NAME[task]} ({npos} test patients)", loc="left")
        ax.grid(axis="y", visible=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.13), fontsize=9)
    fig.suptitle("Trained on earlier recordings, speech does worse on later ones than in cross-validation",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", y=1.02)
    save(fig, "fig4_temporal")
    rows = []
    for _, r in tp[tp.auc.notna()].iterrows():
        rows.append([TASK_NAME[r.task], tex_escape(r.features), f"{int(r.pos_train)}/{int(r.pos_test)}",
                     fmt(r.auc), ci(r.lo, r.hi)])
    write_tex("tab_temporal", ["Task", "Features", "Pos. train/test", "AUC", "95\\% CI"], rows,
              "Train on speakers recorded before 2023-01-01, test on speakers recorded on or after it. "
              "Small test sets: treat as exploratory.", "tab:temporal", "llrrc")


def table_batch(b):
    rows = [[tex_escape(r.target), str(int(r.n_healthy)), fmt(r.score, 2)] for _, r in b.iterrows()]
    write_tex("tab_batch", ["Predicted from speech", "Healthy speakers", "Score"], rows,
              "Can speech features predict recording circumstances? Healthy speakers only, 5-fold CV. "
              "R$^2$ of 0 and AUC of 0.5 mean no information.", "tab:batch", "lrr")


def write_numbers(cv, pr, wm, wmp, tp, b):
    """Write tables/numbers.tex: \newcommand macros so the report prose always matches the CSVs."""
    M = {}
    key = {"AD/MCI vs HC": "ad", "PD vs HC": "pd"}
    f3 = lambda x: f"{x:.3f}"
    sg = lambda x: f"{x:+.3f}".replace("-", "$-$")

    def put(name, val):
        M[name] = val

    nm = {"age only": "Age", "all metadata": "Meta", "speech": "Speech", "speech + all metadata": "SpeechMeta",
          "speech (age/sex removed)": "SpeechRes", "speech (age/sex/month removed)": "SpeechMonth"}
    if cv is not None:
        for _, r in cv.iterrows():
            k = key.get(r.task)
            if k and r.features in nm:
                put(f"{k}{nm[r.features]}", f3(r.auc_mean_over_repeats))
                put(f"{k}{nm[r.features]}Lo", f3(r.auc_boot_lo)); put(f"{k}{nm[r.features]}Hi", f3(r.auc_boot_hi))
        if "adSpeech" in M:
            se = (float(M["adSpeechHi"]) - float(M["adSpeechLo"])) / 3.92
            a = float(M["adSpeech"]); npairs = 197127
            put("adSpeechSE", f"{se:.4f}"); put("adNaiveSE", f"{(a*(1-a)/npairs)**0.5:.4f}")
            put("seRatio", f"{se/((a*(1-a)/npairs)**0.5):.0f}")
    if pr is not None:
        pn = {("speech + all metadata", "all metadata"): "AddMeta", ("speech (age/sex removed)", "age only"): "ResAge",
              ("speech (age/sex/month removed)", "age only"): "MonthAge"}
        for _, r in pr.iterrows():
            k = key.get(r.task)
            if k and (r.candidate, r.reference) in pn:
                n = f"{k}{pn[(r.candidate, r.reference)]}"
                put(n, sg(r.auc_diff)); put(n + "Lo", sg(r.lo)); put(n + "Hi", sg(r.hi))
    if wm is not None:
        w = wm[wm.subset == "all ages"]
        for t, k in key.items():
            g = w[(w.task == t)]
            m = g[(g.condition == "same month + age within 5y") & (g.features == "speech")]
            a = g[(g.condition == "same month + age within 5y") & (g.features == "age only")]
            al = g[(g.condition == "all pairs") & (g.features == "speech")]
            if len(m) and len(a) and len(al):
                put(f"{k}MatchSpeech", f3(m.auc.iloc[0])); put(f"{k}MatchAge", f3(a.auc.iloc[0]))
                put(f"{k}AllSpeech", f3(al.auc.iloc[0]))
                put(f"{k}MatchPairs", f"{int(m.n_pairs.iloc[0]):,}".replace(",", "{,}"))
                put(f"{k}MatchCov", str(int(m.n_patients_covered.iloc[0]))); put(f"{k}NPat", str(int(m.n_patients.iloc[0])))
            sp = g[g.features == "speech"].auc
            if len(sp):
                put(f"{k}CondMin", f"{sp.min():.2f}"); put(f"{k}CondMax", f"{sp.max():.2f}")
            cal = g[(g.condition == "age within 5y") & (g.features == "age only")]
            if len(cal):
                put(f"{k}CaliperAge", f3(cal.auc.iloc[0]))
        g70 = wm[(wm.subset == "aged 70+") & (wm.task == "AD/MCI vs HC") & (wm.condition == "same month + age within 5y")
                 & (wm.features == "speech")]
        if len(g70):
            put("adSeventyCov", str(int(g70.n_patients_covered.iloc[0]))); put("adSeventyN", str(int(g70.n_patients.iloc[0])))
        gm = wm[(wm.subset == "all ages") & (wm.task == "AD/MCI vs HC") & (wm.condition == "same month")
                & (wm.features == "age only")]
        if len(gm):
            put("adMonthAge", f3(gm.auc.iloc[0]))
    if wmp is not None:
        for t, k in key.items():
            for sub, tag in [("all ages", ""), ("aged 70+", "Seventy")]:
                g = wmp[(wmp.task == t) & (wmp.subset == sub) & (wmp.condition == "same month + age within 5y")
                        & (wmp.candidate == "speech") & (wmp.reference == "age only")]
                if len(g):
                    n = f"{k}Gap{tag}"
                    put(n, sg(g.auc_diff.iloc[0])); put(n + "Lo", sg(g.lo.iloc[0])); put(n + "Hi", sg(g.hi.iloc[0]))
        g = wmp[(wmp.task == "AD/MCI vs HC") & (wmp.subset == "all ages") & (wmp.condition == "same month")
                & (wmp.candidate == "speech (age/sex removed)") & (wmp.reference == "age only")]
        if len(g):
            put("adMonthGap", sg(g.auc_diff.iloc[0])); put("adMonthGapLo", sg(g.lo.iloc[0])); put("adMonthGapHi", sg(g.hi.iloc[0]))
    if tp is not None:
        for _, r in tp.iterrows():
            k = key.get(r.task)
            if k and r.features in nm and not pd.isna(r.auc):
                put(f"{k}T{nm[r.features]}", f3(r.auc)); put(f"{k}T{nm[r.features]}Lo", f3(r.lo)); put(f"{k}T{nm[r.features]}Hi", f3(r.hi))
                put(f"{k}TposTest", str(int(r.pos_test)))
    if b is not None:
        for _, r in b.iterrows():
            n = "batchAge" if r.target.startswith("age") else "batchDate" if r.target.startswith("rec") else "batchLQ"
            put(n, f"{r.score:.2f}")
    lines = ["% generated by 07_make_figures.py from reports/*.csv -- do not edit by hand"]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(M.items())]
    (TAB / "numbers.tex").write_text("\n".join(lines) + "\n")
    print("wrote", TAB / "numbers.tex", f"({len(M)} numbers)")


# ===================================================================== run
fig1()
cv = load("confound_cv.csv")
if cv is not None:
    fig2(cv)
pr = load("confound_paired.csv")
if pr is not None:
    table_paired(pr)
wm = load("within_month.csv")
if wm is not None:
    fig3(wm)
tp = load("confound_temporal.csv")
if tp is not None:
    fig4(tp, cv)
b = load("confound_batch_predictability.csv")
if b is not None:
    table_batch(b)
write_numbers(cv, pr, wm, load("within_month_paired.csv"), tp, b)
print("\nDone. Figures: reports/figures/   Tables: reports/tables/")
