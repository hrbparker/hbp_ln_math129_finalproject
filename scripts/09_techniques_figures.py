"""Phase 5b: figure, LaTeX tables and number macros for the technique comparison.

    python3 scripts/09_techniques_figures.py        (after 08_compare_techniques.py)

Reads:  reports/techniques_cv.csv, techniques_paired.csv, techniques_temporal.csv,
        reports/techniques_lasso_features.csv, techniques_lasso_summary.csv
Writes (aggregate numbers only, safe to commit):
    reports/figures/fig5_techniques.{png,pdf}   AUC of the three techniques: overall, matched pairs, temporal
    reports/tables/tab_techniques.tex           AUC table (both feature sets)
    reports/tables/tab_techniques_paired.tex    lasso / SVM minus L2 logistic, with paired CIs
    reports/tables/tab_lasso.tex                what the lasso keeps, and how date-sensitive it is
    reports/tables/tab_lasso_top.tex            the most often selected features
    reports/tables/techniques_numbers.tex       \\newcommand macros used by Section 3 of the report

Same look as 07_make_figures.py. Copy reports/figures and reports/tables next to the report.
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
TECH = ["L2 logistic", "L1 logistic", "RBF SVM"]
TCOL = {"L2 logistic": "#2a78d6", "L1 logistic": "#eb6834", "RBF SVM": "#7a4bc2", "age only": "#8a8985"}
TMARK = {"L2 logistic": "D", "L1 logistic": "s", "RBF SVM": "^", "age only": "o"}
TLABEL = {"L2 logistic": "L2 logistic (ridge)", "L1 logistic": "L1 logistic (lasso)",
          "RBF SVM": "Kernel SVM (RBF)", "age only": "age only (L2 logistic)"}
TASKS = ["AD/MCI vs HC", "PD vs HC"]
TASK_NAME = {"AD/MCI vs HC": "AD/MCI vs. healthy", "PD vs HC": "Parkinson's vs. healthy"}
SETS = ["speech", "speech (age/sex removed)"]
MATCH = "same month + age within 5y"

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
        raise SystemExit(f"{p} not found: run scripts/08_compare_techniques.py first")
    return pd.read_csv(p)


def fmt(x, nd=3):
    return "--" if pd.isna(x) else f"{x:.{nd}f}"


def ci(lo, hi):
    return f"[{fmt(lo, 2)}, {fmt(hi, 2)}]"


def sgn(x):
    return "--" if pd.isna(x) else f"{x:+.3f}"


def tex_escape(s):
    return str(s).replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")


def write_tex(name, header, rows, caption, label, align):
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\resizebox{\linewidth}{!}{%", rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [(r"\midrule" if r == "MID" else " & ".join(r) + r" \\") for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print("wrote", TAB / f"{name}.tex")


cv = load("techniques_cv.csv")
pr = load("techniques_paired.csv")
tp = load("techniques_temporal.csv")
lf = load("techniques_lasso_features.csv")
ls = load("techniques_lasso_summary.csv")
allages = cv[cv.subset == "all ages"]


def cv_row(task, tech, feats, cond):
    r = allages[(allages.task == task) & (allages.technique == tech) & (allages.features == feats)
                & (allages.condition == cond)]
    return r.iloc[0] if len(r) else None


def tp_row(task, tech, feats):
    r = tp[(tp.task == task) & (tp.technique == tech) & (tp.features == feats)]
    return r.iloc[0] if len(r) and "auc" in r and not pd.isna(r.iloc[0].get("auc", np.nan)) else None


# ===================================================================== Fig 5
groups = [("Cross-validation\n(all pairs)", "all pairs"), ("Matched pairs\n(same month, age ±5y)", MATCH),
          ("Temporal split\n(train <2023, test 2023+)", "temporal")]
fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), sharey=True)
for i, feats in enumerate(SETS):
    for j, task in enumerate(TASKS):
        ax = axes[i, j]
        series = TECH + (["age only"] if feats == "speech" else [])
        for k, tech in enumerate(series):
            off = (k - (len(series) - 1) / 2) * 0.17
            for g, (_, cond) in enumerate(groups):
                if tech == "age only":
                    if cond == "temporal":
                        continue
                    r = cv_row(task, "L2 logistic", "age only", cond)
                elif cond == "temporal":
                    r = tp_row(task, tech, feats)
                else:
                    r = cv_row(task, tech, feats, cond)
                if r is None:
                    continue
                ax.vlines(g + off, r.lo, r.hi, color=TCOL[tech], lw=1.8)
                ax.plot(g + off, r.auc, TMARK[tech], color=TCOL[tech], ms=7, mec=SURF, mew=1.2,
                        label=TLABEL[tech] if g == 0 or (tech == "age only" and g == 0) else None)
        ax.axhline(0.5, color=INK2, lw=1, ls=(0, (3, 3)))
        ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=8.5)
        ax.set_ylim(0.3, 1.0)
        ax.grid(axis="x", visible=False)
        npat = cv_row(task, "L2 logistic", feats, "all pairs")
        ax.set_title(f"{TASK_NAME[task]} | {feats}", loc="left", fontsize=10)
        if j == 0:
            ax.set_ylabel("AUC (95% CI over speakers)")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=9)
fig.suptitle("The three techniques agree in every check; the Parkinson's drop over time is shared by all three",
             x=0.01, ha="left", fontsize=11, fontweight="bold", y=1.0)
fig.tight_layout(rect=(0, 0.04, 1, 0.97))
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig5_techniques.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", FIG / "fig5_techniques.png")

# ===================================================================== Table: AUC
rows = []
for task in TASKS:
    for feats in SETS:
        for tech in TECH:
            a, mt, t = cv_row(task, tech, feats, "all pairs"), cv_row(task, tech, feats, MATCH), tp_row(task, tech, feats)
            rows.append([TASK_NAME[task] if (feats == SETS[0] and tech == TECH[0]) else "",
                         tex_escape(feats) if tech == TECH[0] else "", TLABEL[tech],
                         f"{fmt(a.auc)} {ci(a.lo, a.hi)}" if a is not None else "--",
                         fmt(a.mean_fold_auc) if a is not None else "--",
                         f"{fmt(mt.auc)} {ci(mt.lo, mt.hi)}" if mt is not None else "--",
                         f"{fmt(t.auc)} {ci(t.lo, t.hi)}" if t is not None else "--"])
    if task != TASKS[-1]:
        rows.append("MID")
write_tex("tab_techniques",
          ["Task", "Features", "Technique", "CV AUC [95\\% CI]", "Mean fold AUC", "Matched AUC", "Temporal AUC"],
          rows, "AUC of the three techniques on identical folds. CV: 5x5-fold out-of-fold scores, all pairs. "
          "Matched: pairs recorded in the same month with ages within 5 years. Temporal: train before 2023, test 2023+. "
          "Mean fold AUC is the average of per-fold AUCs, which does not depend on score scale.",
          "tab:techniques", "lllcccc")

# ===================================================================== Table: paired differences vs L2
rows = []
allp = pr[pr.subset == "all ages"]
for task in TASKS:
    for feats in SETS:
        for tech in TECH[1:]:
            cand, ref = f"{tech} | {feats}", f"L2 logistic | {feats}"
            cells = []
            for cond in ["all pairs", MATCH]:
                r = allp[(allp.task == task) & (allp.condition == cond) & (allp.candidate == cand) & (allp.reference == ref)]
                cells.append(f"{sgn(r.auc_diff.iloc[0])} {ci(r.lo.iloc[0], r.hi.iloc[0])}" if len(r) else "--")
            t = tp_row(task, tech, feats)
            cells.append(f"{sgn(t.diff_vs_L2)} {ci(t.diff_lo, t.diff_hi)}" if t is not None else "--")
            rows.append([TASK_NAME[task] if (feats == SETS[0] and tech == TECH[1]) else "",
                         tex_escape(feats) if tech == TECH[1] else "", TLABEL[tech]] + cells)
    if task != TASKS[-1]:
        rows.append("MID")
write_tex("tab_techniques_paired", ["Task", "Features", "Technique", "CV", "Matched", "Temporal"], rows,
          "AUC difference from L2 logistic regression on the same speakers (positive: better than L2). "
          "Brackets: paired 95\\% bootstrap CI over speakers.", "tab:techniques_paired", "lllccc")

# ===================================================================== Tables: lasso
TASKS4 = ["phonation", "pataka", "picture", "naming"]
rows = []
for _, r in ls.iterrows():
    rows.append([TASK_NAME[r.task], f"{r.mean_kept_per_fold:.1f} / {int(r.n_features)}", str(int(r.n_stable_80pct)),
                 f"{r.mean_trill_pcs_kept:.1f}"] + [f"{r[f'share_{t}']:.0%}".replace("%", r"\%") for t in TASKS4]
                + [fmt(r.date_corr_stable, 2), fmt(r.date_corr_all, 2), fmt(r.get("date_corr_perm_p", np.nan), 3)])
write_tex("tab_lasso", ["Task", "Kept per fold", "Kept in $\\geq$80\\%", "Trill PCs kept"]
          + [t.capitalize() for t in TASKS4] + ["$|r_{\\text{date}}|$ kept", "$|r_{\\text{date}}|$ all", "perm.\\ $p$"],
          rows, "What the lasso keeps (speech features, 25 outer folds). Task columns: share of kept GeMAPS/NeuroSpeech "
          "features by speech task. $|r_{\\text{date}}|$: mean absolute correlation with recording date among healthy "
          "speakers, for features kept in at least 80\\% of folds versus all features; perm.\\ $p$: chance that a random "
          "set of the same size is at least as date-sensitive.", "tab:lasso", "lrrrrrrrrrr")
rows = []
for task in TASKS:
    top = lf[(lf.task == task) & ~lf.feature.str.startswith("trill")].nlargest(8, "frac_folds_selected")
    for k, (_, r) in enumerate(top.iterrows()):
        rows.append([TASK_NAME[task] if k == 0 else "", f"\\texttt{{{tex_escape(r.feature)}}}",
                     f"{r.frac_folds_selected:.0%}".replace("%", r"\%"),
                     "higher" if r.mean_sign > 0 else "lower", fmt(r.abs_corr_date_hc, 2), fmt(r.abs_corr_age_hc, 2)])
    if task != TASKS[-1]:
        rows.append("MID")
write_tex("tab_lasso_top", ["Task", "Feature (task:feature)", "Folds kept", "In patients",
                            "$|r_{\\text{date}}|$ (HC)", "$|r_{\\text{age}}|$ (HC)"], rows,
          "Features the lasso keeps most often. ``In patients'': sign of the weight (higher or lower values push towards "
          "the patient class). Correlations are among healthy speakers only.", "tab:lasso_top", "llrlrr")

# ===================================================================== macros for Section 3
M = {}
key = {"AD/MCI vs HC": "Ad", "PD vs HC": "Pd"}
tk = {"L2 logistic": "Ltwo", "L1 logistic": "Lone", "RBF SVM": "Svm"}
fk = {"speech": "", "speech (age/sex removed)": "Res"}
f3 = lambda x: "--" if pd.isna(x) else f"{x:.3f}"
sg = lambda x: "--" if pd.isna(x) else f"{x:+.3f}".replace("-", "$-$")
for task, k in key.items():
    for feats, fs in fk.items():
        for tech, tt in tk.items():
            for cond, cs in [("all pairs", "All"), (MATCH, "Match")]:
                r = cv_row(task, tech, feats, cond)
                if r is not None:
                    n = f"tq{k}{tt}{fs}{cs}"
                    M[n], M[n + "Lo"], M[n + "Hi"] = f3(r.auc), f3(r.lo), f3(r.hi)
                    if cond == "all pairs":
                        M[f"tq{k}{tt}{fs}Fold"] = f3(r.mean_fold_auc)
            t = tp_row(task, tech, feats)
            if t is not None:
                n = f"tq{k}{tt}{fs}Temp"
                M[n], M[n + "Lo"], M[n + "Hi"] = f3(t.auc), f3(t.lo), f3(t.hi)
                M[f"tq{k}TempPos"] = str(int(t.pos_test))
                if tech != "L2 logistic":
                    M[f"tq{k}{tt}{fs}TempDiff"] = sg(t.diff_vs_L2)
                    M[f"tq{k}{tt}{fs}TempDiffLo"], M[f"tq{k}{tt}{fs}TempDiffHi"] = sg(t.diff_lo), sg(t.diff_hi)
            if tech != "L2 logistic":
                for cond, cs in [("all pairs", "All"), (MATCH, "Match")]:
                    r = allp[(allp.task == task) & (allp.condition == cond)
                             & (allp.candidate == f"{tech} | {feats}") & (allp.reference == f"L2 logistic | {feats}")]
                    if len(r):
                        n = f"tq{k}{tt}{fs}{cs}Diff"
                        M[n], M[n + "Lo"], M[n + "Hi"] = sg(r.auc_diff.iloc[0]), sg(r.lo.iloc[0]), sg(r.hi.iloc[0])
            r = allp[(allp.task == task) & (allp.condition == MATCH)
                     & (allp.candidate == f"{tech} | {feats}") & (allp.reference == "L2 logistic | age only")]
            if len(r):
                n = f"tq{k}{tt}{fs}GapAge"
                M[n], M[n + "Lo"], M[n + "Hi"] = sg(r.auc_diff.iloc[0]), sg(r.lo.iloc[0]), sg(r.hi.iloc[0])
    r = cv_row(task, "L2 logistic", "age only", MATCH)
    if r is not None:
        M[f"tq{k}AgeMatch"] = f3(r.auc)
        M[f"tq{k}MatchPairs"] = f"{int(r.n_pairs):,}".replace(",", "{,}")
    s = ls[ls.task == task]
    if len(s):
        s = s.iloc[0]
        M[f"tq{k}LassoKept"] = f"{s.mean_kept_per_fold:.1f}"
        M[f"tq{k}LassoStable"] = str(int(s.n_stable_80pct))
        M[f"tq{k}LassoTrill"] = f"{s.mean_trill_pcs_kept:.1f}"
        M[f"tq{k}LassoDateStable"] = fmt(s.date_corr_stable, 2)
        M[f"tq{k}LassoDateAll"] = fmt(s.date_corr_all, 2)
        M[f"tq{k}LassoAgeStable"] = fmt(s.age_corr_stable, 2)
        M[f"tq{k}LassoAgeAll"] = fmt(s.age_corr_all, 2)
        M[f"tq{k}LassoP"] = fmt(s.get("date_corr_perm_p", np.nan), 3)
        # same permutation test for AGE sensitivity, from the per-feature table (seeded)
        g = lf[(lf.task == task) & ~lf.feature.str.startswith("trill")]
        st = g[g.frac_folds_selected >= 0.8].abs_corr_age_hc.dropna()
        pool = g.abs_corr_age_hc.dropna().to_numpy()
        if len(st):
            prng = np.random.default_rng(42)
            null = np.array([prng.choice(pool, len(st), replace=False).mean() for _ in range(5000)])
            M[f"tq{k}LassoAgeP"] = f"{(np.sum(null >= st.mean()) + 1) / 5001:.3f}"
        for t in TASKS4:
            M[f"tq{k}LassoShare{t.capitalize()}"] = f"{s[f'share_{t}']:.0%}".replace("%", r"\%")
        best = max(TASKS4, key=lambda t: s[f"share_{t}"])
        M[f"tq{k}LassoTopTask"] = best
        M[f"tq{k}LassoTopShare"] = f"{s[f'share_{best}']:.0%}".replace("%", r"\%")
lines = ["% generated by 09_techniques_figures.py from reports/techniques_*.csv -- do not edit by hand"]
lines += [rf"\providecommand{{\{k}}}{{}}\renewcommand{{\{k}}}{{{v}}}" for k, v in sorted(M.items())]
(TAB / "techniques_numbers.tex").write_text("\n".join(lines) + "\n")
print("wrote", TAB / "techniques_numbers.tex", f"({len(M)} numbers)")
print("\nDone. Figure: reports/figures/fig5_techniques.*   Tables: reports/tables/tab_techniques*.tex, tab_lasso*.tex")
