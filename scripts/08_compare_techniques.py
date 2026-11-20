"""Phase 5: compare three course techniques on the same data, folds and checks.

    python3 scripts/08_compare_techniques.py         (about 10-20 minutes)
    EWA_N_REPEATS=1 python3 scripts/08_compare_techniques.py   (quick trial run)

Needs (all in cache/): speakers_clean.csv, splits.csv, task_features.csv, task_trill.npy

Techniques (Section 2 of the Course Techniques Report)
  L2 logistic        logistic regression, L2 penalty, C=0.01 fixed (exactly script 05's model)
  L1 logistic        lasso: L1 penalty, C tuned by inner 3-fold CV over L1_GRID
  RBF SVM            kernel SVM with a Gaussian kernel, C and gamma tuned by inner 3-fold CV

Protocol (identical for all three, so differences come from the technique)
  * Same 5x5 repeated stratified CV over all speakers as script 05 (seed 42).
  * All preprocessing (median impute, scale, Trill PCA to 32, age/sex residualisation)
    is fitted on the outer training fold only. Hyperparameters are chosen by inner CV
    on the outer training fold only, so test speakers never influence any choice.
    (Within the inner CV the unsupervised preprocessing was fitted on the whole outer
    training fold; that can only affect which hyperparameter wins, not the test scores.)
  * Feature sets: "speech" and "speech (age/sex removed)". "age only" (L2 logistic) is
    kept as the reference for matched-pair gaps, as in script 06.
  * Scores: logistic models give probabilities, the SVM a decision value. AUC only needs
    a ranking, but an SVM's scale can change between folds when its hyperparameters do,
    which can lower a pooled out-of-fold AUC. So we also report the mean of per-fold AUCs
    (scale-free) as a check.

Checks (the same three used for the L2 model in scripts 05 and 06)
  1. Overall out-of-fold AUC with speaker-bootstrap CIs.
  2. Conditional AUC: pairs in the same month, and same month + age within 5 years
     (script 06's definition); all ages and aged 70+.
  3. Temporal split: train before 2023-01-01, test on/after (hyperparameters tuned on
     the early speakers only). Exploratory: small test sets.
  Paired bootstrap differences against L2 logistic share the same speaker resamples.

Lasso feature selection (speech set): how often each feature has a non-zero weight across
the 25 outer folds, its average sign, and, among HEALTHY speakers only, how strongly the
raw feature correlates with recording date and with age. If the features the lasso keeps
are much more date-sensitive than average, the lasso is leaning on recording period.
Trill PCA components are refitted in every fold, so they are only counted as a group.

Outputs (aggregate numbers only, safe to commit):
  reports/techniques_cv.csv            AUC per technique x feature set x condition (+ CIs)
  reports/techniques_paired.csv        lasso/SVM minus L2, and every model minus age only
  reports/techniques_temporal.csv      temporal split per technique
  reports/techniques_lasso_features.csv   selection frequency + batch sensitivity per feature
  reports/techniques_lasso_summary.csv    per task: features kept, share by task, date sensitivity
  reports/techniques_hyperparams.csv   hyperparameters chosen in every outer fold
"""
import os
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
SEED, N_BOOT, N_PCA = 42, 500, 32
C_L2 = 0.01
N_SPLITS, N_REPEATS = 5, int(os.environ.get("EWA_N_REPEATS", 5))
N_INNER = 3
N_JOBS = int(os.environ.get("EWA_N_JOBS", -1))
AGE_TOL = 5.0
TASKS = ["phonation", "pataka", "picture", "naming"]
TEMPORAL_CUTOFF = pd.Timestamp("2023-01-01")
L1_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
SVM_C_GRID = [0.1, 1.0, 10.0]
SVM_GAMMA_FACTORS = [0.1, 1.0, 10.0]          # multiples of 1/(n_features * Var(X))

MODELS = ["L2 logistic", "L1 logistic", "RBF SVM"]
SETS = {"speech": False, "speech (age/sex removed)": True}   # name -> residualise age/sex
TASK_DEFS = {"AD/MCI vs HC": ["AD", "MCI"], "PD vs HC": ["PD"]}
CONDITIONS = ["all pairs", "same month", "same month + age within 5y"]

# ------------------------------------------------------------------ load (as in script 05)
spk = pd.read_csv("cache/speakers_clean.csv")
spl = pd.read_csv("cache/splits.csv")[["SPEAKER_CODE", "split"]]
tf = pd.read_csv("cache/task_features.csv")
trill = np.load("cache/task_trill.npy")
assert len(tf) == len(trill), "task_features.csv and task_trill.npy are misaligned"

df = spk.merge(spl, on="SPEAKER_CODE", how="inner").reset_index(drop=True)
df["lowq"] = df.LOW_QUALITY.astype(int)
df = df.rename(columns={"EDUCATION_YEARS": "edu_years"})
codes = df.SPEAKER_CODE
rec_date = pd.Timestamp("2021-01-01") + pd.to_timedelta(df.rec_days, unit="D")
MONTH = rec_date.dt.strftime("%Y-%m").to_numpy()
AGE = df.age.to_numpy(float)

feat_cols = [c for c in tf.columns if c.startswith(("gm_", "ns_"))]
G = np.hstack([
    tf[tf.task == t].set_index("SPEAKER_CODE")[feat_cols].reindex(codes).to_numpy(float)
    for t in TASKS])
GM_NAMES = [f"{t}:{c}" for t in TASKS for c in feat_cols]
tr_df = pd.DataFrame(trill)
tr_df["SPEAKER_CODE"] = tf.SPEAKER_CODE.to_numpy()
T = tr_df.groupby("SPEAKER_CODE").mean().reindex(codes).to_numpy(float)
D = df[["age", "male", "edu_years", "rec_days", "lowq"]].to_numpy(float)
BLOCKS = {"age": D[:, [0]], "gmns": G, "trill": T, "agesex": D[:, [0, 1]]}
N_GM = G.shape[1]


def prep(name, tr, others):
    """Median-impute and scale on train rows; PCA for Trill. Returns [train, *others]."""
    X = BLOCKS[name]
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X[tr])
    sc = StandardScaler().fit(imp.transform(X[tr]))
    out = [sc.transform(imp.transform(X[i])) for i in [tr] + list(others)]
    if name == "trill":
        pca = PCA(N_PCA, random_state=SEED).fit(out[0])
        out = [pca.transform(z) for z in out]
    return out


def design(blocks, tr, te, resid):
    parts = [prep(b, tr, [te]) for b in blocks]
    Xtr, Xte = (np.hstack([p[k] for p in parts]) for k in (0, 1))
    if resid:
        ctr, cte = prep("agesex", tr, [te])
        lin = Ridge(alpha=1e-3).fit(ctr, Xtr)
        Xtr, Xte = Xtr - lin.predict(ctr), Xte - lin.predict(cte)
    return Xtr, Xte


def fit(model, Xtr, ytr):
    """Return (fitted classifier, chosen hyperparameters). Tuning uses the training rows only."""
    if model == "L2 logistic":
        return LogisticRegression(C=C_L2, class_weight="balanced", max_iter=3000).fit(Xtr, ytr), {"C": C_L2}
    inner = StratifiedKFold(N_INNER, shuffle=True, random_state=SEED)
    if model == "L1 logistic":
        base = LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=5000)
        grid = {"C": L1_GRID}
    else:
        g0 = 1.0 / (Xtr.shape[1] * Xtr.var())
        base = SVC(kernel="rbf", class_weight="balanced")
        grid = {"C": SVM_C_GRID, "gamma": [g0 * f for f in SVM_GAMMA_FACTORS]}
    gs = GridSearchCV(base, grid, scoring="roc_auc", cv=inner, n_jobs=N_JOBS).fit(Xtr, ytr)
    params = dict(gs.best_params_)
    if "gamma" in params:   # report gamma as the multiple of the default scale
        params["gamma"] = round(params["gamma"] / g0, 3)
    return gs.best_estimator_, params


def score(clf, X):
    return clf.predict_proba(X)[:, 1] if isinstance(clf, LogisticRegression) else clf.decision_function(X)


def pair_index(pos, neg, cond, ok_mask, month, age):
    pi, nj = np.meshgrid(pos, neg, indexing="ij")
    pi, nj = pi.ravel(), nj.ravel()
    ok = ok_mask[pi] & ok_mask[nj]
    if "month" in cond:
        ok &= month[pi] == month[nj]
    if "age" in cond:
        ok &= np.abs(age[pi] - age[nj]) <= AGE_TOL
    return pi[ok], nj[ok]


def boot_auc(conc, pi, nj, counts, chunk=50):
    """Bootstrap conditional AUC; pair weight = product of the two speakers' counts (script 06)."""
    res = []
    for a in range(0, len(counts), chunk):
        w = counts[a:a + chunk][:, pi] * counts[a:a + chunk][:, nj]
        with np.errstate(invalid="ignore", divide="ignore"):
            res.append((w @ conc) / w.sum(1))
    return np.concatenate(res)


def pct(x):
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    return round(lo, 3), round(hi, 3)


rng = np.random.default_rng(SEED)
cv_rows, pair_rows, temp_rows, hp_rows, lasso_rows, lasso_sum_rows = [], [], [], [], [], []

for task, pos_groups in TASK_DEFS.items():
    print(f"\n### {task}: cross-validation ({N_REPEATS}x{N_SPLITS} outer folds)", flush=True)
    m = np.where(df.group.isin(pos_groups + ["HC"]))[0]
    y_all = df.group.isin(pos_groups).astype(int).to_numpy()
    y = y_all[m]
    keys = [("L2 logistic", "age only")] + [(mod, s) for s in SETS for mod in MODELS]
    oof = {k: np.zeros((N_REPEATS, len(m))) for k in keys}
    fold_auc = {k: [] for k in keys}
    sel_count = np.zeros(N_GM + N_PCA)
    sign_sum = np.zeros(N_GM + N_PCA)
    n_kept = []
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    n_folds = N_SPLITS * N_REPEATS
    for fold, (tr_l, te_l) in enumerate(rskf.split(m, y)):
        rep, tr, te = fold // N_SPLITS, m[tr_l], m[te_l]
        Xtr, Xte = design(["age"], tr, te, False)
        clf, _ = fit("L2 logistic", Xtr, y_all[tr])
        s = score(clf, Xte)
        oof[("L2 logistic", "age only")][rep, te_l] = s
        fold_auc[("L2 logistic", "age only")].append(roc_auc_score(y_all[te], s))
        for sname, resid in SETS.items():
            Xtr, Xte = design(["gmns", "trill"], tr, te, resid)
            for mod in MODELS:
                clf, params = fit(mod, Xtr, y_all[tr])
                s = score(clf, Xte)
                oof[(mod, sname)][rep, te_l] = s
                fold_auc[(mod, sname)].append(roc_auc_score(y_all[te], s))
                hp_rows.append(dict(task=task, features=sname, technique=mod, fold=fold, **params))
                if mod == "L1 logistic" and sname == "speech":
                    w = clf.coef_.ravel()
                    sel_count += w != 0
                    sign_sum += np.sign(w)
                    n_kept.append(int((w != 0).sum()))
        print(f"  fold {fold + 1}/{n_folds} done", flush=True)
    sc = {k: v.mean(0) for k, v in oof.items()}

    # ---------------- overall + conditional AUC, paired differences (shared resamples)
    month, age = MONTH[m], AGE[m]
    counts = rng.multinomial(len(m), np.ones(len(m)) / len(m), size=N_BOOT).astype(np.float64)
    for subset, ok_mask in [("all ages", np.ones(len(m), bool)), ("aged 70+", np.nan_to_num(age, nan=0) >= 70)]:
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        for cond in CONDITIONS:
            pi, nj = pair_index(pos, neg, cond, ok_mask, month, age)
            if len(pi) < 200:
                continue
            est, boots = {}, {}
            for k in keys:
                s = sc[k]
                conc = (s[pi] > s[nj]).astype(float) + 0.5 * (s[pi] == s[nj])
                est[k], boots[k] = float(conc.mean()), boot_auc(conc, pi, nj, counts)
                lo, hi = pct(boots[k])
                row = dict(task=task, subset=subset, condition=cond, technique=k[0], features=k[1],
                           n_pairs=len(pi), n_patients_covered=len(np.unique(pi)),
                           n_patients=int(y[ok_mask].sum()), auc=round(est[k], 3), lo=lo, hi=hi)
                if cond == "all pairs" and subset == "all ages":
                    row["mean_fold_auc"] = round(float(np.mean(fold_auc[k])), 3)
                    row["sd_fold_auc"] = round(float(np.std(fold_auc[k])), 3)
                cv_rows.append(row)
            comparisons = [((mod, s), ("L2 logistic", s)) for s in SETS for mod in MODELS[1:]]
            comparisons += [((mod, s), ("L2 logistic", "age only")) for s in SETS for mod in MODELS]
            for cand, ref in comparisons:
                lo, hi = pct(boots[cand] - boots[ref])
                pair_rows.append(dict(task=task, subset=subset, condition=cond,
                                      candidate=f"{cand[0]} | {cand[1]}", reference=f"{ref[0]} | {ref[1]}",
                                      auc_diff=round(est[cand] - est[ref], 3), lo=lo, hi=hi,
                                      ci_excludes_zero=bool(lo > 0 or hi < 0)))

    # ---------------- lasso feature selection
    hc_rows = m[y == 0]
    date_hc = df.rec_days.to_numpy(float)[hc_rows]
    age_hc = AGE[hc_rows]
    Ghc = pd.DataFrame(G[hc_rows])
    r_date = Ghc.corrwith(pd.Series(date_hc)).abs().to_numpy()
    r_age = Ghc.corrwith(pd.Series(age_hc)).abs().to_numpy()
    frac = sel_count / n_folds
    for j, name in enumerate(GM_NAMES):
        lasso_rows.append(dict(task=task, feature=name, speech_task=name.split(":")[0],
                               frac_folds_selected=round(frac[j], 3),
                               mean_sign=round(sign_sum[j] / max(sel_count[j], 1), 2),
                               abs_corr_date_hc=round(r_date[j], 3), abs_corr_age_hc=round(r_age[j], 3)))
    lasso_rows.append(dict(task=task, feature=f"trill: any of {N_PCA} PCA components",
                           speech_task="all (Trill)", frac_folds_selected=round(float((sel_count[N_GM:] > 0).mean()), 3),
                           mean_sign=np.nan, abs_corr_date_hc=np.nan, abs_corr_age_hc=np.nan))
    stable = frac[:N_GM] >= 0.8
    summ = dict(task=task, n_features=N_GM + N_PCA, mean_kept_per_fold=round(float(np.mean(n_kept)), 1),
                n_stable_80pct=int(stable.sum()),
                mean_trill_pcs_kept=round(float(sel_count[N_GM:].sum() / n_folds), 1),
                date_corr_all=round(float(np.nanmean(r_date)), 3),
                date_corr_stable=round(float(np.nanmean(r_date[stable])), 3) if stable.any() else np.nan,
                age_corr_all=round(float(np.nanmean(r_age)), 3),
                age_corr_stable=round(float(np.nanmean(r_age[stable])), 3) if stable.any() else np.nan)
    # share of selections by speech task (GeMAPS/NeuroSpeech only), and a permutation p-value:
    # are stable features more date-sensitive than random sets of the same size?
    for t_ in TASKS:
        idx = [j for j, nm in enumerate(GM_NAMES) if nm.startswith(t_ + ":")]
        summ[f"share_{t_}"] = round(float(sel_count[idx].sum() / max(sel_count[:N_GM].sum(), 1)), 3)
    if stable.any():
        valid = ~np.isnan(r_date)
        k = int((stable & valid).sum())
        obs = np.nanmean(r_date[stable])
        null = [np.mean(rng.choice(r_date[valid], k, replace=False)) for _ in range(2000)]
        summ["date_corr_perm_p"] = round(float((np.sum(np.array(null) >= obs) + 1) / 2001), 4)
    lasso_sum_rows.append(summ)

    # ---------------- temporal split
    late = (rec_date >= TEMPORAL_CUTOFF).to_numpy()
    sel = df.group.isin(pos_groups + ["HC"]).to_numpy()
    tr, te = np.where(sel & ~late)[0], np.where(sel & late)[0]
    yt = y_all[te]
    n_pos_tr, n_pos_te = int(y_all[tr].sum()), int(yt.sum())
    print(f"### {task}: temporal split ({n_pos_tr} train / {n_pos_te} test patients)", flush=True)
    if min(n_pos_tr, n_pos_te) < 3 or yt.min() == yt.max():
        temp_rows.append(dict(task=task, technique="(too few positives)", features="", n_train=len(tr),
                              n_test=len(te), pos_train=n_pos_tr, pos_test=n_pos_te))
        continue
    idx = []
    while len(idx) < N_BOOT:
        i = rng.integers(0, len(te), len(te))
        if yt[i].min() != yt[i].max():
            idx.append(i)
    tscore = {}
    for sname, resid in SETS.items():
        Xtr, Xte = design(["gmns", "trill"], tr, te, resid)
        for mod in MODELS:
            clf, params = fit(mod, Xtr, y_all[tr])
            tscore[(mod, sname)] = s = score(clf, Xte)
            hp_rows.append(dict(task=task, features=sname, technique=mod, fold="temporal", **params))
            b = np.array([roc_auc_score(yt[i], s[i]) for i in idx])
            row = dict(task=task, technique=mod, features=sname, n_train=len(tr), n_test=len(te),
                       pos_train=n_pos_tr, pos_test=n_pos_te, auc=round(roc_auc_score(yt, s), 3))
            row["lo"], row["hi"] = pct(b)
            if mod != "L2 logistic":
                ref = tscore[("L2 logistic", sname)]
                d = np.array([roc_auc_score(yt[i], s[i]) - roc_auc_score(yt[i], ref[i]) for i in idx])
                row["diff_vs_L2"] = round(row["auc"] - roc_auc_score(yt, ref), 3)
                row["diff_lo"], row["diff_hi"] = pct(d)
            temp_rows.append(row)

# ------------------------------------------------------------------ save + print
Path("reports").mkdir(exist_ok=True)
out = {
    "techniques_cv": pd.DataFrame(cv_rows),
    "techniques_paired": pd.DataFrame(pair_rows),
    "techniques_temporal": pd.DataFrame(temp_rows),
    "techniques_lasso_features": pd.DataFrame(lasso_rows).sort_values(
        ["task", "frac_folds_selected"], ascending=[True, False]),
    "techniques_lasso_summary": pd.DataFrame(lasso_sum_rows),
    "techniques_hyperparams": pd.DataFrame(hp_rows),
}
for name, d in out.items():
    d.to_csv(f"reports/{name}.csv", index=False)
    print("wrote", f"reports/{name}.csv")

pd.set_option("display.width", 230)
cv = out["techniques_cv"]
print("\n##### AUC by technique (speech features)")
key = cv[(cv.subset == "all ages") & (cv.features == "speech")]
print(key.pivot_table(index=["task", "technique"], columns="condition", values="auc", sort=False).to_string())
print("\n##### Paired differences vs L2 logistic (all ages)")
pr = out["techniques_paired"]
print(pr[(pr.subset == "all ages") & pr.reference.str.startswith("L2 logistic | speech")]
      [["task", "condition", "candidate", "auc_diff", "lo", "hi", "ci_excludes_zero"]].to_string(index=False))
print("\n##### Temporal split")
print(out["techniques_temporal"].to_string(index=False))
print("\n##### Lasso summary")
print(out["techniques_lasso_summary"].T.to_string())
print("\n##### Hyperparameters chosen (most common per task x set x technique)")
hp = out["techniques_hyperparams"]
for (t, f, mod), g in hp[hp.technique != "L2 logistic"].groupby(["task", "features", "technique"], sort=False):
    cols = [c for c in ["C", "gamma"] if c in g and g[c].notna().any()]
    common = Counter(tuple(r) for r in g[cols].itertuples(index=False)).most_common(1)[0]
    print(f"  {t} | {f} | {mod}: {dict(zip(cols, common[0]))} in {common[1]}/{len(g)} fits")
