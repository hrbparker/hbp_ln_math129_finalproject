"""Phase 3: stress-test the claim "speech adds information beyond age".

Needs (all in cache/): speakers_clean.csv, splits.csv, task_features.csv, task_trill.npy
    python3 scripts/05_confound_checks.py          (takes several minutes)

Checks
  A  Repeated stratified 5x5 cross-validation over ALL speakers (train/val/test
     pooled), so conclusions do not hang on one split with ~30 positives.
     Paired bootstrap (over speakers) of AUC differences on out-of-fold scores.
  B  Can speech encode the recording batch? Among HEALTHY speakers only (so disease
     cannot drive it): CV prediction of age (R^2), recording date (R^2) and the
     LOW_QUALITY flag (AUC) from speech features.
  C  Residualisation: speech with age+sex removed, and with age+sex+recording MONTH
     removed (month dummies). If AUC survives the second, the signal is not a
     simple month effect. Linear adjustment only.
  D  Speakers aged 70+ only, with bootstrap CIs (the lead from script 04).
  E  Temporal split: train on speakers recorded before 2023-01-01, test on
     2023-01-01 and later. Positives per side are printed; treat as exploratory.

Choices: logistic regression, C=0.01, class_weight="balanced" (fixed, not tuned,
to avoid leaking). All preprocessing is fitted inside each training fold.

Outputs: reports/confound_cv.csv, reports/confound_paired.csv,
         reports/confound_batch_predictability.csv, reports/confound_temporal.csv
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED, N_BOOT, N_PCA, C = 42, 1000, 32, 0.01
N_SPLITS, N_REPEATS = 5, 5
TASKS = ["phonation", "pataka", "picture", "naming"]
TEMPORAL_CUTOFF = pd.Timestamp("2023-01-01")

# ------------------------------------------------------------------ load
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
df["month"] = rec_date.dt.strftime("%Y-%m")

feat_cols = [c for c in tf.columns if c.startswith(("gm_", "ns_"))]
G = np.hstack([
    tf[tf.task == t].set_index("SPEAKER_CODE")[feat_cols].reindex(codes).to_numpy(float)
    for t in TASKS])
tr_df = pd.DataFrame(trill)
tr_df["SPEAKER_CODE"] = tf.SPEAKER_CODE.to_numpy()
T = tr_df.groupby("SPEAKER_CODE").mean().reindex(codes).to_numpy(float)

D = df[["age", "male", "edu_years", "rec_days", "lowq"]].to_numpy(float)
BLOCKS = {"age": D[:, [0]], "meta": D, "gmns": G, "trill": T, "agesex": D[:, [0, 1]]}
MONTHS = df.month.to_numpy()


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


def covariates(tr, others, use_month):
    cov = prep("agesex", tr, others)
    if not use_month:
        return cov
    cats = sorted(set(MONTHS[tr]))
    mk = lambda idx: (MONTHS[idx][:, None] == np.array(cats)[None, :]).astype(float)
    return [np.hstack([c, mk(i)]) for c, i in zip(cov, [tr] + list(others))]


def design(blocks, tr, te, resid=None):
    """resid: None, 'agesex' or 'agesex+month'."""
    parts = [prep(b, tr, [te]) for b in blocks]
    Xtr, Xte = (np.hstack([p[k] for p in parts]) for k in (0, 1))
    if resid:
        ctr, cte = covariates(tr, [te], use_month=(resid == "agesex+month"))
        lin = Ridge(alpha=1e-3).fit(ctr, Xtr)
        Xtr, Xte = Xtr - lin.predict(ctr), Xte - lin.predict(cte)
    return Xtr, Xte


SETS = {  # name -> (blocks, residualisation)
    "age only": (["age"], None),
    "all metadata": (["meta"], None),
    "speech": (["gmns", "trill"], None),
    "speech + all metadata": (["gmns", "trill", "meta"], None),
    "speech (age/sex removed)": (["gmns", "trill"], "agesex"),
    "speech (age/sex/month removed)": (["gmns", "trill"], "agesex+month"),
}
PAIRS = [("speech + all metadata", "all metadata"),
         ("speech (age/sex removed)", "age only"),
         ("speech (age/sex/month removed)", "age only")]
TASK_DEFS = {"AD/MCI vs HC": ["AD", "MCI"], "PD vs HC": ["PD"]}


def fit_predict(blocks, resid, tr, te, y):
    Xtr, Xte = design(blocks, tr, te, resid)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(Xtr, y[tr])
    return clf.predict_proba(Xte)[:, 1]


def boot_ci(rng, y, fn, n=N_BOOT):
    vals = []
    while len(vals) < n:
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(fn(i))
    return np.percentile(vals, [2.5, 97.5])


rng = np.random.default_rng(SEED)
cv_rows, pair_rows = [], []

# ------------------------------------------------------------------ A, C, D
for task, pos in TASK_DEFS.items():
    m = np.where(df.group.isin(pos + ["HC"]))[0]
    y_all = df.group.isin(pos).astype(int).to_numpy()
    y = y_all[m]
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    oof = {k: np.zeros((N_REPEATS, len(m))) for k in SETS}
    for fold, (tr_l, te_l) in enumerate(rskf.split(m, y)):
        rep = fold // N_SPLITS
        tr, te = m[tr_l], m[te_l]
        for name, (blocks, resid) in SETS.items():
            oof[name][rep, te_l] = fit_predict(blocks, resid, tr, te, y_all)
    age_m = df.age.to_numpy()[m]
    old = age_m >= 70
    mean_scores = {k: v.mean(0) for k, v in oof.items()}   # speaker score averaged over repeats
    for name in SETS:
        aucs = [roc_auc_score(y, oof[name][r]) for r in range(N_REPEATS)]
        lo, hi = boot_ci(rng, y, lambda i, s=mean_scores[name]: roc_auc_score(y[i], s[i]))
        old_ok = y[old].min() != y[old].max()
        if old_ok:
            lo70, hi70 = boot_ci(rng, y[old], lambda i, s=mean_scores[name][old], yy=y[old]: roc_auc_score(yy[i], s[i]))
            auc70 = roc_auc_score(y[old], mean_scores[name][old])
        else:
            lo70 = hi70 = auc70 = np.nan
        cv_rows.append(dict(task=task, features=name, n=len(m), n_pos=int(y.sum()),
                            auc_mean_over_repeats=round(np.mean(aucs), 3),
                            auc_sd_over_repeats=round(np.std(aucs), 3),
                            auc_boot_lo=round(lo, 3), auc_boot_hi=round(hi, 3),
                            n_70plus=int(old.sum()), n_pos_70plus=int(y[old].sum()),
                            auc_70plus=round(auc70, 3), auc_70plus_lo=round(lo70, 3),
                            auc_70plus_hi=round(hi70, 3)))
    for cand, ref in PAIRS:
        f = lambda i: roc_auc_score(y[i], mean_scores[cand][i]) - roc_auc_score(y[i], mean_scores[ref][i])
        lo, hi = boot_ci(rng, y, f)
        pair_rows.append(dict(task=task, candidate=cand, reference=ref,
                              auc_diff=round(f(np.arange(len(y))), 3), lo=round(lo, 3), hi=round(hi, 3),
                              ci_excludes_zero=bool(lo > 0 or hi < 0)))
    # paired 70+ comparison (candidate speech-age-removed vs age only)
    if y[old].min() != y[old].max():
        yo = y[old]
        for cand, ref in PAIRS[1:]:
            so_c, so_r = mean_scores[cand][old], mean_scores[ref][old]
            f = lambda i: roc_auc_score(yo[i], so_c[i]) - roc_auc_score(yo[i], so_r[i])
            lo, hi = boot_ci(rng, yo, f)
            pair_rows.append(dict(task=task + " (age 70+ only)", candidate=cand, reference=ref,
                                  auc_diff=round(f(np.arange(len(yo))), 3), lo=round(lo, 3), hi=round(hi, 3),
                                  ci_excludes_zero=bool(lo > 0 or hi < 0)))

# ------------------------------------------------------------------ B: batch predictability (healthy only)
hc = np.where(df.group == "HC")[0]
batch_rows = []
kf = KFold(5, shuffle=True, random_state=SEED)
targets = {"age (R^2)": df.age.to_numpy(), "recording date (R^2)": df.rec_days.to_numpy(float)}
for tname, tgt in targets.items():
    ok = hc[~np.isnan(tgt[hc])]
    pred = np.zeros(len(ok))
    for tr_l, te_l in kf.split(ok):
        Xtr, Xte = design(["gmns", "trill"], ok[tr_l], ok[te_l])
        pred[te_l] = Ridge(alpha=1000.0).fit(Xtr, tgt[ok[tr_l]]).predict(Xte)
    batch_rows.append(dict(target=tname, n_healthy=len(ok), score=round(r2_score(tgt[ok], pred), 3)))
y_lq = df.lowq.to_numpy()
ok = hc
pred = np.zeros(len(ok))
if y_lq[ok].min() != y_lq[ok].max():
    for tr_l, te_l in kf.split(ok):
        Xtr, Xte = design(["gmns", "trill"], ok[tr_l], ok[te_l])
        pred[te_l] = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(Xtr, y_lq[ok[tr_l]]).predict_proba(Xte)[:, 1]
    batch_rows.append(dict(target="LOW_QUALITY flag (AUC)", n_healthy=len(ok),
                           score=round(roc_auc_score(y_lq[ok], pred), 3)))
batch = pd.DataFrame(batch_rows)

# ------------------------------------------------------------------ E: temporal split
temp_rows = []
late = (rec_date >= TEMPORAL_CUTOFF).to_numpy()
for task, pos in TASK_DEFS.items():
    m = df.group.isin(pos + ["HC"]).to_numpy()
    y_all = df.group.isin(pos).astype(int).to_numpy()
    tr, te = np.where(m & ~late)[0], np.where(m & late)[0]
    n_pos_tr, n_pos_te = int(y_all[tr].sum()), int(y_all[te].sum())
    if min(n_pos_tr, n_pos_te) < 3 or y_all[te].min() == y_all[te].max():
        temp_rows.append(dict(task=task, features="(too few positives)", n_train=len(tr), n_test=len(te),
                              pos_train=n_pos_tr, pos_test=n_pos_te, auc=np.nan, lo=np.nan, hi=np.nan))
        continue
    for name in ["age only", "all metadata", "speech", "speech (age/sex removed)"]:
        blocks, resid = SETS[name]
        s = fit_predict(blocks, resid, tr, te, y_all)
        yt = y_all[te]
        lo, hi = boot_ci(rng, yt, lambda i: roc_auc_score(yt[i], s[i]))
        temp_rows.append(dict(task=task, features=name, n_train=len(tr), n_test=len(te),
                              pos_train=n_pos_tr, pos_test=n_pos_te,
                              auc=round(roc_auc_score(yt, s), 3), lo=round(lo, 3), hi=round(hi, 3)))

# ------------------------------------------------------------------ save + print
cv, pairs, temp = pd.DataFrame(cv_rows), pd.DataFrame(pair_rows), pd.DataFrame(temp_rows)
Path("reports").mkdir(exist_ok=True)
cv.to_csv("reports/confound_cv.csv", index=False)
pairs.to_csv("reports/confound_paired.csv", index=False)
batch.to_csv("reports/confound_batch_predictability.csv", index=False)
temp.to_csv("reports/confound_temporal.csv", index=False)

pd.set_option("display.width", 220)
print(f"\n##### A/C/D: {N_REPEATS}x{N_SPLITS}-fold CV over all speakers (out-of-fold AUC)")
for task, g in cv.groupby("task", sort=False):
    print(f"\n=== {task}: n={g.n.iloc[0]}, positives={g.n_pos.iloc[0]}, "
          f"aged 70+: n={g.n_70plus.iloc[0]}, positives={g.n_pos_70plus.iloc[0]}")
    print(g[["features", "auc_mean_over_repeats", "auc_sd_over_repeats", "auc_boot_lo", "auc_boot_hi",
             "auc_70plus", "auc_70plus_lo", "auc_70plus_hi"]].to_string(index=False))
print("\n##### Paired AUC differences (candidate minus reference; bootstrap over speakers)")
print(pairs.to_string(index=False))
print("\n##### B: can speech encode batch? (healthy speakers only, 5-fold CV)")
print(batch.to_string(index=False))
print(f"\n##### E: temporal split (train before {TEMPORAL_CUTOFF.date()}, test on/after)")
print(temp.to_string(index=False))
