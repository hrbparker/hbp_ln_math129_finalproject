"""Phase 3b: does speech separate patients from controls recorded in the SAME period?

Removing recording month from the features (script 05) over-corrects, because
patients are clustered in a few months. This script instead keeps the models as
they are and changes how they are SCORED: the AUC is computed only over pairs
(one patient, one healthy control) that were recorded in the same window, so a
score that merely tracks the recording period earns no credit.

    python3 scripts/06_within_month.py        (several minutes)

Needs (all in cache/): speakers_clean.csv, splits.csv, task_features.csv, task_trill.npy

Method
  1. Out-of-fold scores from 3x5-fold stratified CV over all speakers (same models
     as script 05; fixed C=0.01; all preprocessing fitted inside each training fold).
  2. "Conditional AUC" = share of patient-control pairs, restricted to a condition,
     in which the patient scores higher (ties count 1/2).
  3. Conditions: all pairs | same calendar month | same 14-day window |
     age within 5 years | same month AND age within 5 years.
     Each is also run on speakers aged 70+ only.
  4. 95% CIs: 500 bootstrap resamples of speakers (pair weight = product of the
     two speakers' resampling counts). Paired CIs for speech minus age-only.

Reading the numbers: a condition's AUC is only meaningful with enough pairs, so
the number of pairs and the number of patients with at least one valid pair are
printed. Limits: month is a coarse proxy for session; device, site and app
version are not recorded in the released metadata.

Output: reports/within_month.csv, reports/within_month_paired.csv
"""
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED, N_BOOT, N_PCA, C = 42, 500, 32, 0.01
N_SPLITS, N_REPEATS = 5, int(os.environ.get("EWA_N_REPEATS", 3))
AGE_TOL = 5.0
TASKS = ["phonation", "pataka", "picture", "naming"]

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
MONTH = rec_date.dt.strftime("%Y-%m").to_numpy()
WINDOW = (df.rec_days // 14).to_numpy()
AGE = df.age.to_numpy(float)

feat_cols = [c for c in tf.columns if c.startswith(("gm_", "ns_"))]
G = np.hstack([tf[tf.task == t].set_index("SPEAKER_CODE")[feat_cols].reindex(codes).to_numpy(float)
               for t in TASKS])
tr_df = pd.DataFrame(trill)
tr_df["SPEAKER_CODE"] = tf.SPEAKER_CODE.to_numpy()
T = tr_df.groupby("SPEAKER_CODE").mean().reindex(codes).to_numpy(float)
D = df[["age", "male", "edu_years", "rec_days", "lowq"]].to_numpy(float)
BLOCKS = {"age": D[:, [0]], "meta": D, "gmns": G, "trill": T, "agesex": D[:, [0, 1]]}


def prep(name, tr, others):
    X = BLOCKS[name]
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X[tr])
    sc = StandardScaler().fit(imp.transform(X[tr]))
    out = [sc.transform(imp.transform(X[i])) for i in [tr] + list(others)]
    if name == "trill":
        pca = PCA(N_PCA, random_state=SEED).fit(out[0])
        out = [pca.transform(z) for z in out]
    return out


def fit_predict(blocks, resid, tr, te, y):
    parts = [prep(b, tr, [te]) for b in blocks]
    Xtr, Xte = (np.hstack([p[k] for p in parts]) for k in (0, 1))
    if resid:
        ctr, cte = prep("agesex", tr, [te])
        lin = Ridge(alpha=1e-3).fit(ctr, Xtr)
        Xtr, Xte = Xtr - lin.predict(ctr), Xte - lin.predict(cte)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(Xtr, y[tr])
    return clf.predict_proba(Xte)[:, 1]


SETS = {
    "age only": (["age"], False),
    "all metadata": (["meta"], False),
    "speech": (["gmns", "trill"], False),
    "speech + all metadata": (["gmns", "trill"] + ["meta"], False),
    "speech (age/sex removed)": (["gmns", "trill"], True),
}
CONDITIONS = ["all pairs", "same month", "same 14-day window", "age within 5y",
              "same month + age within 5y"]
TASK_DEFS = {"AD/MCI vs HC": ["AD", "MCI"], "PD vs HC": ["PD"]}
rng = np.random.default_rng(SEED)


def pair_index(pos, neg, cond, subset_ok):
    """All (patient, control) index pairs, in local indices, valid under a condition."""
    pi, nj = np.meshgrid(pos, neg, indexing="ij")
    pi, nj = pi.ravel(), nj.ravel()
    ok = subset_ok[pi] & subset_ok[nj]
    if "month" in cond:
        ok &= MONTH[pi] == MONTH[nj]
    if "14-day" in cond:
        ok &= WINDOW[pi] == WINDOW[nj]
    if "age" in cond:
        ok &= np.abs(AGE[pi] - AGE[nj]) <= AGE_TOL     # NaN ages compare False
    return pi[ok], nj[ok]


def weighted_auc(conc, w):
    return float((w * conc).sum() / w.sum()) if w.sum() > 0 else np.nan


def boot_counts(n_rows, n_boot):
    return rng.multinomial(n_rows, np.ones(n_rows) / n_rows, size=n_boot).astype(np.float64)


def boot_auc(conc, pi, nj, counts, chunk=50):
    """Bootstrap conditional AUC; pair weight = product of the two speakers' counts."""
    res = []
    for a in range(0, len(counts), chunk):
        w = counts[a:a + chunk][:, pi] * counts[a:a + chunk][:, nj]
        res.append((w @ conc) / w.sum(1))
    return np.concatenate(res)


rows, pair_rows = [], []
for task, pos_groups in TASK_DEFS.items():
    m = np.where(df.group.isin(pos_groups + ["HC"]))[0]
    y_all = df.group.isin(pos_groups).astype(int).to_numpy()
    y = y_all[m]
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    oof = {k: np.zeros((N_REPEATS, len(m))) for k in SETS}
    for fold, (tr_l, te_l) in enumerate(rskf.split(m, y)):
        for name, (blocks, resid) in SETS.items():
            oof[name][fold // N_SPLITS, te_l] = fit_predict(blocks, resid, m[tr_l], m[te_l], y_all)
    score = {k: v.mean(0) for k, v in oof.items()}          # local indexing (rows of m)

    # re-index global arrays to the local sample
    g = dict(MONTH=MONTH[m], WINDOW=WINDOW[m], AGE=AGE[m])
    MONTH_g, WINDOW_g, AGE_g = MONTH, WINDOW, AGE
    MONTH, WINDOW, AGE = g["MONTH"], g["WINDOW"], g["AGE"]
    counts = boot_counts(len(m), N_BOOT)                     # shared by all sets: paired CIs
    for subset_name, subset_ok in [("all ages", np.ones(len(m), bool)),
                                   ("aged 70+", np.nan_to_num(AGE, nan=0) >= 70)]:
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        for cond in CONDITIONS:
            pi, nj = pair_index(pos, neg, cond, subset_ok)
            if len(pi) < 200:
                continue
            conc = {k: (s[pi] > s[nj]).astype(float) + 0.5 * (s[pi] == s[nj]) for k, s in score.items()}
            n_pos_cov = len(np.unique(pi))
            est, boots = {}, {}
            for k, c in conc.items():
                est[k] = float(c.mean())
                boots[k] = boot_auc(c, pi, nj, counts)
                lo, hi = np.nanpercentile(boots[k], [2.5, 97.5])
                rows.append(dict(task=task, subset=subset_name, condition=cond, features=k,
                                 n_pairs=len(pi), n_patients_covered=n_pos_cov,
                                 n_patients=int(y[subset_ok].sum()),
                                 auc=round(est[k], 3), lo=round(lo, 3), hi=round(hi, 3)))
            for cand in ["speech", "speech + all metadata", "speech (age/sex removed)"]:
                for ref in ["age only", "all metadata"]:
                    d = boots[cand] - boots[ref]
                    lo, hi = np.nanpercentile(d, [2.5, 97.5])
                    pair_rows.append(dict(task=task, subset=subset_name, condition=cond,
                                          candidate=cand, reference=ref,
                                          auc_diff=round(est[cand] - est[ref], 3),
                                          lo=round(lo, 3), hi=round(hi, 3),
                                          ci_excludes_zero=bool(lo > 0 or hi < 0)))
    MONTH, WINDOW, AGE = MONTH_g, WINDOW_g, AGE_g           # restore global arrays

out, pairs = pd.DataFrame(rows), pd.DataFrame(pair_rows)
Path("reports").mkdir(exist_ok=True)
out.to_csv("reports/within_month.csv", index=False)
pairs.to_csv("reports/within_month_paired.csv", index=False)

pd.set_option("display.width", 230)
print(f"\n##### Conditional AUC ({N_REPEATS}x{N_SPLITS}-fold out-of-fold scores; pairs restricted by condition)")
for (task, subset), g_ in out.groupby(["task", "subset"], sort=False):
    wide = g_.pivot_table(index="condition", columns="features", values="auc", sort=False)
    meta = g_.drop_duplicates("condition").set_index("condition")[["n_pairs", "n_patients_covered", "n_patients"]]
    print(f"\n=== {task} | {subset}")
    print(pd.concat([meta, wide], axis=1).to_string())
print("\n##### Paired differences (candidate minus reference), key rows")
key = pairs[(pairs.candidate.isin(["speech", "speech (age/sex removed)"])) & (pairs.reference == "age only")]
print(key[["task", "subset", "condition", "candidate", "auc_diff", "lo", "hi", "ci_excludes_zero"]].to_string(index=False))
