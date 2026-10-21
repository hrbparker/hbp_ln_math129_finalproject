"""Phase 2a: how well do age / sex / education / recording date alone predict
diagnosis? Any speech model has to beat these numbers to be interesting.

Run after 01_audit.py (reads cache/speakers_clean.csv and cache/splits.csv):
    python3 scripts/02_metadata_baselines.py

Protocol: fit on train speakers, report on held-out test speakers.
The validation split picks the decision threshold (90% specificity on
validation healthy controls); test data never influences any choice.
Confidence intervals: 1000 bootstrap resamples of test SPEAKERS.

Output: reports/baseline_metadata.csv (aggregate numbers, safe to commit)
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED, N_BOOT = 42, 1000

spk = pd.read_csv("cache/speakers_clean.csv")
spl = pd.read_csv("cache/splits.csv")[["SPEAKER_CODE", "split"]]
df = spk.merge(spl, on="SPEAKER_CODE", how="inner")
df["lowq"] = df.LOW_QUALITY.astype(int)
df = df.rename(columns={"EDUCATION_YEARS": "edu_years"})

FEATURE_SETS = {
    "age only": ["age"],
    "age+sex+edu": ["age", "male", "edu_years"],
    "recording date only": ["rec_days"],
    "all metadata": ["age", "male", "edu_years", "rec_days", "lowq"],
}
MODELS = {
    "logreg": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                    LogisticRegression(max_iter=1000)),
    "gboost": lambda: make_pipeline(SimpleImputer(strategy="median"),
                                    GradientBoostingClassifier(n_estimators=100, max_depth=2,
                                                               random_state=SEED)),
}
TASKS = {  # name -> positive groups
    "AD/MCI vs HC": ["AD", "MCI"],
    "PD vs HC": ["PD"],
}


def boot_ci(y, p, fn, rng):
    vals = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(y[i], p[i]))
    return np.percentile(vals, [2.5, 97.5])


rows = []
rng = np.random.default_rng(SEED)
for subset_name, subset in [("all speakers", df), ("audio subset", df[df.has_audio])]:
    for task, pos in TASKS.items():
        d = subset[subset.group.isin(pos + ["HC"])].copy()
        d["y"] = d.group.isin(pos).astype(int)
        tr, va, te = (d[d.split == s] for s in ("train", "val", "test"))
        n_pos_te = int(te.y.sum())
        for fs_name, feats in FEATURE_SETS.items():
            for m_name, make in MODELS.items():
                model = make().fit(tr[feats], tr.y)
                p_va = model.predict_proba(va[feats])[:, 1]
                p_te = model.predict_proba(te[feats])[:, 1]
                y_te = te.y.to_numpy()
                # threshold = 90th percentile of validation healthy scores
                thr = np.quantile(p_va[va.y.to_numpy() == 0], 0.90)
                pred = p_te > thr
                sens = pred[y_te == 1].mean()
                spec = (~pred[y_te == 0]).mean()
                auc = roc_auc_score(y_te, p_te)
                ap = average_precision_score(y_te, p_te)
                lo, hi = boot_ci(y_te, p_te, roc_auc_score, rng)
                rows.append(dict(subset=subset_name, task=task, features=fs_name,
                                 model=m_name, n_test=len(y_te), n_test_pos=n_pos_te,
                                 test_prevalence=round(y_te.mean(), 3),
                                 auc=round(auc, 3), auc_lo=round(lo, 3), auc_hi=round(hi, 3),
                                 avg_precision=round(ap, 3),
                                 sens_at_val90spec=round(sens, 3),
                                 test_specificity=round(spec, 3)))

out = pd.DataFrame(rows)
Path("reports").mkdir(exist_ok=True)
out.to_csv("reports/baseline_metadata.csv", index=False)

pd.set_option("display.width", 200)
for (sub, task), g in out.groupby(["subset", "task"]):
    print(f"\n=== {sub} | {task} | test n={g.n_test.iloc[0]}, positives={g.n_test_pos.iloc[0]}"
          f" (prevalence {g.test_prevalence.iloc[0]}; AP of random guessing = prevalence)")
    print(g[["features", "model", "auc", "auc_lo", "auc_hi", "avg_precision",
             "sens_at_val90spec", "test_specificity"]].to_string(index=False))
