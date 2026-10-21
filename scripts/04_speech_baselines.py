"""Phase 2c: do speech features carry diagnostic information beyond age?

Needs (all in cache/): speakers_clean.csv, splits.csv   (from 01_audit.py)
                       task_features.csv, task_trill.npy (from 03_extract_features.py)

    python3 scripts/04_speech_baselines.py

Protocol (same splits as 02):
  * Fit on train speakers. The VALIDATION split picks the logistic-regression
    regularisation strength C and the 90%-specificity threshold.
    The TEST split is only scored, never used for any choice.
  * All preprocessing (median imputation, scaling, PCA on Trill, age/sex
    residualisation) is fitted on train only.
  * 95% CIs: 1000 bootstrap resamples of test speakers.

Feature sets
  age only / age+sex+edu / all metadata     reference baselines (see 02)
  speech: GeMAPS+NeuroSpeech                 90 features x 4 tasks
  speech: Trill                              2048-d embedding (mean over tasks) -> PCA 32
  speech: all
  speech + age/sex/edu                       does speech ADD to demographics?
  speech + all metadata                      ... and to demographics + recording date?
  speech (age/sex removed)                   speech features with the linear effect of
                                             age and sex regressed out first
  task: phonation / pataka / picture / naming   GeMAPS+NeuroSpeech from one task only

Extra checks: AUC within speakers aged 70+ only (so age cannot separate classes
as easily), and paired bootstrap of AUC differences on the SAME test speakers.

Outputs: reports/speech_baselines.csv, reports/speech_vs_age_paired.csv
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED, N_BOOT, N_PCA = 42, 1000, 32
C_GRID = [0.001, 0.01, 0.1, 1.0]
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

feat_cols = [c for c in tf.columns if c.startswith(("gm_", "ns_"))]
wide = []
for t in TASKS:
    sub = tf[tf.task == t].set_index("SPEAKER_CODE")[feat_cols].add_prefix(f"{t}__")
    wide.append(sub.reindex(codes).reset_index(drop=True))
G_task = {t: w.to_numpy(float) for t, w in zip(TASKS, wide)}
G_all = np.hstack([G_task[t] for t in TASKS])

tr_df = pd.DataFrame(trill)
tr_df["SPEAKER_CODE"] = tf.SPEAKER_CODE.to_numpy()
T_all = tr_df.groupby("SPEAKER_CODE").mean().reindex(codes).to_numpy(float)

DEMO_COLS = ["age", "male", "edu_years", "rec_days", "lowq"]
D = df[DEMO_COLS].to_numpy(float)
BLOCKS = {
    "age": D[:, [0]], "agesexedu": D[:, [0, 1, 2]], "meta": D,
    "gmns": G_all, "trill": T_all,
    **{f"gmns_{t}": G_task[t] for t in TASKS},
}

# name -> (blocks, residualise speech on age+sex)
SETS = {
    "age only": (["age"], False),
    "age+sex+edu": (["agesexedu"], False),
    "all metadata": (["meta"], False),
    "speech: GeMAPS+NeuroSpeech": (["gmns"], False),
    "speech: Trill": (["trill"], False),
    "speech: all": (["gmns", "trill"], False),
    "speech + age/sex/edu": (["gmns", "trill", "agesexedu"], False),
    "speech + all metadata": (["gmns", "trill", "meta"], False),
    "speech (age/sex removed)": (["gmns", "trill"], True),
    **{f"task: {t}": ([f"gmns_{t}"], False) for t in TASKS},
}
PAIRS = [  # (candidate, reference): does the candidate beat the reference?
    ("speech + age/sex/edu", "age+sex+edu"),
    ("speech + all metadata", "all metadata"),
]


# ------------------------------------------------------------------ model
def prep_block(name, tr, others):
    """Impute + scale (+PCA for Trill) fitted on train rows only."""
    X = BLOCKS[name]
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X[tr])
    sc = StandardScaler().fit(imp.transform(X[tr]))
    f = lambda idx: sc.transform(imp.transform(X[idx]))
    out = [f(tr)] + [f(o) for o in others]
    if name == "trill":
        pca = PCA(N_PCA, random_state=SEED).fit(out[0])
        out = [pca.transform(z) for z in out]
    return out


def fit_scores(tr, va, te, y, blocks, resid):
    """Return validation and test scores for one feature set."""
    parts = [prep_block(b, tr, [va, te]) for b in blocks]            # per block: [tr, va, te]
    if resid:
        demo = prep_block("agesexedu", tr, [va, te])                  # age, male, edu (scaled)
        cov = [d[:, :2] for d in demo]                                # age + sex only
        lin = LinearRegression().fit(cov[0], np.hstack([p[0] for p in parts]))
        stacked = [np.hstack([p[i] for p in parts]) - lin.predict(cov[i]) for i in range(3)]
    else:
        stacked = [np.hstack([p[i] for p in parts]) for i in range(3)]
    best = None
    for C in C_GRID:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
        clf.fit(stacked[0], y[tr])
        s_va = clf.predict_proba(stacked[1])[:, 1]
        auc_va = roc_auc_score(y[va], s_va)
        if best is None or auc_va > best[0]:
            best = (auc_va, C, s_va, clf.predict_proba(stacked[2])[:, 1])
    return best  # (val_auc, C, val_scores, test_scores)


def boot_idx(rng, y):
    while True:
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            return i


# ------------------------------------------------------------------ run
rows, pair_rows = [], []
rng = np.random.default_rng(SEED)
TASK_DEFS = {"AD/MCI vs HC": ["AD", "MCI"], "PD vs HC": ["PD"]}

for sub_name, sub_mask in [("all speakers", np.ones(len(df), bool)), ("audio subset", df.has_audio.to_numpy())]:
    for task, pos in TASK_DEFS.items():
        m = sub_mask & df.group.isin(pos + ["HC"]).to_numpy()
        y = df.group.isin(pos).astype(int).to_numpy()
        tr, va, te = (np.where(m & (df.split == s).to_numpy())[0] for s in ("train", "val", "test"))
        y_te, age_te = y[te], df.age.to_numpy()[te]
        old = age_te >= 70
        scores = {}
        for name, (blocks, resid) in SETS.items():
            auc_va, C, s_va, s_te = fit_scores(tr, va, te, y, blocks, resid)
            scores[name] = s_te
            thr = np.quantile(s_va[y[va] == 0], 0.90)
            pred = s_te > thr
            boots = [roc_auc_score(y_te[i], s_te[i]) for i in (boot_idx(rng, y_te) for _ in range(N_BOOT))]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            auc70 = roc_auc_score(y_te[old], s_te[old]) if len(set(y_te[old])) == 2 else np.nan
            rows.append(dict(subset=sub_name, task=task, features=name, C=C,
                             n_test=len(te), n_test_pos=int(y_te.sum()),
                             auc=round(roc_auc_score(y_te, s_te), 3), auc_lo=round(lo, 3), auc_hi=round(hi, 3),
                             auc_age70plus=round(auc70, 3), n_pos_70plus=int(y_te[old].sum()),
                             avg_precision=round(average_precision_score(y_te, s_te), 3),
                             sens_at_val90spec=round(pred[y_te == 1].mean(), 3),
                             test_specificity=round((~pred[y_te == 0]).mean(), 3)))
        for cand, ref in PAIRS:
            d = []
            for _ in range(N_BOOT):
                i = boot_idx(rng, y_te)
                d.append(roc_auc_score(y_te[i], scores[cand][i]) - roc_auc_score(y_te[i], scores[ref][i]))
            lo, hi = np.percentile(d, [2.5, 97.5])
            pair_rows.append(dict(subset=sub_name, task=task, candidate=cand, reference=ref,
                                  auc_diff=round(roc_auc_score(y_te, scores[cand]) - roc_auc_score(y_te, scores[ref]), 3),
                                  diff_lo=round(lo, 3), diff_hi=round(hi, 3),
                                  ci_excludes_zero=bool(lo > 0 or hi < 0)))

out, pairs = pd.DataFrame(rows), pd.DataFrame(pair_rows)
Path("reports").mkdir(exist_ok=True)
out.to_csv("reports/speech_baselines.csv", index=False)
pairs.to_csv("reports/speech_vs_age_paired.csv", index=False)

pd.set_option("display.width", 220)
show = ["features", "auc", "auc_lo", "auc_hi", "auc_age70plus", "avg_precision", "sens_at_val90spec", "test_specificity"]
for (sub, task), g in out.groupby(["subset", "task"], sort=False):
    print(f"\n=== {sub} | {task} | test n={g.n_test.iloc[0]}, positives={g.n_test_pos.iloc[0]}, "
          f"positives aged 70+: {g.n_pos_70plus.iloc[0]}")
    print(g[show].to_string(index=False))
print("\n=== Paired AUC differences (candidate minus reference, same test speakers)")
print(pairs.to_string(index=False))
