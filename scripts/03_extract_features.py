"""Phase 2b: collect the precomputed acoustic features from the EWA-DB JSON files.

For every speaker and task (phonation, pataka, picture, naming) this averages
the features over that task's recordings and saves:

    cache/task_features.csv   one row per (speaker, task): n_files + gm_* (GeMAPS, 62)
                              + ns_* (NeuroSpeech, 28)
    cache/task_trill.npy      float32 array (rows aligned with task_features.csv, 2048)

IMPORTANT: the "info" block inside each JSON contains the speaker's age and
DIAGNOSIS. It is never read here. Labels and demographics come only from
cache/speakers_clean.csv (made by 01_audit.py).

Usage:
    export EWA_DB_ROOT="$HOME/Downloads/EWA-DB-v1.0/EWA-DB"
    python3 scripts/03_extract_features.py
Takes a few minutes (about 113k files). All outputs stay in cache/ (not committed).
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("EWA_DB_ROOT", ""))
GROUPS = ["Alzheimer", "Alzheimer-Parkinson", "Healthy", "MCI", "Parkinson"]
TASKS = ["phonation", "pataka", "picture", "naming"]
PRACTICE_ITEMS = {"melon", "plave"}          # practice naming items, per README
if not all((ROOT / g).is_dir() for g in ("Healthy", "Alzheimer")):
    sys.exit("EWA_DB_ROOT must be the folder that contains Healthy/, Alzheimer/, ...")

Path("cache").mkdir(exist_ok=True)
warnings.filterwarnings("ignore", message="Mean of empty slice")


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


gm_keys = ns_keys = None
acc = {}              # (speaker, task) -> lists of vectors
n_seen = n_bad = 0
t0 = time.time()

for group in GROUPS:
    for path in sorted((ROOT / group).glob("*/*/*.json")):
        speaker, task = path.parts[-3], path.parts[-2]
        if task not in TASKS or path.stem in PRACTICE_ITEMS:
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
            gm, ns, tr = d["OpenSmile-GeMAPS"], d["NeuroSpeech"], d["TRILL-distill-embedding"]
            assert d["code"] == speaker, "speaker code does not match folder"
        except Exception as e:                       # keep going, but count and report
            n_bad += 1
            if n_bad <= 5:
                print(f"skipped {path}: {e}")
            continue
        if gm_keys is None:
            gm_keys, ns_keys = sorted(gm), sorted(ns)
        vec = np.array([to_float(gm.get(k)) for k in gm_keys]
                       + [to_float(ns.get(k)) for k in ns_keys], dtype=np.float64)
        acc.setdefault((speaker, task), []).append((vec, np.asarray(tr, dtype=np.float32)))
        n_seen += 1
        if n_seen % 10000 == 0:
            print(f"  {n_seen} files read ({time.time() - t0:.0f}s)")

rows, trills = [], []
for (speaker, task), items in sorted(acc.items()):
    vecs = np.vstack([v for v, _ in items])
    mean = np.nanmean(vecs, axis=0) if np.isfinite(vecs).any() else np.full(vecs.shape[1], np.nan)
    rows.append([speaker, task, len(items)] + mean.tolist())
    trills.append(np.mean(np.vstack([t for _, t in items]), axis=0))

cols = (["SPEAKER_CODE", "task", "n_files"]
        + [f"gm_{k}" for k in gm_keys] + [f"ns_{k}" for k in ns_keys])
out = pd.DataFrame(rows, columns=cols)
out.to_csv("cache/task_features.csv", index=False)
np.save("cache/task_trill.npy", np.vstack(trills).astype(np.float32))

print(f"\nfiles read: {n_seen}   skipped: {n_bad}   time: {time.time() - t0:.0f}s")
print(f"rows (speaker x task): {len(out)}   speakers: {out.SPEAKER_CODE.nunique()}")
print(out.groupby("task").SPEAKER_CODE.nunique().to_string())
print(f"features: {len(gm_keys)} GeMAPS + {len(ns_keys)} NeuroSpeech + 2048 Trill")
print(f"fraction of missing (NaN) feature values: {out.iloc[:, 3:].isna().mean().mean():.4f}")
