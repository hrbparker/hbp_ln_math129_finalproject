"""Phase 1: audit EWA-DB metadata and build participant-level splits.

Usage (from the repo root):
    set EWA_DB_ROOT=C:\\path\\to\\EWA-DB      (Windows)
    export EWA_DB_ROOT=/path/to/EWA-DB       (Mac/Linux)
    python scripts/01_audit.py

Outputs
    cache/speakers_clean.csv   speaker-level manifest   (LOCAL ONLY, gitignored)
    cache/splits.csv           speaker -> split         (LOCAL ONLY, gitignored)
    reports/audit_counts.csv   aggregate counts only    (safe to commit)
    reports/split_fingerprint.txt  hash of the split    (safe to commit)

The EWA-DB license forbids redistributing the data, so only aggregate
numbers and a hash are meant for the public repo.
"""
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
SPLIT_FRACS = {"train": 0.65, "val": 0.15, "test": 0.20}
MAX_PLAUSIBLE_AGE = 110  # a few ages are entered as years (e.g. 2022)

ROOT = Path(os.environ.get("EWA_DB_ROOT", ""))
if not (ROOT / "SPEAKERS.TSV").exists():
    sys.exit("Set EWA_DB_ROOT to the folder containing SPEAKERS.TSV and FILES.TSV")

CACHE, REPORTS = Path("cache"), Path("reports")
CACHE.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

# ---------------------------------------------------------------- load
spk = pd.read_csv(ROOT / "SPEAKERS.TSV", sep="\t")
files = pd.read_csv(ROOT / "FILES.TSV", sep="\t",
                    usecols=["PICTURE", "SPEAKER_CODE", "DIAGNOSIS", "AUDIOFILE"])
assert spk.SPEAKER_CODE.is_unique, "duplicate speaker codes"
assert set(files.SPEAKER_CODE) <= set(spk.SPEAKER_CODE), "files for unknown speakers"

# ---------------------------------------------------------------- clean
spk["age_was_invalid"] = (spk.AGE > MAX_PLAUSIBLE_AGE) | (spk.AGE < 18)
spk["age"] = spk.AGE.where(~spk.age_was_invalid)          # NaN, not imputed here
spk["male"] = (spk.SEX == "male").astype(int)
spk["rec_date"] = pd.to_datetime(spk.RECORDED.str[:10], errors="coerce")
spk["rec_days"] = (spk.rec_date - pd.Timestamp("2021-01-01")).dt.days

# ---------------------------------------------------------------- labels
# Explicit mapping. Do not infer from folder names.
DIAG_MAP = {"Healthy": "HC", "Alzheimer": "AD", "MCI": "MCI",
            "Parkinson": "PD", "Alzheimer-Parkinson": "ADPD"}
spk["group"] = spk.DIAGNOSIS.map(DIAG_MAP)
assert spk.group.notna().all(), "unmapped diagnosis"
spk["label3"] = spk.group.map({"HC": 0, "PD": 1, "AD": 2, "MCI": 2})  # ADPD -> NaN
spk["label_admci"] = spk.group.map({"HC": 0, "AD": 1, "MCI": 1})      # others -> NaN

# audio availability per speaker, from FILES.TSV
files["has_audio"] = files.AUDIOFILE != "<not-available>"
n_audio = files.groupby("SPEAKER_CODE").has_audio.sum().rename("n_audio_files")
spk = spk.merge(n_audio, on="SPEAKER_CODE", how="left")
spk["has_audio"] = spk.PUBLISH_AGREEMENT & (spk.n_audio_files > 0)
mismatch = (spk.PUBLISH_AGREEMENT != (spk.n_audio_files > 0)).sum()
print(f"speakers where PUBLISH_AGREEMENT disagrees with FILES.TSV audio: {mismatch}")

# ---------------------------------------------------------------- splits
# Participant-level, stratified by (group x has_audio) so the audio-only
# subset keeps roughly the same proportions. Manual assignment because some
# strata are tiny (e.g. MCI with audio = 3).
rng = np.random.default_rng(SEED)
keep = spk[spk.group != "ADPD"].copy()          # 2 mixed cases excluded initially
keep["stratum"] = keep.group + "_" + keep.has_audio.map({True: "audio", False: "noaudio"})
split = {}
for _, idx in keep.groupby("stratum").groups.items():
    codes = keep.loc[idx, "SPEAKER_CODE"].to_numpy().copy()
    rng.shuffle(codes)
    n = len(codes)
    n_tr, n_va = int(round(n * SPLIT_FRACS["train"])), int(round(n * SPLIT_FRACS["val"]))
    for i, c in enumerate(codes):
        split[c] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
keep["split"] = keep.SPEAKER_CODE.map(split)

# gates: no overlap, every kept speaker assigned exactly once
assert keep.SPEAKER_CODE.is_unique and keep.split.notna().all()
parts = {k: set(keep.SPEAKER_CODE[keep.split == k]) for k in SPLIT_FRACS}
assert not (parts["train"] & parts["val"] or parts["train"] & parts["test"]
            or parts["val"] & parts["test"]), "participant overlap between splits"

# ---------------------------------------------------------------- outputs
cols = ["SPEAKER_CODE", "group", "label3", "label_admci", "age", "age_was_invalid",
        "male", "EDUCATION_YEARS", "MOCA", "WAIS", "rec_days", "LOW_QUALITY",
        "INCLUSIVE_CRITERIA", "PUBLISH_AGREEMENT", "has_audio", "n_audio_files"]
spk[cols].to_csv(CACHE / "speakers_clean.csv", index=False)
keep[["SPEAKER_CODE", "split", "stratum"]].to_csv(CACHE / "splits.csv", index=False)

counts = (keep.groupby(["split", "group", "has_audio"]).size()
          .rename("n_speakers").reset_index())
counts.to_csv(REPORTS / "audit_counts.csv", index=False)
fp = hashlib.sha256(
    "\n".join(f"{c},{s}" for c, s in sorted(split.items())).encode()).hexdigest()
(REPORTS / "split_fingerprint.txt").write_text(f"seed={SEED}\nsha256={fp}\n")

# ---------------------------------------------------------------- print summary
print("\nSpeakers by group x audio availability")
print(pd.crosstab(spk.group, spk.has_audio, margins=True))
print("\nSpeakers per split (rows) x group, audio subset only")
print(pd.crosstab(keep[keep.has_audio].split, keep[keep.has_audio].group, margins=True))
print("\nSpeakers per split x group, ALL speakers (feature-level analysis)")
print(pd.crosstab(keep.split, keep.group, margins=True))
print(f"\ninvalid ages set to NaN: {int(spk.age_was_invalid.sum())}")
print(f"split fingerprint: {fp[:16]}...")
