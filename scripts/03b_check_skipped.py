"""Which JSON files did 03_extract_features.py skip, and why?

    export EWA_DB_ROOT="$HOME/Downloads/EWA-DB-v1.0/EWA-DB"
    python3 scripts/03b_check_skipped.py

Prints the skips grouped by diagnosis group, task and reason, plus the number of
affected speakers. Takes about a minute. Nothing is written to disk.
"""
import collections
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("EWA_DB_ROOT", ""))
GROUPS = ["Alzheimer", "Alzheimer-Parkinson", "Healthy", "MCI", "Parkinson"]
TASKS = {"phonation", "pataka", "picture", "naming"}
PRACTICE = {"melon", "plave"}
if not (ROOT / "Healthy").is_dir():
    sys.exit("EWA_DB_ROOT must be the folder that contains Healthy/, Alzheimer/, ...")

by_reason = collections.Counter()
speakers = collections.defaultdict(set)
total_per_group = collections.Counter()
for g in GROUPS:
    for p in (ROOT / g).glob("*/*/*.json"):
        spk, task = p.parts[-3], p.parts[-2]
        if task not in TASKS or p.stem in PRACTICE:
            continue
        total_per_group[g] += 1
        try:
            with open(p, encoding="utf-8") as fh:
                d = json.load(fh)
            d["OpenSmile-GeMAPS"], d["NeuroSpeech"], d["TRILL-distill-embedding"]
            assert d["code"] == spk, "speaker code does not match folder"
        except Exception as e:
            reason = f"{type(e).__name__}: {str(e)[:50]}"
            by_reason[(g, task, reason)] += 1
            speakers[g].add(spk)

print(f"{'count':>6}  group / task / reason")
for (g, task, reason), n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
    print(f"{n:6d}  {g} / {task} / {reason}")
print("\nSkipped files as a share of each group's files, and speakers affected:")
for g in GROUPS:
    n = sum(v for (gg, _, _), v in by_reason.items() if gg == g)
    print(f"  {g:22s} {n:5d} of {total_per_group[g]:6d} "
          f"({100 * n / max(total_per_group[g], 1):.2f}%)   speakers affected: {len(speakers[g])}")
print(f"\nTotal skipped: {sum(by_reason.values())}")
