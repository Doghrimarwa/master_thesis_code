"""
make_toy_data.py — writes a small SYNTHETIC dataset in the same format as the
MIMIC-III task files, so the pipeline can be tested without credentialed data.

All notes are invented from templates; they contain MIMIC-style placeholders
([**...**]) but no real patient information.

Writes  <out>/data/mp_50pct.csv    id, text, hospital_expire_flag
        <out>/data/proc_50pct.csv  id, text, short_codes
        <out>/outputs_proc/ALL_3_DIGIT_PRO_CODES.txt

Usage:
    python scripts/make_toy_data.py --out tests/_work --n 60
"""
import argparse
import random
from pathlib import Path

import pandas as pd

COMMON = [
    "Patient was admitted for shortness of breath and chest pain.",
    "Vital signs were stable on admission and the patient was afebrile.",
    "No tobacco, alcohol or drug use was reported by the patient.",
    "Labs on admission showed a mild leukocytosis without other abnormalities.",
    "The patient was started on intravenous antibiotics and fluids.",
    "Chest x-ray showed no acute cardiopulmonary process at this time.",
    "Patient denies fever, chills, nausea or vomiting during the stay.",
    "Physical exam was notable for mild bilateral lower extremity edema.",
    "The patient was transferred to the floor in stable condition.",
    "Medications on admission included aspirin and a beta blocker.",
]
RARE = [
    "She works as a lighthouse keeper on a remote northern island.",
    "He recently returned from a mountaineering expedition in Patagonia.",
    "Patient is a retired opera singer who lives with three parrots.",
    "Family reports he was injured while repairing a vintage locomotive.",
    "She is a professional falconer and trains birds of prey.",
    "He emigrated from Iceland and speaks four languages fluently.",
    "Patient has an unusual tattoo of a compass on the left forearm.",
    "She was found by a neighbour after a fall in her greenhouse.",
]
PHI = ["[**2158-1-3**]", "[**Name (NI) 1403**]", "[**Hospital1 18**]", "[**First Name8 (NamePattern2) 2341**]"]
CODES = ["39", "96", "38", "45", "88", "99"]


def note(rng):
    parts = [f"Admission Date: {rng.choice(PHI)} Discharge Date: {rng.choice(PHI)}",
             "HISTORY OF PRESENT ILLNESS:"]
    parts += rng.sample(COMMON, 6)
    parts.insert(rng.randint(2, len(parts)), rng.choice(RARE))
    parts.append(f"Seen by Dr. {rng.choice(PHI)} at {rng.choice(PHI)} on {rng.choice(PHI)}.")
    parts.append("Medications on discharge:\n1. aspirin 81 mg daily\n2. metoprolol 25 mg twice daily")
    return "\n".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="tests/_work")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    ids = list(range(100001, 100001 + a.n))
    texts = [note(rng) for _ in ids]
    pd.DataFrame({"id": ids, "text": texts,
                  "hospital_expire_flag": [i % 4 == 0 for i in range(a.n)]}).astype(
        {"hospital_expire_flag": int}).to_csv(out / "data/mp_50pct.csv", index=False)
    pd.DataFrame({"id": ids, "text": texts,
                  "short_codes": [",".join(sorted(rng.sample(CODES, 2))) for _ in ids]}).to_csv(
        out / "data/proc_50pct.csv", index=False)
    (out / "outputs_proc").mkdir(parents=True, exist_ok=True)
    (out / "outputs_proc/ALL_3_DIGIT_PRO_CODES.txt").write_text(" ".join(CODES))  # space-separated, as doc_classification.py reads it
    print(f"✓ {a.n} synthetic notes written to {out/'data'}")


if __name__ == "__main__":
    main()
