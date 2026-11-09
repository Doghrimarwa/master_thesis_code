"""
Step 7: Rebuild Privatized Notes
Reconstruct complete notes with DPMLM rewrites integrated
"""
import ast
import pandas as pd
from pathlib import Path


def rebuild_notes_with_dpmlm(df_dpmlm, df_original,
                              id_col="id",
                              sents_col="sentences_mimic"):
    print("Rebuilding privatized notes...")

    mp_private = df_original.copy()

    # FIX: parse string-encoded lists from CSV
    def parse_sents(x):
        if isinstance(x, list):
            return list(x)
        if isinstance(x, str):
            try:
                parsed = ast.literal_eval(x)
                return list(parsed) if isinstance(parsed, list) else [x]
            except Exception:
                return [x]
        return []

    mp_private["sentences_private"] = mp_private[sents_col].apply(parse_sents)

    # Create mapping: (note_id, sent_idx) -> rewritten sentence
    repl = {
        (int(r.note_id), int(r.sent_idx)): r.dpmlm_sentence
        for r in df_dpmlm.itertuples(index=False)
        if pd.notna(r.dpmlm_sentence)
    }

    # Map note_id -> DataFrame index label
    id_to_indexlabel = pd.Series(
        mp_private.index,
        index=mp_private[id_col].astype(int)
    ).to_dict()

    # Replace rare sentences with DPMLM rewrites
    replaced = 0
    for (nid, si), new_sent in repl.items():
        if nid not in id_to_indexlabel:
            continue
        idx_label = id_to_indexlabel[nid]
        sents = mp_private.at[idx_label, "sentences_private"]
        if isinstance(sents, list) and 0 <= si < len(sents):
            sents[si] = new_sent
            mp_private.at[idx_label, "sentences_private"] = sents
            replaced += 1

    # Join sentences into full note
    mp_private["note_private"] = mp_private["sentences_private"].apply(
        lambda s: " ".join(s) if isinstance(s, list) and len(s) > 0 else None
    )

    n_privatized = mp_private["note_private"].notna().sum()
    print(f"✓ Rebuilt {n_privatized:,} privatized notes")
    print(f"  Total sentences replaced: {replaced:,} / {len(repl):,}")

    return mp_private


def rebuild_notes_for_outputs(dpmlm_paths, df_original, output_dir,
                               id_col="id", sents_col="sentences_mimic"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for dpmlm_path in dpmlm_paths:
        print(f"\nProcessing: {Path(dpmlm_path).name}")
        df_dpmlm = pd.read_csv(dpmlm_path)
        mp_private = rebuild_notes_with_dpmlm(
            df_dpmlm, df_original, id_col=id_col, sents_col=sents_col
        )
        out_name = f"priv_notes_{Path(dpmlm_path).stem}.csv"
        out_path = output_dir / out_name
        mp_private.to_csv(out_path, index=False)
        print(f"✓ Saved → {out_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python step7_rebuild_notes.py <dpmlm_csv_or_dir> <original_csv> <output_dir>")
        sys.exit(1)

    dpmlm_input = Path(sys.argv[1])
    original_path = sys.argv[2]   # use step1_segmented.csv
    output_target = sys.argv[3]

    df_original = pd.read_csv(original_path)

    if dpmlm_input.is_dir():
        dpmlm_paths = sorted(dpmlm_input.glob("df_rare_dpmlm_*.csv"))
        if not dpmlm_paths:
            raise FileNotFoundError(f"No DPMLM outputs found in {dpmlm_input}")
        rebuild_notes_for_outputs(dpmlm_paths, df_original, output_target)
    else:
        df_dpmlm = pd.read_csv(dpmlm_input)
        mp_private = rebuild_notes_with_dpmlm(df_dpmlm, df_original)
        Path(output_target).parent.mkdir(parents=True, exist_ok=True)
        mp_private.to_csv(output_target, index=False)
        print(f"✓ Saved → {output_target}")