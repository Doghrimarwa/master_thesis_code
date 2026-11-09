"""
Step 6: Apply DP-MLM (Meisenbacher et al., 2024) to the selected sentences.

For every epsilon a file  <output_dir>/df_rare_dpmlm_sigma_<tag>_eps_<eps>.csv
is written, with the rewritten text in column `dpmlm_sentence`.

* Sentences longer than MAX_CHUNK_TOKENS sub-word tokens are rewritten in chunks
  and joined again (technical safeguard for RoBERTa's 512-token limit in
  CONCAT mode; not described in the thesis text).
* CUDA crashes are recovered by rebuilding the model.
* Progress is checkpointed every --checkpoint_every sentences; re-running the
  same command resumes from the checkpoint.

Fixes 2026-11 (repository clean-up, no change to how a successful run computes):
  1. DPMLM path points to <repo>/external/DPMLM (was one folder too high).
  2. The output folder is created before the first checkpoint (before, early
     checkpoints silently failed).
  3. If a run aborts (CUDA recovery fails), the checkpoint is kept and NO final
     file is written (before, a truncated final file was written and the
     checkpoint deleted, leaving the remaining sentences un-privatized).
"""
import os
import sys
import json
import inspect
import pandas as pd
import torch
from tqdm.auto import tqdm
import wn

# path setup
DPMLM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external", "DPMLM")
DPMLM_PATH = os.path.abspath(DPMLM_PATH)
if DPMLM_PATH not in sys.path:
    sys.path.insert(0, DPMLM_PATH)

from DPMLM import DPMLM

# constants
MODEL_MAX_LENGTH   = 512   # RoBERTa hard limit
MAX_CHUNK_TOKENS   = 200   # safe per-chunk limit in CONCAT mode
CHECKPOINT_EVERY   = 5000  # save checkpoint every N sentences


# ── helpers ───────────────────────────────────────────────────────────────────

def _ensure_wordnet():
    try:
        _ = wn.Wordnet("oewn:2022")
    except Exception:
        wn.download("oewn:2022")


def _init_dpmlm(model_name, alpha, force_cuda,
                clip_min=None, clip_max=None):
    if force_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available but force_cuda=True.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        dp = DPMLM(MODEL=model_name, alpha=alpha,
                   clip_min=clip_min, clip_max=clip_max)
    except TypeError:
        dp = DPMLM(MODEL=model_name, alpha=alpha)

    dp.device = device
    for attr in ["lm_model", "raw_model", "model", "masked_lm", "mlm_model"]:
        if hasattr(dp, attr):
            m = getattr(dp, attr)
            if isinstance(m, torch.nn.Module):
                setattr(dp, attr, m.to(device))

    return dp, device


def _set_clipping_bounds(dp, clipping_bounds):
    cmin = float(clipping_bounds["C_min"])
    cmax = float(clipping_bounds["C_max"])

    if hasattr(dp, "set_clipping_bounds") and callable(
            getattr(dp, "set_clipping_bounds")):
        try:
            dp.set_clipping_bounds(cmin, cmax)
        except Exception:
            pass

    dp.clip_min    = cmin
    dp.clip_max    = cmax
    dp.C_min       = cmin
    dp.C_max       = cmax
    dp.sensitivity = abs(cmax - cmin)


def _is_cuda_dead(e):
    msg = str(e).lower()
    return "cuda error" in msg or "device-side assert" in msg


def _get_tokenizer(dp):
    for attr in ["tokenizer", "tok", "roberta_tokenizer", "bert_tokenizer"]:
        if hasattr(dp, attr):
            return getattr(dp, attr)
    return None


def _split_into_chunks(dp, s):
    """
    Split s into chunks where each chunk has at most MAX_CHUNK_TOKENS tokens.
    In CONCAT mode: 2 x MAX_CHUNK_TOKENS + 4 special tokens = 404 < 512.
    The <mask> token is therefore always present and never truncated away.
    Returns a list of strings (single element if sentence already fits).
    """
    tokenizer = _get_tokenizer(dp)
    words     = s.split()

    if tokenizer is not None:
        chunks        = []
        current_words = []

        for word in words:
            candidate = " ".join(current_words + [word])
            n_tokens  = len(tokenizer.encode(candidate,
                                             add_special_tokens=False))
            if n_tokens > MAX_CHUNK_TOKENS:
                if current_words:
                    chunks.append(" ".join(current_words))
                current_words = [word]
            else:
                current_words.append(word)

        if current_words:
            chunks.append(" ".join(current_words))

        return chunks if chunks else [s]

    else:
        # word-count heuristic fallback
        max_words = int(MAX_CHUNK_TOKENS / 1.4)
        chunks = []
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            if chunk.strip():
                chunks.append(chunk)
        return chunks if chunks else [s]


def _dpmlm_rewrite_safe(dp, s, epsilon, concat=True):
    fn     = dp.dpmlm_rewrite
    params = set(inspect.signature(fn).parameters.keys())
    kwargs = {}

    if "epsilon" in params:
        kwargs["epsilon"] = epsilon
    elif "eps" in params:
        kwargs["eps"] = epsilon

    if "TEMP" in params:
        kwargs["TEMP"] = True
    elif "temp" in params:
        kwargs["temp"] = True

    if "CONCAT" in params:
        kwargs["CONCAT"] = concat
    elif "concat" in params:
        kwargs["concat"] = concat

    if "STOP" in params:
        kwargs["STOP"] = False
    elif "stop" in params:
        kwargs["stop"] = False

    try:
        res = fn(s, **kwargs)
    except TypeError:
        clean_kw = {k: v for k, v in kwargs.items()
                    if k not in ("epsilon", "eps")}
        try:
            res = fn(s, epsilon, **clean_kw)
        except TypeError:
            res = fn(s, epsilon)

    if isinstance(res, (tuple, list)):
        new_s = res[0] if len(res) > 0 else s
        p     = res[1] if len(res) > 1 and res[1] is not None else 0
        t     = res[2] if len(res) > 2 and res[2] is not None else 0
        return str(new_s), int(p), int(t)

    if isinstance(res, dict):
        for k in ["dpmlm_sentence", "rewritten", "rewrite",
                  "sentence", "text", "output"]:
            if k in res:
                new_s = res[k]
                break
        else:
            new_s = s
        p = res.get("perturbed", res.get("n_perturbed", 0)) or 0
        t = res.get("total",     res.get("n_total",     0)) or 0
        return str(new_s), int(p), int(t)

    return str(res), 0, 0


# ── checkpoint helpers ────────────────────────────────────────────────────────

def _checkpoint_path(output_dir, prefix, epsilon):
    """Return the path to the checkpoint CSV for this epsilon."""
    return os.path.join(output_dir,
                        f"checkpoint_{prefix}_eps_{epsilon}.csv")


def _load_checkpoint(checkpoint_path):
    """
    Load an existing checkpoint.
    Returns (df_done, n_done) where df_done is the already-processed rows
    and n_done is how many sentences have been processed so far.
    Returns (None, 0) if no checkpoint exists.
    """
    if not os.path.exists(checkpoint_path):
        return None, 0

    try:
        df_done = pd.read_csv(checkpoint_path)
        n_done  = len(df_done)
        print(f"  [checkpoint] Resuming from {checkpoint_path}")
        print(f"  [checkpoint] Already processed: {n_done:,} sentences")
        return df_done, n_done
    except Exception as e:
        print(f"  [checkpoint] Could not load checkpoint: {e} — starting fresh")
        return None, 0


def _save_checkpoint(df_so_far, checkpoint_path):
    """Save current progress to the checkpoint CSV."""
    try:
        df_so_far.to_csv(checkpoint_path, index=False)
    except Exception as e:
        print(f"  [checkpoint] Warning: could not save checkpoint: {e}")


# ── main function ─────────────────────────────────────────────────────────────

def apply_dpmlm_to_rare_sentences(
    df_rare,
    clipping_bounds,
    epsilon_values=(25, 50, 75, 100),
    text_column="dpmlm_input_sentence",
    model_name="FacebookAI/roberta-base",
    alpha=0.003,
    force_cuda=False,
    metadata=None,
    output_dir=None,
    output_prefix=None,
    write_immediately=False,
    checkpoint_every=CHECKPOINT_EVERY,
):
    """
    Apply DPMLM to every sentence for each epsilon value.

    Supports checkpoint/resume: if the job is killed and restarted,
    processing continues from the last checkpoint. No work is lost.
    """
    print("=" * 70)
    print("APPLYING DPMLM")
    print("=" * 70)

    _ensure_wordnet()

    cmin = float(clipping_bounds["C_min"])
    cmax = float(clipping_bounds["C_max"])

    print("\nInitializing DPMLM...")
    print(f"  Model            : {model_name}")
    print(f"  Alpha            : {alpha}")
    print(f"  C_min            : {cmin:.6f}")
    print(f"  C_max            : {cmax:.6f}")
    print(f"  Sensitivity      : {abs(cmax - cmin):.6f}")
    print(f"  Max chunk tokens : {MAX_CHUNK_TOKENS}")
    print(f"  Checkpoint every : {checkpoint_every:,} sentences")

    dp, device = _init_dpmlm(
        model_name=model_name,
        alpha=alpha,
        force_cuda=force_cuda,
        clip_min=cmin,
        clip_max=cmax,
    )
    _set_clipping_bounds(dp, clipping_bounds)

    print(f"  Device           : {device}")
    if hasattr(dp, "sensitivity"):
        try:
            print(f"  Sensitivity (confirmed): {float(dp.sensitivity):.6f}")
        except Exception:
            pass

    results        = {}
    epsilon_values = [float(e) for e in epsilon_values]

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)   # needed for checkpoints

    for epsilon in epsilon_values:
        print(f"\n{'=' * 70}")
        print(f"Processing epsilon = {epsilon}")
        print(f"{'=' * 70}")

        prefix = f"{output_prefix}_" if output_prefix else ""

        # ── checkpoint: check if we can resume ───────────────────────────
        ckpt_path = _checkpoint_path(output_dir, prefix.rstrip("_"), epsilon) \
            if output_dir else None

        df_done, n_done = _load_checkpoint(ckpt_path) \
            if ckpt_path else (None, 0)

        # Sentences remaining to process
        df_todo    = df_rare.iloc[n_done:].reset_index(drop=True)
        sentences  = df_todo[text_column].tolist()

        print(f"  Total sentences  : {len(df_rare):,}")
        print(f"  Already done     : {n_done:,}")
        print(f"  Remaining        : {len(sentences):,}")

        # Accumulators for this run
        rewritten     = []
        perturbed     = []
        total         = []
        error_count   = 0
        error_samples = []
        n_chunked     = 0
        cuda_is_dead  = False
        cuda_resets   = 0

        for idx, s in enumerate(tqdm(sentences, desc=f"eps={epsilon}")):

            # ── Guard 1: empty / non-string ───────────────────────────────
            if not isinstance(s, str) or not s.strip():
                rewritten.append(s)
                perturbed.append(0)
                total.append(0)

            # ── Guard 2: too short to mask ────────────────────────────────
            elif len(s.split()) < 3:
                rewritten.append(s)
                perturbed.append(0)
                total.append(0)

            else:
                # ── Split into safe chunks (MAIN FIX) ────────────────────
                chunks = _split_into_chunks(dp, s)
                if len(chunks) > 1:
                    n_chunked += 1

                # ── CUDA recovery ─────────────────────────────────────────
                if cuda_is_dead:
                    try:
                        print(f"\n  [CUDA recovery] Rebuilding model "
                              f"(reset #{cuda_resets + 1})...")
                        del dp
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        dp, device = _init_dpmlm(
                            model_name=model_name,
                            alpha=alpha,
                            force_cuda=force_cuda,
                            clip_min=cmin,
                            clip_max=cmax,
                        )
                        _set_clipping_bounds(dp, clipping_bounds)
                        cuda_is_dead = False
                        cuda_resets += 1
                        print(f"  [CUDA recovery] Done.")
                    except Exception as reinit_err:
                        print(f"  [CUDA recovery] FAILED: {reinit_err}")
                        break

                # ── Rewrite each chunk, join back ─────────────────────────
                rewritten_chunks = []
                total_p          = 0
                total_t          = 0

                for cidx, chunk in enumerate(chunks):
                    try:
                        with torch.no_grad():
                            new_chunk, p, t = _dpmlm_rewrite_safe(
                                dp, chunk, epsilon, concat=True
                            )
                        rewritten_chunks.append(new_chunk)
                        total_p += p
                        total_t += t

                    except Exception as e:
                        error_count += 1
                        if len(error_samples) < 5:
                            error_samples.append(repr(e))
                        if _is_cuda_dead(e):
                            cuda_is_dead = True
                        rewritten_chunks.append(chunk)
                        if cuda_is_dead:
                            rewritten_chunks.extend(chunks[cidx + 1:])
                            break

                rewritten.append(" ".join(rewritten_chunks))
                perturbed.append(total_p)
                total.append(total_t)

            # ── Checkpoint: save progress every N sentences ───────────────
            # This is the key addition — partial results are saved regularly
            # so if the job dies nothing is lost beyond the last checkpoint.
            global_idx = n_done + idx + 1
            if ckpt_path and global_idx % checkpoint_every == 0:

                # Build partial DataFrame from what we have so far
                df_new_so_far = df_todo.iloc[:len(rewritten)].copy()
                df_new_so_far["dpmlm_sentence"]    = rewritten
                df_new_so_far["perturbed"]         = perturbed
                df_new_so_far["total"]             = total
                df_new_so_far["epsilon"]           = epsilon
                df_new_so_far["C_min"]             = cmin
                df_new_so_far["C_max"]             = cmax
                df_new_so_far["perturbation_rate"] = (
                    df_new_so_far["perturbed"]
                    / df_new_so_far["total"].replace(0, 1)
                )

                # Merge with already-done rows from previous run
                if df_done is not None:
                    df_ckpt = pd.concat(
                        [df_done, df_new_so_far], ignore_index=True
                    )
                else:
                    df_ckpt = df_new_so_far

                _save_checkpoint(df_ckpt, ckpt_path)
                print(f"\n  [checkpoint] Saved {global_idx:,} / "
                      f"{len(df_rare):,} sentences → {ckpt_path}")

        # ── Build final output DataFrame ───────────────────────────────────
        df_new = df_todo.iloc[:len(rewritten)].copy()
        df_new["dpmlm_sentence"]    = rewritten
        df_new["perturbed"]         = perturbed
        df_new["total"]             = total
        df_new["epsilon"]           = epsilon
        df_new["C_min"]             = cmin
        df_new["C_max"]             = cmax
        df_new["perturbation_rate"] = (
            df_new["perturbed"] / df_new["total"].replace(0, 1)
        )

        if isinstance(metadata, dict):
            for key, value in metadata.items():
                df_new[key] = value

        # Merge checkpoint rows with new rows
        if df_done is not None and len(df_done) > 0:
            df_out = pd.concat([df_done, df_new], ignore_index=True)
        else:
            df_out = df_new

        # Aborted run (e.g. CUDA recovery failed): keep a checkpoint, write no
        # final file, so the run can be resumed instead of silently leaving
        # sentences un-privatized.
        if len(df_out) < len(df_rare):
            print(f"\n  [abort] Only {len(df_out):,} / {len(df_rare):,} sentences "
                  f"processed for eps={epsilon}.")
            if ckpt_path:
                _save_checkpoint(df_out, ckpt_path)
                print(f"  [abort] Checkpoint kept at {ckpt_path}; "
                      f"re-run the same command to resume.")
            continue

        # ── Statistics ─────────────────────────────────────────────────────
        n_changed = (df_out["dpmlm_sentence"] != df_out[text_column]).sum()
        print("\nStatistics:")
        print(f"  Total sentences      : {len(df_out):,}")
        print(f"  Sentences chunked    : {n_chunked:,}  (split due to length)")
        print(f"  Changed sentences    : {n_changed:,}")
        print(f"  Avg perturbed tokens : {df_out['perturbed'].mean():.2f}")
        print(f"  Avg total tokens     : {df_out['total'].mean():.2f}")
        print(f"  Avg perturbation     : "
              f"{df_out['perturbation_rate'].mean() * 100:.2f}%")
        if error_count > 0:
            print(f"  DPMLM errors         : {error_count}")
            print(f"  CUDA resets          : {cuda_resets}")
            print(f"  First error samples  : {error_samples}")

        results[epsilon] = df_out

        # ── Save final output ──────────────────────────────────────────────
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(
                    output_dir,
                    f"df_rare_dpmlm_{prefix}eps_{epsilon}.csv"
                )
                df_out.to_csv(output_path, index=False)
                print(f"Saved epsilon={epsilon} to {output_path}")

                # Remove checkpoint now that the final file is written
                if ckpt_path and os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                    print(f"  [checkpoint] Removed checkpoint "
                          f"(final file written)")

            except Exception as e:
                print(f"Warning: could not save epsilon={epsilon}: {e}")

    print("\n" + "=" * 70)
    print("DPMLM APPLICATION COMPLETE")
    print("=" * 70)

    return results


# ── CLI entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Usage: python step6_apply_dpmlm.py "
            "<input_csv> <clipping_bounds_json> <output_dir> "
            "[--eps 25,50,75,100] [--text_col col] "
            "[--sigma tag] [--model name] "
            "[--checkpoint_every N]"
        )
        sys.exit(1)

    input_path  = sys.argv[1]
    bounds_path = sys.argv[2]
    output_dir  = sys.argv[3]

    eps_values        = [25.0, 50.0, 75.0, 100.0]
    text_column       = "dpmlm_input_sentence"
    sigma_tag         = "all"
    model_name        = "FacebookAI/roberta-base"
    checkpoint_every  = CHECKPOINT_EVERY

    if "--eps" in sys.argv:
        eps_values = [
            float(x)
            for x in sys.argv[sys.argv.index("--eps") + 1].split(",")
            if x
        ]
    if "--text_col" in sys.argv:
        text_column = sys.argv[sys.argv.index("--text_col") + 1]
    if "--sigma" in sys.argv:
        sigma_tag = sys.argv[sys.argv.index("--sigma") + 1]
    if "--model" in sys.argv:
        model_name = sys.argv[sys.argv.index("--model") + 1]
    if "--checkpoint_every" in sys.argv:
        checkpoint_every = int(
            sys.argv[sys.argv.index("--checkpoint_every") + 1]
        )

    df_rare = pd.read_csv(input_path)

    with open(bounds_path, "r") as f:
        clipping_bounds = json.load(f)

    results = apply_dpmlm_to_rare_sentences(
        df_rare,
        clipping_bounds,
        epsilon_values=eps_values,
        text_column=text_column,
        model_name=model_name,
        output_dir=output_dir,
        output_prefix=f"sigma_{sigma_tag}",
        write_immediately=True,
        checkpoint_every=checkpoint_every,
    )

    print(f"All per-epsilon files written to: {output_dir}")