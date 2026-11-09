"""
Patched DPMLM.py (Meisenbacher et al., 2024 — https://github.com/sjmeis/DPMLM)

Recovered from chat history (2026-02, "DPMLM attribute error on MIMIC III dataset").
Copy this file over external/DPMLM/DPMLM.py after cloning the upstream repo
(scripts/setup_external.sh does this). The upstream data/ folder
(opposites.json, vocab.txt) is still required.

Changes vs. upstream:
  FIX 1  self.MODEL = MODEL stored (upstream raised AttributeError)
  FIX 2  clip_min / clip_max constructor arguments (MIMIC-III bounds from step 5)
  FIX 3  sensitivity computed after the final clip bounds are set
  FIX 4  _encode_safe(): <mask> token always survives truncation
  FIX 5  privatize_batch(): if mask logits are empty, return the original token

NOTE: the version later used on the cluster may have kept the upstream TEMP/POS
keyword arguments in dpmlm_rewrite(). step6_apply_dpmlm.py inspects the
signature at runtime and only passes arguments that exist, so it works with
either variant. Long sentences are additionally chunked in step 6.
"""
import os
import torch
import nltk
import json
import string
import numpy as np
from collections import Counter
import wn
nltk.download('words', quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("words", quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, logging
import importlib_resources as impresources
import gc

torch.set_float32_matmul_precision('medium')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.set_verbosity_warning()

stop = set([x for x in stopwords.words("english")])


def nth_repl(s, sub, repl, n):
    s_split = nltk.word_tokenize(s)
    i = 0
    try:
        find = s_split.index(sub)
        i += 1
    except ValueError:
        return s

    while i != n:
        try:
            find = s_split.index(sub, find + 1)
            i += 1
        except ValueError:
            break
    if i == n:
        return " ".join(s_split[:find] + [repl] + s_split[find+1:])
    return s


def nth_rem(s, sub, n):
    s_split = s.split()
    i = 0
    try:
        find = s_split.index(sub)
        i += 1
    except ValueError:
        return s

    while i != n:
        try:
            find = s_split.index(sub, find + 1)
            i += 1
        except ValueError:
            break
    if i == n:
        return " ".join(s_split[:find] + s_split[find+1:])
    return s


def sentence_enum(tokens):
    counts = Counter()
    n = []
    for t in tokens:
        counts[t] += 1
        n.append(counts[t])
    return n


def _data_dir():
    # Works whether DPMLM is imported as a module (DPMLM.py next to data/)
    # or as a package (DPMLM/__init__.py with DPMLM/data/).
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.isdir(os.path.join(here, "data")):
        return os.path.join(here, "data")
    return str(impresources.files("DPMLM") / "data")


def get_opposites():
    with open(os.path.join(_data_dir(), "opposites.json"), 'r') as f:
        opposites = json.load(f)
    return opposites


def get_vocab():
    with open(os.path.join(_data_dir(), "vocab.txt"), 'r') as f:
        vocab = set([x.strip() for x in f.readlines()])
    return vocab


class DPMLM():
    opposites = get_opposites()
    vocab = get_vocab()
    lemmatizer = WordNetLemmatizer()
    detokenizer = TreebankWordDetokenizer()
    tokenizer = None
    lm_model = None
    raw_model = None
    device = None
    nlp = None
    alpha = None

    def __init__(self, MODEL="FacebookAI/roberta-base", SPACY="en_core_web_md", alpha=0.003,
                 clip_min=None, clip_max=None):
        # FIX 1: store MODEL as instance attribute (prevents AttributeError)
        self.MODEL = MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.lm_model = AutoModelForMaskedLM.from_pretrained(MODEL)
        self.raw_model = AutoModel.from_pretrained(MODEL, output_hidden_states=True, output_attentions=True)
        self.alpha = alpha

        # FIX 2: accept custom clip bounds so sensitivity is correct from the start
        if clip_min is not None and clip_max is not None:
            self.clip_min = clip_min
            self.clip_max = clip_max
        else:
            # Default values (empirically derived from roberta-base on SST2)
            self.clip_min = -3.2093127
            self.clip_max = 16.304797887802124

        # FIX 3: sensitivity computed once here — never stale
        self.sensitivity = abs(self.clip_max - self.clip_min)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.lm_model = self.lm_model.to(self.device)
        self.raw_model = self.raw_model.to(self.device)

    def load_transformers(self):
        return self.tokenizer, self.lm_model, self.raw_model

    def _encode_safe(self, original_sent, masked_sent, CONCAT):
        """
        FIX 4: Safe encoding that guarantees the <mask> token survives truncation.

        Strategy:
          - If CONCAT=True:  try [original][SEP][masked] with truncation='only_first'.
                             If <mask> is still cut off, fall back to CONCAT=False.
          - If CONCAT=False: encode masked_sent only, truncate from the right.
                             If <mask> is still missing (extremely rare), hard-slice.

        Returns (input_ids, mode_used) where mode_used is 'concat' or 'single'.
        """
        mask_id = self.tokenizer.mask_token_id
        max_len = self.tokenizer.model_max_length - 32  # safety margin

        if CONCAT:
            input_ids = self.tokenizer.encode(
                " " + original_sent.replace("MASK", ""),
                " " + masked_sent,
                add_special_tokens=True,
                truncation="only_first",
                max_length=max_len,
            )
            if mask_id in input_ids:
                return input_ids, "concat"
            # <mask> was truncated away -> fall through to single mode

        input_ids = self.tokenizer.encode(
            " " + masked_sent,
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
        )
        if mask_id in input_ids:
            return input_ids, "single"

        # Last resort: mask is beyond max_length even in single mode.
        full_ids = self.tokenizer.encode(" " + masked_sent, add_special_tokens=False)
        try:
            mask_pos_full = full_ids.index(mask_id)
        except ValueError:
            return None, None  # mask token absent from sentence — caller handles

        half = max_len // 2
        lo = max(0, mask_pos_full - half)
        hi = min(len(full_ids), mask_pos_full + half)
        sliced = full_ids[lo:hi]
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        if cls_id is not None and sep_id is not None:
            sliced = [cls_id] + sliced + [sep_id]
        if mask_id in sliced:
            return sliced, "single"

        return None, None

    def privatize(self, sentence, target, n, start_index, CONCAT=True, epsilon=1):
        split_sent = nltk.word_tokenize(sentence)
        original_sent = ' '.join(split_sent)

        # Mask the target word
        masked_sent = ' '.join(split_sent)
        masked_sent = nth_repl(masked_sent, target, self.tokenizer.mask_token, n)
        n = [n]

        # Sliding-window on the masked sentence to stay within model limits
        encoded = self.tokenizer.encode(masked_sent, add_special_tokens=False)
        lower, upper = self.sliding_window(
            encoded, start_index,
            int((self.tokenizer.model_max_length - 32) / 2)
        )
        masked_sent = self.tokenizer.decode(encoded[lower:upper], skip_special_tokens=False)

        # FIX 4 applied: safe encoding with automatic fallback
        input_ids, mode = self._encode_safe(original_sent, masked_sent, CONCAT)

        if input_ids is None:
            # Absolute fallback: return original token unchanged
            return {"{}_{}".format(target[0] if isinstance(target, list) else target,
                                   n[0]): (target[0] if isinstance(target, list) else target)}

        mask_id = self.tokenizer.mask_token_id
        masked_position = [input_ids.index(mask_id)]
        target = [target]

        with torch.no_grad():
            output = self.lm_model(
                torch.tensor(input_ids).reshape(1, len(input_ids)).to(self.device)
            )

        logits = output[0].squeeze().detach().cpu().numpy()

        predictions = {}
        for t, m, nn in zip(target, masked_position, n):
            current = "{}_{}".format(t, nn)
            mask_logits = logits[m].squeeze()
            # Exponential mechanism: clip -> temperature 2*Δu/ε -> softmax -> sample
            mask_logits = np.clip(mask_logits, self.clip_min, self.clip_max)
            mask_logits = mask_logits / (2 * self.sensitivity / epsilon)

            logits_idx = [i for i, x in enumerate(mask_logits)]
            scores = torch.softmax(torch.from_numpy(mask_logits), dim=0)
            scores = scores / scores.sum()
            chosen_idx = np.random.choice(logits_idx, p=scores.numpy())
            predictions[current] = (self.tokenizer.decode(chosen_idx).strip(), scores[chosen_idx])

        for p in predictions:
            predictions[p] = predictions[p][0]

        return predictions

    def privatize_batch(self, sentences, targets, n, epsilon, CONCAT=True, STOP=False, batch_size=16):
        predictions = {}
        outputs = []
        masked_position = []

        new_sentences = []
        new_targets = []
        new_n = []
        new_epsilon = []
        target_indices = []
        for i, (t, s, x, e) in enumerate(zip(targets, sentences, n, epsilon)):
            if (STOP == False and t in stop) or t in string.punctuation:
                predictions["{}_{}".format(t, x)] = t
                continue
            new_sentences.append(s)
            new_targets.append(t)
            new_n.append(x)
            new_epsilon.append(e)
            target_indices.append(i)
        sentences = new_sentences.copy()
        targets = new_targets.copy()
        n = new_n.copy()
        epsilon = new_epsilon.copy()

        split_size = int(np.ceil(len(sentences) / batch_size))

        start_index = 0
        begin = 0
        end = batch_size
        for idx in range(split_size):
            if end is not None:
                batch = sentences[begin:end]
                targets_batch = targets[begin:end]
                n_batch = n[begin:end]
            else:
                batch = sentences[begin:]
                targets_batch = targets[begin:]
                n_batch = n[begin:]

            begin += batch_size
            if idx == split_size - 2 or idx == split_size - 1:
                end = None
            else:
                end += batch_size

            split_sents = [nltk.word_tokenize(sentence) for sentence in batch]
            original_sents = [' '.join(split_sent) for split_sent in split_sents]

            masked_sents = [' '.join(split_sent) for split_sent in split_sents]

            for i, (t, nn) in enumerate(zip(targets_batch, n_batch)):
                temp = nth_repl(masked_sents[i], t, self.tokenizer.mask_token, nn)
                encoded = self.tokenizer.encode(temp, add_special_tokens=False)
                lower, upper = self.sliding_window(
                    encoded, target_indices[start_index],
                    int((self.tokenizer.model_max_length - 32) / 2)
                )
                masked_sent = self.tokenizer.decode(encoded[lower:upper], skip_special_tokens=False)
                masked_sents[i] = masked_sent
                start_index += 1

            max_len = self.tokenizer.model_max_length - 32
            mask_id = self.tokenizer.mask_token_id

            # FIX 4 (batch version): per-item safe encoding with fallback
            all_input_ids = []
            batch_masked_positions = []

            for orig, masked in zip(original_sents, masked_sents):
                ids, _ = self._encode_safe(orig, masked, CONCAT)
                if ids is None:
                    ids = self.tokenizer.encode(
                        " " + masked, truncation=True, max_length=max_len
                    )
                all_input_ids.append(ids)
                pos_flags = [tok == mask_id for tok in ids]
                batch_masked_positions.append(pos_flags)

            # Pad to same length for batched inference
            max_batch_len = max(len(ids) for ids in all_input_ids)
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

            padded_ids = []
            attention_masks = []
            for ids in all_input_ids:
                pad_len = max_batch_len - len(ids)
                padded_ids.append(ids + [pad_id] * pad_len)
                attention_masks.append([1] * len(ids) + [0] * pad_len)

            inputs = {
                "input_ids": torch.tensor(padded_ids).to(self.device),
                "attention_mask": torch.tensor(attention_masks).to(self.device),
            }

            # pad the mask-position flags too so they index the padded logits
            for flags in batch_masked_positions:
                flags.extend([False] * (max_batch_len - len(flags)))
            masked_position.extend(batch_masked_positions)

            with torch.no_grad():
                outputs.extend(self.lm_model(**inputs).logits)

            del inputs

        for i in range(len(outputs)):
            current = "{}_{}".format(targets[i], n[i])

            mask_logits = outputs[i][masked_position[i]].squeeze().detach().cpu().numpy()
            if len(mask_logits) == 0:
                # FIX 5: mask was lost — return original token instead of crashing
                predictions[current] = targets[i]
                continue
            mask_logits = np.clip(mask_logits, self.clip_min, self.clip_max)
            mask_logits = mask_logits / (2 * self.sensitivity / epsilon[i])

            logits_idx = [j for j, x in enumerate(mask_logits)]
            scores = torch.softmax(torch.from_numpy(mask_logits), dim=0)
            scores = scores / scores.sum()
            chosen_idx = np.random.choice(logits_idx, p=scores.numpy())
            predictions[current] = self.tokenizer.decode(chosen_idx).strip()

        del outputs
        with torch.no_grad():
            torch.cuda.empty_cache()
        gc.collect()

        return predictions

    def sliding_window(self, tokens, target_idx, max_len):
        length = len(tokens)
        lower = max(0, int(target_idx - (max_len / 2)))
        remaining = max_len - (target_idx - lower)
        upper = min(length, target_idx + remaining)
        return lower, upper

    def dpmlm_rewrite(self, sentence, epsilon, REPLACE=False, STOP=False, CONCAT=True):
        sentence = " ".join(sentence.split("\n"))
        tokens = nltk.word_tokenize(sentence)
        encoded = self.tokenizer.encode(sentence, add_special_tokens=False)

        if isinstance(epsilon, list):
            word_eps = epsilon
        else:
            word_eps = [epsilon for _ in range(len(tokens))]
        n = sentence_enum(tokens)
        replace = []
        new_tokens = [str(x) for x in tokens]

        perturbed = 0
        total = 0
        for i, (t, nn, eps) in enumerate(zip(tokens, n, word_eps)):
            if i >= len(tokens):
                break

            if (STOP == False and t in stop) or t in string.punctuation:
                total += 1
                if tokens[i][0].isupper():
                    replace.append(t.capitalize())
                else:
                    replace.append(t)
                continue

            if REPLACE:
                lower, upper = self.sliding_window(
                    new_tokens, i,
                    int((self.tokenizer.model_max_length - 32) / 2)
                )
                new_n = sentence_enum(new_tokens[lower:upper])
                t_sentence = self.tokenizer.decode(encoded[lower:upper], skip_special_tokens=True)
                res = self.privatize(t_sentence, t, n=new_n[i], start_index=i,
                                     epsilon=eps, CONCAT=CONCAT)
                r = res[t + "_{}".format(new_n[i])]
                new_tokens[i] = r
            else:
                res = self.privatize(sentence, t, nn, i, epsilon=eps, CONCAT=CONCAT)
                r = res[t + "_{}".format(nn)]

            if tokens[i][0].isupper():
                replace.append(r.capitalize())
            else:
                replace.append(r.lower())

            if r != t:
                perturbed += 1
            total += 1

        return self.detokenizer.detokenize(replace), perturbed, total

    def dpmlm_rewrite_batch(self, sentence, epsilon, REPLACE=False, STOP=False, CONCAT=True, batch_size=16):
        sentence = " ".join(sentence.split("\n"))
        tokens = nltk.word_tokenize(sentence)
        tokens_final = tokens.copy()

        if isinstance(epsilon, list):
            word_eps = epsilon
        else:
            word_eps = [epsilon for _ in range(len(tokens))]

        n = sentence_enum(tokens)
        n_final = n.copy()
        batch = [sentence] * len(tokens)
        res = self.privatize_batch(
            batch, tokens, n=n, epsilon=word_eps,
            CONCAT=CONCAT, STOP=STOP, batch_size=batch_size
        )

        replace = []
        for i, (t, x) in enumerate(zip(tokens_final, n_final)):
            r = "{}_{}".format(t, x)
            if tokens[i][0].isupper():
                replace.append(res[r].capitalize())
            else:
                replace.append(res[r].lower())

        return self.detokenizer.detokenize(replace)

    def dpmlm_rewrite_plus(self, sentence, epsilon, CONCAT=True, ADD_PROB=0.15, DEL_PROB=0.05):
        """Variable-length rewriting (Algorithm 3 from the paper)."""
        if isinstance(sentence, list):
            tokens = sentence
        else:
            tokens = nltk.word_tokenize(sentence)

        if isinstance(epsilon, list):
            word_eps = epsilon
        else:
            word_eps = [epsilon for _ in range(len(tokens))]

        n = sentence_enum(tokens)
        replace = []
        new_tokens = [str(x) for x in tokens]

        perturbed = 0
        total = 0
        deleted = 0
        added = 0

        for i, (t, nn, eps) in enumerate(zip(tokens, n, word_eps)):
            if t in string.punctuation:
                total += 1
                replace.append(t)
                continue

            if i == len(tokens) - 1:
                DELETE = 1.0
            else:
                DELETE = np.random.rand()

            if DELETE >= DEL_PROB:
                new_n = sentence_enum(new_tokens)
                res = self.privatize(
                    sentence, t,
                    n=new_n[i + added - deleted],
                    start_index=i + added - deleted,
                    epsilon=eps,
                    CONCAT=CONCAT,
                )
                r = res[t + "_{}".format(new_n[i + added - deleted])]
                idx = i + added - deleted
                if idx > len(new_tokens) - 1:
                    new_tokens.insert(idx, r)
                else:
                    new_tokens[idx] = r
                replace.append(r)

                if r != t:
                    perturbed += 1
                total += 1
            else:
                new_n = sentence_enum(new_tokens)
                temp = nth_rem(" ".join(new_tokens), t, new_n[i + added - deleted])
                new_tokens = [str(x) for x in temp.split()]
                deleted += 1
                continue

            ADD = np.random.rand()
            if ADD <= ADD_PROB:
                res = self.privatize(
                    sentence, "MASK",
                    n=1,
                    start_index=i + 1 + added - deleted,
                    epsilon=eps,
                    CONCAT=CONCAT,
                )
                r = res["MASK_1"]
                new_tokens.insert(i + 1 + added - deleted, r)
                replace.append(r)
                added += 1

        return self.detokenizer.detokenize(replace), perturbed, total, added, deleted
