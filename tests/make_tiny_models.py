"""
make_tiny_models.py — builds TINY, RANDOMLY INITIALISED models locally, so the
smoke test runs on CPU without downloading anything from Hugging Face.

    <out>/tiny-roberta     masked LM   (stands in for RoBERTa-base / -clinical; DP-MLM, step 5)
    <out>/tiny-distilbert  DistilBERT  (stands in for the TRI attacker)
    <out>/tiny-gpt2        causal LM   (stands in for GPT-2 perplexity and Qwen2.5 backgrounds)

Their outputs are meaningless; they only exercise the code paths.
"""
import argparse
from pathlib import Path

import pandas as pd
from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer
from transformers import (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizerFast,
                          GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast,
                          RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast)

CHAT_TEMPLATE = ("{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }}\n{% endfor %}"
                 "{% if add_generation_prompt %}<|assistant|> {% endif %}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--texts", required=True, help="CSV with a 'text' column")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    texts = pd.read_csv(a.texts)["text"].astype(str).tolist() * 3
    tmp = out / "_corpus.txt"
    tmp.write_text("\n".join(texts))

    # RoBERTa-style masked LM
    bpe = ByteLevelBPETokenizer()
    bpe.train([str(tmp)], vocab_size=800, min_frequency=1,
              special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    d = out / "tiny-roberta"
    d.mkdir(exist_ok=True)
    bpe.save_model(str(d))
    tok = RobertaTokenizerFast(vocab_file=str(d / "vocab.json"), merges_file=str(d / "merges.txt"),
                               model_max_length=512)
    tok.save_pretrained(d)
    RobertaForMaskedLM(RobertaConfig(vocab_size=tok.vocab_size, hidden_size=32, num_hidden_layers=2,
                                     num_attention_heads=2, intermediate_size=64,
                                     max_position_embeddings=520, pad_token_id=tok.pad_token_id)
                       ).save_pretrained(d)

    # DistilBERT (directory name must contain "distilbert" for tri.py)
    wp = BertWordPieceTokenizer(lowercase=True)
    wp.train([str(tmp)], vocab_size=600, min_frequency=1,
             special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    d = out / "tiny-distilbert"
    d.mkdir(exist_ok=True)
    wp.save_model(str(d))
    tok = DistilBertTokenizerFast(vocab_file=str(d / "vocab.txt"), model_max_length=512)
    tok.save_pretrained(d)
    DistilBertForMaskedLM(DistilBertConfig(vocab_size=tok.vocab_size, dim=32, n_layers=2, n_heads=2,
                                           hidden_dim=64)).save_pretrained(d)

    # GPT-2-style causal LM with a chat template (Qwen stand-in)
    d = out / "tiny-gpt2"
    d.mkdir(exist_ok=True)
    bpe2 = ByteLevelBPETokenizer()
    bpe2.train([str(tmp)], vocab_size=800, min_frequency=1, special_tokens=["<|endoftext|>"])
    bpe2.save_model(str(d))
    tok = GPT2TokenizerFast(vocab_file=str(d / "vocab.json"), merges_file=str(d / "merges.txt"))
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(d)
    GPT2LMHeadModel(GPT2Config(vocab_size=tok.vocab_size, n_embd=32, n_layer=2, n_head=2,
                               n_positions=1024)).save_pretrained(d)
    tmp.unlink()
    print(f"✓ tiny models in {out}")


if __name__ == "__main__":
    main()
