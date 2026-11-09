# Future work

Ideas for the next student, ordered by how much they help interpret the results.

## 1. Make the evaluation conclusive

- **Random-selection control.** Rewrite randomly chosen sentences at the same rate as each k. If IDF selection does not beat random selection, rarity is not finding the risky sentences. This is the most important missing experiment.
- **Report how much text is rewritten:** the share of sentences and tokens per k and task. Utility and privacy depend mostly on this amount.
- **More targets per attack** (200–500 instead of 50) or more seeds. With 250 attempts, one hit is worth 0.4 points.
- **Uncertainty everywhere:** 95 % intervals or seed standard deviations on TRIR and ROC-AUC, plus significance tests for the main comparisons. Mark chance level (2 % / 0.2 %) in the plots.
- **A stronger attacker in addition to TRI:** an LLM inferring attributes, or a TRI model with extra pre-training.
- **Same text format for the original condition:** use rebuilt but unprivatized notes, so the comparison is not affected by line breaks and placeholder brackets.
- **5 classification seeds** for procedures, where ROC-AUC jumps by up to 0.08.
- **Rethink γ:**
  - avoid dividing by a tiny baseline TRIR;
  - report absolute differences or a privacy–utility scatter with the non-dominated configurations;
  - add a minimum-utility constraint (e.g. C ≥ 0.5).
- **Compare at the same document-level budget.** ε is per token, so a note's total budget grows with the number of rewritten tokens.

## 2. Improve the method

- **Protect placeholders** during rewriting and rewrite only real words.
- **Rewrite spans, not sentences:** only the rare or identifying words inside a selected sentence.
- **Split long segments at clause boundaries** instead of every 200 tokens.
- **Compute clipping bounds on public clinical text**, or with a DP estimate, so the guarantee does not depend on the private data.

## 3. Combine rarity with identifier detection (DP-IPI)

DP-IPI (Baroud et al., [arXiv 2609.29684](https://arxiv.org/pdf/2609.29684)) rewrites only spans that a fine-tuned RoBERTa-large model detects as indirect personal identifiers, using DP-MLM. It keeps utility close to the non-private baseline, but depends on the identifier categories the detector was trained on. IDF rarity needs no labels and can flag unusual one-off details outside those categories. Possible hybrids:

1. **Union:** rewrite every detected identifier span, plus rare words the detector did not flag.
2. **Two stages:** use IDF to pick candidate sentences, then rewrite only the detected spans inside them.
3. **Rarity-adaptive budget:** rewrite the detected spans, giving rarer spans a smaller ε (more noise) under a fixed per-note budget.

Evaluate against DP-IPI, IDF-only selection and a random control with the same rewriting rate, using TRI and an LLM-based leakage judge.

## 4. Related methods

- Papadopoulou et al. 2022/2023: text sanitization with explicit privacy-risk indicators ([AACL 2022](https://aclanthology.org/2022.aacl-main.18/), [arXiv 2310.14312](https://arxiv.org/abs/2310.14312)).
- Shi et al. 2022, Selective Differential Privacy for Language Modeling ([NAACL](https://aclanthology.org/2022.naacl-main.205)).
- Utpala et al. 2023, DP-Prompt: document-level DP paraphrasing with LLMs ([EMNLP Findings](https://aclanthology.org/2023.findings-emnlp.566)).
- Staab et al. 2025, Language Models are Advanced Anonymizers ([ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f478b2e8ad9ff0756bf5b79fb31c330f-Abstract-Conference.html)).
- Pilán et al. 2024, INTACT: truthful sanitization guided by inference attacks ([arXiv 2412.12928](https://arxiv.org/pdf/2412.12928.pdf)).
- Pilán et al. 2022, Text Anonymization Benchmark (TAB) ([CL](https://aclanthology.org/2022.cl-4.19/)).
- Manzanares-Salor et al. 2024, ML-based re-identification attack (TRI) ([DMKD](https://nr.no/en/publication/2295881)).
