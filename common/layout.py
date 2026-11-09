"""
common/layout.py — single source of truth for experiment settings and file paths.

Every script imports its paths from here, so all steps agree on where files are.
All outputs of one task live under   <base>/outputs_<task>/
TRI outputs live under               <base>/privacy_eval/seed_<seed>/
Aggregated tables live under         <base>/results/

Naming convention (used in folder names):
    original        no privatization (classification baseline)
    selective_k1.0  Selective DP-MLM, rare sentences with r(s) >= mu + 1.0 sigma
    selective_k1.5  ...
    selective_k2.0  ...
    full            Full DP-MLM, every sentence rewritten
"""
from pathlib import Path

# ── Experiment grid used in the thesis ─────────────────────────────────────────
SIGMAS   = ["1.0", "1.5", "2.0"]
EPSILONS = [25.0, 50.0, 75.0, 100.0]

# short name -> Hugging Face id of the DP-MLM rewriting model
MODELS = {
    "roberta_base":     "FacebookAI/roberta-base",
    "roberta_clinical": "pminervini/RoBERTa-base-PM-M3-Voc-hf",
}

TASKS = {
    "mp":   {"label_col": "hospital_expire_flag", "task_name": "MP_IN",
             "multilabel": False, "input_csv": "data/mp_50pct.csv"},
    "proc": {"label_col": "short_codes", "task_name": "PRO_GROUPS_3_DIGITS",
             "multilabel": True, "input_csv": "data/proc_50pct.csv"},
}

CLASSIFICATION_SEEDS = [11, 22, 33]
TRI_SEEDS            = [42, 123, 456, 789, 1337]
SPLIT_SEED           = 123

# Clipping bounds: one file per rewriting model, estimated on the k=2.0 rare pool
# and used for ALL conditions (this is what produced the thesis numbers).
BOUNDS_SOURCE_SIGMA = "2.0"


def setup_names():
    """All privatized setups (without 'original')."""
    return [f"selective_k{k}" for k in SIGMAS] + ["full"]


def dpmlm_tag(setup):
    """Tag used inside step6/step7 file names: '1.0', ..., or 'all'."""
    if setup == "full":
        return "all"
    if setup.startswith("selective_k"):
        return setup.replace("selective_k", "")
    raise ValueError(f"No DP-MLM files for setup '{setup}'")


class Layout:
    def __init__(self, base, task):
        if task not in TASKS:
            raise ValueError(f"task must be one of {list(TASKS)}")
        self.base = Path(base).resolve()
        self.task = task
        self.root = self.base / f"outputs_{task}"
        self.cfg = TASKS[task]

    # ── steps 1-5 ──────────────────────────────────────────────────────────────
    @property
    def input_csv(self):      return self.base / self.cfg["input_csv"]
    @property
    def segmented(self):      return self.root / "step1_segmented.csv"
    @property
    def tokenized(self):      return self.root / "step2_tokenized.csv"
    @property
    def idf(self):            return self.root / "step3_idf.csv"
    def rare(self, k):        return self.root / f"step4_rare_sigma_{k}.csv"
    @property
    def all_sentences(self):  return self.root / "step4_all_sentences.csv"

    def bounds(self, model):
        return self.root / "step5_clipping" / model / f"step5_clipping_bounds_sigma_{BOUNDS_SOURCE_SIGMA}.json"

    def step4_input(self, setup):
        return self.all_sentences if setup == "full" else self.rare(dpmlm_tag(setup))

    # ── steps 6-7 ──────────────────────────────────────────────────────────────
    def dpmlm_dir(self, setup, model):  return self.root / "step6_dpmlm" / setup / model
    def notes_dir(self, setup, model):  return self.root / "step7_notes" / setup / model

    def dpmlm_file(self, setup, model, eps):
        return self.dpmlm_dir(setup, model) / f"df_rare_dpmlm_sigma_{dpmlm_tag(setup)}_eps_{float(eps)}.csv"

    def notes_file(self, setup, model, eps):
        return self.notes_dir(setup, model) / f"priv_notes_{self.dpmlm_file(setup, model, eps).stem}.csv"

    # ── step 8 (classification) ────────────────────────────────────────────────
    def _exp(self, setup, model, eps):
        if setup == "original":
            return Path("original")
        return Path(setup) / model / f"eps_{float(eps)}"

    def splits_dir(self, setup, model=None, eps=None):
        return self.root / "step8_splits" / self._exp(setup, model, eps)

    def config_file(self, setup, model=None, eps=None):
        return self.root / "step8_configs" / self._exp(setup, model, eps) / "config.yaml"

    def results_dir(self, seed, setup, model=None, eps=None):
        return self.root / "step8_results" / f"seed_{seed}" / self._exp(setup, model, eps)

    @property
    def label_file(self):     return self.root / "ALL_3_DIGIT_PRO_CODES.txt"

    # ── evaluation tables ──────────────────────────────────────────────────────
    def classification_seed_csv(self, seed): return self.base / "results" / f"classification_{self.task}_seed_{seed}.csv"
    @property
    def classification_avg(self):  return self.base / "results" / f"classification_{self.task}_averaged.csv"
    @property
    def utility_csv(self):         return self.base / "results" / f"utility_metrics_{self.task}.csv"
    @property
    def gamma_csv(self):           return self.base / "results" / f"gamma_tradeoff_{self.task}.csv"
    @property
    def figures_dir(self):         return self.base / "results" / "figures"


# ── privacy (TRI) ──────────────────────────────────────────────────────────────
def tri_run_dir(base, seed):
    return Path(base).resolve() / "privacy_eval" / f"seed_{seed}"


def trir_averaged_csv(base):
    return Path(base).resolve() / "results" / "trir_averaged_results.csv"


def repo_root():
    return Path(__file__).resolve().parent.parent
