"""Stand-in for spaCy's en_core_web_lg in the smoke test. tri.py imports it at
the top, but only uses it for background anonymization / document curation,
which the thesis configuration switches off."""


def load():
    raise RuntimeError("en_core_web_lg stub: install the real model with "
                       "`python -m spacy download en_core_web_lg`")
