# -*- coding: utf-8 -*-
"""Path helpers for the released data layout.

Layout (see data/README.md for the preparation rules):

    data/corpus/<dataset>/text.npy, labels.npy     whole labelled corpus
    data/splits/<dataset>/fold<N>/<split>.{text,labels}.npy
    data/fold1/<split>.{text,emb,labels}.npy       runnable slice of fold 1 (shipped)
    data/embeddings/dataset2/fold1/...             runnable evaluation slice (shipped)
    data/support/...                               kmeans, top16, per-fold priors (shipped, fold 1)

Full embeddings for all five folds are 4-6 GB per fold and are NOT shipped; only the
demo slices above are, so that src/hcemor/train_DCD_v6_4.py and src/hcemor/test6_4.py
run end-to-end out of the box.

Overrides: HCEMOR_DATA_ROOT, HCEMOR_EMB_ROOT, HCEMOR_SUPPORT_ROOT, HCEMOR_MODEL_ROOT.
"""
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_ROOT = os.environ.get("HCEMOR_DATA_ROOT", os.path.join(REPO, "data"))
EMB_ROOT = os.environ.get("HCEMOR_EMB_ROOT", os.path.join(DATA_ROOT, "embeddings"))
SUPPORT_ROOT = os.environ.get("HCEMOR_SUPPORT_ROOT", os.path.join(DATA_ROOT, "support"))
MODEL_ROOT = os.environ.get("HCEMOR_MODEL_ROOT", os.path.join(DATA_ROOT, "model"))

DATASET = os.environ.get("HCEMOR_DATASET", "dataset1")
SETTING = "full"
DEMO_DIR = os.path.join(DATA_ROOT, "fold1")


def _fold(value):
    """Accept '0'..'4' or '1/'..'5/' and return the 1-based fold number."""
    s = str(value).strip().rstrip("/")
    if not s.isdigit():
        raise ValueError("fold must be numeric, got %r" % value)
    n = int(s)
    return n + 1 if (n < 5 and s in ("0", "1", "2", "3", "4")) else n


def _demo(fold, split, suffix, setting="full"):
    if (setting or SETTING) != "full" or _fold(fold) != 1:
        return None
    p = os.path.join(DEMO_DIR, "%s.%s.npy" % (split, suffix))
    return p if os.path.isfile(p) else None


def _split_dir(fold, setting=None):
    return os.path.join(DATA_ROOT, "splits", DATASET, "fold%d" % _fold(fold))


def text_path(fold, split, setting=None):
    p = _demo(fold, split, "text", setting or SETTING)
    if p:
        return p
    return os.path.join(_split_dir(fold, setting), "%s.text.npy" % split)


def label_path(fold, split, setting=None, dataset=None):
    if dataset in (None, DATASET):
        p = _demo(fold, split, "labels", setting or SETTING)
        if p:
            return p
    if dataset not in (None, DATASET):
        # demo labels live beside the demo embeddings
        cand = os.path.join(EMB_ROOT, dataset, "fold%d" % _fold(fold), "%s.labels.npy" % split)
        if os.path.isfile(cand):
            return cand
    p = os.path.join(DATA_ROOT, "splits", dataset or DATASET, "fold%d" % _fold(fold),
                     "%s.labels.npy" % split)
    _require(p, "labels")
    return p


def corpus_path(kind, dataset=None):
    return os.path.join(DATA_ROOT, "corpus", dataset or DATASET, "%s.npy" % kind)


def emb_path(fold, split, dataset=None):
    if dataset in (None, DATASET):
        p = _demo(fold, split, "emb")
        if p:
            return p
    p = os.path.join(EMB_ROOT, dataset or DATASET, "fold%d" % _fold(fold), "%s.npy" % split)
    _require(p, "dense embeddings")
    return p


def support_path(fold, name):
    if fold is not None:
        for ds in (DATASET, "dataset1"):
            cand = os.path.join(SUPPORT_ROOT, ds, "fold%d" % _fold(fold), "%s.npy" % name)
            if os.path.isfile(cand):
                return cand
    cand = os.path.join(SUPPORT_ROOT, "%s.npy" % name)
    if os.path.isfile(cand):
        return cand
    _require(cand, "support matrix %r" % name)
    return cand


def _require(path, what):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "%s not found: %s\n"
            "Only the fold-1 slice is shipped in data/fold1. Full embeddings and the "
            "remaining support matrices are 4-6 GB per fold; see README.md (section "
            "'Data and what is shipped') and scripts/make_embeddings.md." % (what, path))