# Data preparation specification

This directory documents *how the data were prepared*, not just where the files sit.
Every artifact is one of: a corpus-level array (`corpus/`), a per-fold split (`splits/`), a
shipped demo slice (`fold1/`, `embeddings/`, `support/`).
Working files that merely restate another artifact, or that the released pipeline never reads,
are not included. `index.csv` lists every file with size, md5, shape, dtype and row count.

---

## What is shipped, and why only fold 1 carries embeddings

GitHub rejects any file over **100 MB**, so the dense token embeddings cannot be shipped at full
width: one fold is about 4 GB and its `train` file alone is 2.8 GB. The full fold-1 tensors are
deposited at **https://doi.org/10.5281/zenodo.24201644**; see *Not in this repository* at the end
of this file.

All five folds are shipped as **text and labels** under `splits/`; the dense embeddings are not.
Two small slices carry embeddings so that both pipelines run on a fresh clone:

| Shipped slice | Rows | Where |
|---|---:|---|
| dataset1 fold-1 training slice | 16 train / 16 val | `fold1/` |
| dataset2 fold-1 evaluation slice | 16 per split | `embeddings/dataset2/fold1/` |
| Support matrices for fold 1 | - | `support/` |

Every fold, including the shipped slices, keeps the same convention per split:

```
<split>.text.npy      (N,)            comment text
<split>.labels.npy    (N, K)          multi-label targets
<split>.emb.npy       (N, 256, 768)   dense token embeddings (dataset1 slice only)
```

with the same rows in the same order. The slices exist so that `train_DCD_v6_4.py` and
`test6_4.py` run end to end; they are not meant to reproduce the reported numbers.

Full-width embeddings can be produced with `scripts/make_embeddings.md` or taken from the Zenodo
deposit. For folds 2-5 place them under `data/embeddings/dataset1/fold<N>/` and
`src/hcemor/paths.py` picks them up; for fold 1 the shipped slice in `data/fold1/` takes
precedence, so remove or rename that directory first.

---

## Stage 0 — Source material

| Item | Value |
|---|---|
| Platform | Zhihu, a Chinese question-and-answer community |
| Unit of analysis | one comment (a doctor-patient-related statement) |
| Raw postings collected | 143,014 distinct Zhihu postings (reported in the paper) |
| Label table (dataset1) | 35,938 rows: `sentence` + 22 binary tags (+ 2 helper columns `x`, `v`) |
| Label table (dataset2) | 22,500 rows: `id`, `text`, 9 binary tags, `fold`, `split` |

The original spreadsheets are **not** duplicated in this archive: the released content is
exactly the `corpus/` arrays plus the column metadata in `corpus/*/meta.json`. Ask the authors
if you need the original workbooks.

## Stage 1 — Corpus construction

`corpus/dataset1/` and `corpus/dataset2/` hold the whole labelled corpus, one row per comment.
`corpus/` and `splits/` carry the released text.

| File | Shape | dtype | Content |
|---|---|---|---|
| `corpus/dataset1/text.npy` | (35938,) | object | comment text, column `sentence` |
| `corpus/dataset1/labels.npy` | (35938, 22) | int64 | binary multi-label matrix |
| `corpus/dataset1/meta.json` | — | — | `label_names` bound to the matrix columns |
| `corpus/dataset2/text.npy` | (22500,) | object | comment text, column `text` |
| `corpus/dataset2/labels.npy` | (22500, 9) | int64 | binary multi-label matrix |

Row *i* of `text.npy` and row *i* of `labels.npy` are the same comment in both datasets.

### Label schema

The tag vocabulary is a two-level hierarchy: a parent factor (letter) and its sub-classes
(digit). `D` = patient-side factor, `E` = physician-side factor, `F` = industry/practice
factor, `G` = social-environmental factor; `C` appears only in dataset2.

| Dataset | Parent | Sub-classes | Count |
|---|---|---|---:|
| dataset1 | D | `D_1`-`D_5` | 5 |
| dataset1 | E | `E_1`-`E_4` | 4 |
| dataset1 | F | `F_1`-`F_8` | 8 |
| dataset1 | G | `G_1`-`G_5` | 5 |
| dataset2 | D | `D_1`-`D_5` | 5 |
| dataset2 | C | `C_1`-`C_3` | 3 |
| dataset2 | E | `E_1` | 1 |


### dataset1 vs dataset2

| | dataset1 | dataset2 |
|---|---:|---:|
| Labelled comments in the corpus | 35,938 | 22,500 |
| Comments actually used by the five folds | 16,065 distinct | 4,472 distinct |
| Label columns in the corpus | 22 (`D_1`-`G_5`) | 9 (`D_1`-`D_5`, `C_1`-`C_3`, `E_1`) |
| Label columns that are never positive | 2 (`D_5`, `E_4`) | 0 |
| Closed-set label columns per fold | 6 | 6 |
| Open-test label matrix | 9 multi-label columns | 1 binary column |
| Comments with no tag at all | 10,574 (29%) | 10 (0.04%) |
| Mean tags per comment | 1.58 | 2.48 |
| Rows per fold (train / val / close_test / open_test) | 5,554-7,136 | 648-1,040 |
| Mean comment length | 36 characters | 40 characters |
| Overlap with the other dataset | 0 shared comments | 0 shared comments |

The two corpora are independent: no comment appears in both. `dataset2` introduces three classes
(`C_1`-`C_3`) that do not exist in `dataset1`, and its open-set test set is annotated with a
single unknown/known flag rather than a multi-label matrix.

Class names, the parent-subclass hierarchy and the annotation protocol are described in the
paper and its online appendix; this directory ships only the label matrices. The column order of
each matrix is recorded in `corpus/<dataset>/meta.json` (`label_names`).

## Stage 2 — Fold construction

`splits/<dataset>/fold<N>/` holds the evaluation split of fold *N* (N = 1..5).

| File | Content |
|---|---|
| `train.text.npy` / `train.labels.npy` | training rows |
| `val.text.npy` / `val.labels.npy` | validation rows, used for early stopping |
| `close_test.text.npy` / `close_test.labels.npy` | closed-set test rows |
| `open_test.text.npy` / `open_test.labels.npy` | open-set test rows |

Rules:

- The five folds are a **random** split of the corpus, drawn once by the authors and then fixed.
  They are not chosen per method: every model reported in the paper, and any model you train on
  this archive, is evaluated on exactly these folds.
- Within a fold the split is stratified, so that every sub-class present in train is also
  present in val and in the closed-set test; which classes are held out as unknown is likewise a
  random draw per fold (recorded in `settings/open_set_setting.json`).
- `train`, `val` and `close_test` share the same tag vocabulary per fold (the *known* classes).
- `open_test` additionally contains comments whose tags include the *unknown* classes of that
  fold, so its label matrix is wider than the closed-set one: **9 columns for `dataset1`**
  versus **1 binary column for `dataset2`** (1 = the comment carries an unknown class).
  The fold's six known classes are listed in `settings/open_set_setting.json`.
- The unknown classes are chosen per fold and are disjoint from the known ones; the choice is
  recorded in `settings/open_set_setting.json`.

Per-file row counts are listed in `index.csv`.

## Stage 3 — Open-set setting

| File | Content |
|---|---|
| `settings/open_set_setting.json` | known classes and candidate-unknown classes per fold |

## File format specification

| Property | Rule |
|---|---|
| Container | NumPy `.npy`, little-endian |
| Text arrays | 1-D `dtype=object`, element = one Python `str`; load with `allow_pickle=True` |
| Label arrays | 2-D, `int64`, values in {0, 1}; column order as in the matching `meta.json` |
| Naming | `<split>.text.npy` and `<split>.labels.npy`; row *i* of the two files is the same comment |
| Encoding | UTF-8 (Chinese text, no BOM) |
| Order | Rows within a split are in the order of the source table; do not assume any other sort |

The text arrays are object arrays pickled by **NumPy 1.x**, so they load under both NumPy 1.x and
NumPy 2.x. (Arrays pickled by NumPy 2.x cannot be read by NumPy 1.x: `No module named 'numpy._core'`.)
The label arrays are plain `int64` and need no pickle at all.

## Integrity checks

- Every `text.npy` in `splits/` has a matching `labels.npy` with the same number of rows,
  except the four `dataset1` `train` splits listed under *Known deviations*.
- The union of the 16,065 distinct texts used across the five `dataset1` folds is a strict
  subset of the 35,938-row corpus, and every one of those texts matches its corpus row exactly.
- `index.csv` carries an md5 per file so a download can be verified without re-running the
  preparation.

## Known deviations from the intended pipeline

| # | Deviation | Consequence |
|---|---|---|
| 1 | `dataset1` `train` row counts differ from the matching label counts in folds 2-5 (4,222 vs 4,221; 4,755 vs 4,773; 4,582 vs 4,575; 5,720 vs 5,727) | the affected splits cannot be paired row-wise; left as exported rather than silently truncated |
| 2 | `dataset1` `open_test` text is missing for folds 2-5 | only the label file is available there |
| 3 | Corpus rows (35,938) far exceed the rows used by any fold (16,065 distinct) | the folds are subsets; the surplus rows were labelled but not used in the reported experiments |

## Reproducing the prepared arrays

```python
import json, numpy as np
text = np.load("corpus/dataset1/text.npy", allow_pickle=True)
labels = np.load("corpus/dataset1/labels.npy")
meta = json.load(open("corpus/dataset1/meta.json", encoding="utf-8"))
assert len(text) == labels.shape[0] == meta["n_rows"] == 35938
assert labels.shape[1] == len(meta["label_names"]) == 22
print(text[0][:40], labels[0][:6])
```

## Provenance and licence

- Collected and labelled by the authors from Zhihu for research use; user identifiers removed.
- The five-fold splits were generated by the authors.
- Released under the MIT License together with the code; see `../LICENSE`.
- Provided for academic reproduction only. Please cite the paper and this repository.

## Not in this repository

The dense tensors are deposited separately at **https://doi.org/10.5281/zenodo.24201644**.
Two archives are involved:

| Archive | Contents |
|---|---|
| `fold1_full.tar` | dataset1 fold-1 tensors: `*.emb.npy`, `*.text.npy`, `*.labels.npy`, `kmeans.npy`, `support/` |
| `zenodo_extras.tar.gz` | `support/top16.npy`, `support/dataset1/fold1/{ufde,parent_class2,open_test_results}.npy`, and the dataset2 fold-1 demo tensors under `embeddings/dataset2/fold1/` |

Unpack both at the repository root; `src/hcemor/paths.py` then resolves everything.