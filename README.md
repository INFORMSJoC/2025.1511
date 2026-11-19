[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Enhancing Understanding of Doctor-Patient Relationships through Knowledge Sharing in Online Communities: A Deep Learning Framework with Open Set Recognition

This archive is distributed in association with the [INFORMS Journal on Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The software and data in this repository are a snapshot of the software and data that were used in the research reported on in the paper Enhancing Understanding of Doctor-Patient Relationships through Knowledge Sharing in Online Communities: A Deep Learning Framework with Open Set Recognition by Kaixiang Su, Dongxiao Gu, Xiao-Bai Li, Shunlin Yang, and Kaixuan Zhu.

## Cite

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2025.1511

https://doi.org/10.1287/ijoc.2025.1511.cd

Below is the BibTex for citing this snapshot of the repository.

```
@article{SuGuLiYangZhu2026,
 author =    {Kaixiang Su, Dongxiao Gu, Xiao-Bai Li, Shunlin Yang, and Kaixuan Zhu},
 publisher = {INFORMS Journal on Computing},
 title =     {Enhancing Understanding of Doctor-Patient Relationships through Knowledge
             Sharing in Online Communities: A Deep Learning Framework with Open Set Recognition},
 year =      {2026},
 doi =       {10.1287/ijoc.2025.1511.cd},
 url =       {https://github.com/INFORMSJoC/2025.1511},
 note =      {Available for download at https://github.com/INFORMSJoC/2025.1511}
}

```

## Description

Code, annotated data and results of the HCEMOR framework. HCEMOR mines doctor-patient
relationship (DPR) factors from knowledge shared in online communities and uses open set
recognition to surface factors that are absent from the training taxonomy. The reference
implementation is `src/hcemor/DCD_NET_v6_4.py` (DCANet + MCLUFI), trained by
`train_DCD_v6_4.py` and evaluated by `test6_4.py`.

Evaluation uses 5-fold class-based cross-validation over two labelled corpora
(`dataset1`: 35,938 labelled comments, 22 classes; `dataset2`: 22,500 labelled comments,
9 classes).

## Setup

Python 3.9+ with the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

All paths are resolved by `src/hcemor/paths.py`; every location can be overridden:

| Variable | Default | Holds |
|---|---|---|
| `HCEMOR_DATA_ROOT` | `<repo>/data` | text, labels, settings (shipped) |
| `HCEMOR_EMB_ROOT` | `<data root>/embeddings` | dense embeddings (Zenodo) |
| `HCEMOR_SUPPORT_ROOT` | `<data root>/support` | class priors, cluster centres (Zenodo) |
| `HCEMOR_MODEL_ROOT` | `<data root>/model` | checkpoints written by training |
| `HCEMOR_FOLD`, `HCEMOR_DATASET`, `HCEMOR_CKPT`, `HCEMOR_GPU` | — | evaluation switches |

## Data

This repository ships the **text and the multi-label targets** for both
corpora, plus two small slices so that both pipelines can be run end to end straight after cloning:
16 rows of fold 1 of `dataset1` (for training) and 16 rows per split of fold 1 of `dataset2`
(for evaluation), together with the LLM-UFDE auxiliary embeddings, the class prototypes and
the support matrices. See [data/README.md](data/README.md) for the preparation
rules, the file format and the known deviations.

The **full dense embeddings and the full support matrices** are several GB and are not part of
the repository. They are deposited at

**https://doi.org/10.5281/zenodo.24201644**

Download the archive and unpack it so that the tree looks like this:

```
data/embeddings/dataset1/fold1/{train,val,close_test,open_test}.npy
data/embeddings/dataset2/fold1/{train,val,close_test,open_test}.npy   (+ .labels.npy)
data/support/kmeans.npy
data/support/top16.npy
data/support/dataset1/fold1/{ufde,parent_class2,child_class,parent_class_unique,c2c,c2m,m2m}.npy
data/support/dataset1/fold1/open_test_results.npy
```

`scripts/make_embeddings.md` documents how the tensors were produced (encoder, 256-token
limit, full last hidden state, float32), should you prefer to regenerate them.

## Replicating

```bash
# 0. smoke run on the shipped 16-row slice (one epoch, CPU is fine)
scripts/run_train.sh --epochs 1 --gpu -1 --batch_size 16

# 1. train all five folds and ten seeds on the full data (edit FOLD_DIRS / SEEDS to run a subset)
scripts/run_train.sh --batch_size 16

# 2. evaluate on the shipped dataset2 slice (uses the checkpoint saved by step 0)
HCEMOR_DATASET=dataset2 HCEMOR_FOLD=0 \
  HCEMOR_CKPT=data/model/fold1/train_simple(lowe1e2)1.pt scripts/run_test.sh

```

The batch size must equal the number of rows of `data/support/top16.npy` (16): that tensor is
batch-coupled in the forward pass. Steps 0 and 2 both run on the shipped slices with no
download; they print metrics over 16 rows, so the numbers are only a smoke check. Steps 1 and
the full five-fold evaluation need the Zenodo archive.

## Repository structure

| Path | Contents |
|---|---|
| `src/hcemor/` | reference implementation: `DCD_NET_v6_4.py`, `train_DCD_v6_4.py`, `test6_4.py`, `paths.py` |
| `src/evaluation/` | multi-label and open-set metric implementations |
| `data/` | text, labels, open-set setting ([data/README.md](data/README.md)) |
| `results/` | tables as published (`online_results.pdf`) |
| `docs/` | model diagram and the LLM annotation prompt |
| `scripts/` | training and evaluation wrappers |

## Results

`results/online_results.pdf` contains the tables and figures as published. 


## Notes from the experiments

1. **More training epochs help open-set recognition but can hurt closed-set accuracy.**
   The reference configuration selects the checkpoint by validation macro-F1 rather than by
   the final epoch (`--monitor` / `--patience`).

2. **Richer and more varied latent unknown-class semantics tend to improve performance.**
   Increasing both the number and the types of the LLM-generated unknown-factor descriptions
   (LLM-UFDE) improved open-set scores in our preliminary trials.

## License

Code and data in this repository are released under the MIT License; see [LICENSE](LICENSE).
