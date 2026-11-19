"""Training pipeline for HCEMOR (DCANet + MCLUFI).

Training script accompanying the paper
"Enhancing Understanding of Doctor-Patient Relationships through Knowledge
Sharing in Online Communities: A Deep Learning Framework with Open Set
Recognition" (INFORMS Journal on Computing).

The pipeline trains the DCANet backbone together with the MCLUFI contrastive
probing heads on top of frozen RoBERTa embeddings:

- ``inputs1`` (x)  : embeddings of the basic training data with known factors;
- ``inputs2`` (x') : embeddings of the LLM-UFDE generated data (auxiliary
  supervision for open-set learning, never used at inference);
- ``inputs3``      : background semantics without specific factors;
- ``train_Y2``     : multi-label known-class targets.

Training follows a standard supervised regime with model selection:

1. the model is validated after every epoch;
2. whenever the monitored comprehensive validation metric improves, the best
   checkpoint is (re-)saved to disk;
3. early stopping with a patience of 5 epochs: training stops when the
   monitored metric has not improved for 5 consecutive epochs, and the best
   checkpoint (not the last one) is kept.

The total objective is the composite loss of Eq. (21):
    loss = w1 * L_cls + w2 * L_rec + w3 * L_exp + w4 * L_div
with the loss weights selected on the validation set in the paper
(w1 = 1.0, w2 = 0.5, w3 = 0.05, w4 = 0.05).

Example
-------
    python train_DCD_v6_4.py --gpu 0 --epochs 40 --patience 5 --monitor macro_f1
"""

import argparse
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import DCD_NET_v6_4
from paths import (DATA_ROOT, MODEL_ROOT, emb_path, label_path,
                   support_path)

# ---------------------------------------------------------------------------
# Composite loss weights (Eq. 21). w1 is the reference weight of the known-class
# BCE loss; w2/w3/w4 weight the MCLUFI reconstruction, exploration, and
# diversity losses, respectively (selected on the validation set, Section 4.2).
# ---------------------------------------------------------------------------
w1 = 1.0
w2 = 0.5
w3 = 0.05
w4 = 0.05

# Experiment protocol: 5-fold class-based cross-validation x 10 random seeds.
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
FOLD_DIRS = ['1/', '2/', '3/', '4/', '5/']
FOLD_IDS = ['1', '2', '3', '4', '5']
FOLD_INDICES = ['0', '1', '2', '3', '4']
# Per-fold positive-class weights of the BCEWithLogitsLoss (class imbalance).
FOLD_CLASS_WEIGHTS = [
    [3.5, 10., 4.5, 4.0, 8., 5.],
    [5.5, 8.0, 10., 4., 3.5, 4.],
    [4., 10., 5., 3., 7., 5.],
    [3.5, 10., 8., 4.5, 5., 3.5],
    [3., 10., 3.5, 5., 8.5, 5.],
]

logger = logging.getLogger("HCEMOR")

# Computation device (module-level default; refined by the --gpu argument in
# the main entry point via torch.cuda.set_device).
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_args():
    """Parse command-line training configuration."""
    parser = argparse.ArgumentParser(description='Train HCEMOR (DCANet + MCLUFI)')
    # Optimization (hyperparameters reported in Section 4.2).
    parser.add_argument('--lr', default=0.0001, type=float, help="learning rate")
    parser.add_argument('--epochs', default=40, type=int, help="Maximum number of training epochs")
    parser.add_argument('--batch_size', default=16, type=int, help="Batch size")
    parser.add_argument('--momentum', default=0.9, type=float, help="momentum")
    parser.add_argument('--weight_decay', default=0.0005, type=float, help="weight decay")
    parser.add_argument('--lossweight', default=0.7, type=float, help="weight between losses")
    parser.add_argument('--alpha', default=5, type=float, help="P, number of MCLUFI probe networks")
    parser.add_argument('--tempreture', default=1.0, type=float,
                        help="degree of influence of parent-class routing on subclass routing")
    # Early-stopping / model selection.
    parser.add_argument('--patience', default=5, type=int,
                        help="Early-stopping patience: stop after this many consecutive epochs "
                             "without improvement of the monitored validation metric")
    parser.add_argument('--monitor', default='macro_f1',
                        choices=['macro_f1', 'acc', 'loss'],
                        help="Comprehensive validation metric used for best-checkpoint selection "
                             "and early stopping")
    # Environment and data locations (defaults reproduce the original setup).
    parser.add_argument('--gpu', default=0, type=int,
                        help="CUDA device index (-1 forces CPU)")
    parser.add_argument('--data_root', default=DATA_ROOT,
                        help="Directory holding per-fold features/labels (数据/标签 npy files)")
    parser.add_argument('--matrix_root', default=os.path.join(DATA_ROOT, 'support'),
                        help="Directory holding per-fold semantic matrices and LLM-UFDE embeddings")
    parser.add_argument('--background_path', default=os.path.join(DATA_ROOT, 'support', 'top16.npy'),
                        help="Background semantics embeddings (no specific factors)")
    parser.add_argument('--model_dir', default=MODEL_ROOT,
                        help="Output directory for checkpoints and training history")
    parser.add_argument('--num_workers', default=0, type=int, help="DataLoader workers")

    return parser.parse_args()


def set_seed(seed):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _macro_f1(preds, targets, eps=1e-8):
    """Multi-label macro-averaged F1 at threshold 0.5.

    Args:
        preds: binarized predictions, array-like ``(n_samples, n_classes)`` in {0, 1}.
        targets: ground-truth labels with the same shape.

    Returns:
        Macro-averaged F1 over classes (classes with no true and no predicted
        positive contribute 0 through the eps-smoothed formula).
    """
    preds = np.asarray(preds).astype(np.int64)
    targets = np.asarray(targets).astype(np.int64)
    tp = ((preds == 1) & (targets == 1)).sum(axis=0).astype(np.float64)
    fp = ((preds == 1) & (targets == 0)).sum(axis=0).astype(np.float64)
    fn = ((preds == 0) & (targets == 1)).sum(axis=0).astype(np.float64)
    precision = tp / np.maximum(tp + fp, eps)
    recall = tp / np.maximum(tp + fn, eps)
    f1 = 2 * precision * recall / np.maximum(precision + recall, eps)
    return float(f1.mean())


@torch.no_grad()
def validate(net, valid_loader, criterion, device):
    """Evaluate the model on the validation set.

    Returns:
        dict with keys:
            'loss'      : mean known-class BCE loss (Eq. 5);
            'acc'       : label-level accuracy (fraction of correct labels);
            'macro_f1'  : multi-label macro-F1 at threshold 0.5;
            'probe_act' : mean LogSumExp-smoothed maximum probe activation on
                          the LLM-UFDE data (``xlogit``), the statistic later
                          used for Weibull-based open-set detection (Eq. 20).
    """
    net.eval()
    total_loss = 0.0
    total_correct_labels = 0.0
    total_labels = 0
    total_probe_act = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []

    for val_inputs1, val_inputs2, val_labels in valid_loader:
        val_inputs1 = val_inputs1.to(device)
        val_inputs2 = val_inputs2.to(device)
        val_labels = val_labels.to(device)

        val_logit, val_sim = net(val_inputs1, val_inputs2)

        val_cls_loss = criterion(val_logit, val_labels.float())
        total_loss += val_cls_loss.item()
        total_probe_act += float(val_sim[0].mean().item())
        n_batches += 1

        predicted = torch.round(torch.sigmoid(val_logit.detach()))
        all_preds.append(predicted.cpu().numpy())
        all_targets.append(val_labels.cpu().numpy())
        total_correct_labels += (predicted == val_labels).sum().item()
        total_labels += val_labels.numel()

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return {
        'loss': total_loss / n_batches,
        'acc': total_correct_labels / total_labels,
        'macro_f1': _macro_f1(preds, targets),
        'probe_act': total_probe_act / n_batches,
    }


def epoch_train(fold, times, ind, weight, fold2):
    """Train HCEMOR on one cross-validation fold with one random seed.

    Standard training mode: after every epoch the model is validated, the best
    checkpoint (by the monitored comprehensive validation metric) is saved, and
    training stops once the metric has not improved for ``args.patience``
    consecutive epochs.

    Args:
        fold: fold directory name with trailing slash (e.g. ``'1/'``).
        times: random-seed tag used in the checkpoint file name.
        ind: zero-based fold index used in the data paths (e.g. ``'0'``).
        weight: positive-class weights of the BCEWithLogitsLoss.
        fold2: fold id (kept for interface compatibility with the original
            experiment protocol).
    """
    logger.info("start training | fold=%s seed=%s", fold, times)

    # ------------------------------------------------------------------
    # Data loading (frozen RoBERTa embeddings).
    #   train_X0/valid_X0 : basic data embeddings, (num_of_data, 512, 768)
    #   train_X1/valid_X1 : LLM-UFDE generated data embeddings (auxiliary)
    #   train_Y2/valid_Y2 : multi-label known-class targets
    # ------------------------------------------------------------------
    data_dir = os.path.join(args.data_root, ind)
    matrix_dir = os.path.join(args.matrix_root, ind)

    total_extra_data = np.load(support_path(ind, 'ufde'))  # LLM-UFDE
    total_size = total_extra_data.shape[0]

    os.makedirs(args.model_dir, exist_ok=True)
    output_fn = os.path.join(args.model_dir, f"fold{fold}train_simple(lowe1e2){times}.pt")
    os.makedirs(os.path.dirname(output_fn), exist_ok=True)
    history_fn = output_fn[:-3] + '_history.json'

    train_X0 = torch.from_numpy(np.load(emb_path(ind, 'train'), allow_pickle=True))
    # Sample an equal amount of LLM-UFDE auxiliary data without replacement.
    indices1 = np.random.choice(total_size, train_X0.shape[0], replace=False)
    train_X1 = torch.from_numpy(total_extra_data[indices1])

    X2 = np.load(args.background_path).astype(np.float32)          # background semantics
    inputs3 = torch.Tensor(X2)

    train_Y2 = torch.from_numpy(
        np.load(label_path(ind, 'train'), allow_pickle=True).astype(float))

    valid_X0 = torch.from_numpy(np.load(emb_path(ind, 'val'), allow_pickle=True))
    indices2 = np.random.choice(total_size, valid_X0.shape[0], replace=False)
    valid_X1 = torch.from_numpy(total_extra_data[indices2])
    valid_Y2 = torch.from_numpy(
        np.load(label_path(ind, 'val'), allow_pickle=True).astype(float))

    # Expert-defined class prototypes and hierarchy mapping matrix (HCKA).
    semantic_main = torch.from_numpy(np.load(support_path(ind, 'parent_class2'), allow_pickle=True).astype(np.float32))
    semantic_sub = torch.from_numpy(np.load(support_path(ind, 'child_class'), allow_pickle=True).astype(np.float32))
    c2m = torch.from_numpy(np.load(support_path(ind, 'c2m')))

    # ------------------------------------------------------------------
    # Model, optimizer, and loss.
    # ------------------------------------------------------------------
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size

    cls_criterion1 = nn.BCEWithLogitsLoss(weight=weight).to(device)

    net = DCD_NET_v6_4.DCDNet(semantic_main=semantic_main, semantic_sub=semantic_sub,
                              sem_of_nan=inputs3, beta=args.tempreture, c2m=c2m,
                              granularity=int(args.alpha)).to(device)

    optimizer = optim.Adam(net.parameters(), lr=lr)

    train_dataset = TensorDataset(train_X0, train_X1, train_Y2)
    valid_dataset = TensorDataset(valid_X0, valid_X1, valid_Y2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=args.num_workers)
    # drop_last=True is required (as in the original protocol): the background
    # semantics tensor is batch-coupled, so every batch must have the same size.
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              drop_last=True, num_workers=args.num_workers)

    # ------------------------------------------------------------------
    # Standard training loop with best-checkpoint tracking and early stopping
    # (patience = args.patience epochs without improvement).
    # ------------------------------------------------------------------
    higher_better = args.monitor != 'loss'
    best_metric = None
    best_epoch = -1
    epochs_no_improve = 0
    history = []

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0.0
        n_batches = 0

        for i, data in enumerate(train_loader):
            inputs1, inputs2, labels = data
            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logit, sim = net(inputs1, inputs2)

            # Composite objective (Eq. 21):
            #   sim[1] -> MCLUFI reconstruction loss, sim[2] -> exploration loss,
            #   sim[3] -> probe diversity loss.
            cls_loss = cls_criterion1(logit, labels.float())
            constract1 = sim[1]
            constract2 = sim[2]
            extraloss = sim[3]

            loss = w1 * cls_loss + w2 * constract1 + w3 * constract2 + w4 * extraloss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # --------------------------------------------------------------
        # Validation + model selection.
        # --------------------------------------------------------------
        metrics = validate(net, valid_loader, cls_criterion1, device)
        train_loss = epoch_loss / n_batches
        improved = best_metric is None or (
            metrics[args.monitor] > best_metric if higher_better else metrics[args.monitor] < best_metric)

        if improved:
            best_metric = metrics[args.monitor]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(net.state_dict(), output_fn)  # save the best checkpoint
            marker = f"* best so far (saved)"
        else:
            epochs_no_improve += 1
            marker = f"no improvement for {epochs_no_improve}/{args.patience} epochs"

        logger.info(
            "fold %s seed %s | epoch %03d/%03d | train loss %.4f | val loss %.4f | "
            "val acc %.2f%% | val macro-F1 %.4f | probe act %.4f | %s",
            fold, times, epoch + 1, epochs, train_loss, metrics['loss'],
            100.0 * metrics['acc'], metrics['macro_f1'], metrics['probe_act'], marker)

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            **metrics,
            'is_best': improved,
        })

        if epochs_no_improve >= args.patience:
            logger.info("early stopping triggered: %s did not improve for %d epochs "
                        "(best epoch %d, best %s = %.4f)",
                        args.monitor, args.patience, best_epoch + 1, args.monitor, best_metric)
            break

    # Record the full training history next to the checkpoint for reproducibility.
    with open(history_fn, 'w', encoding='utf-8') as f:
        json.dump({'fold': fold, 'seed': times, 'monitor': args.monitor,
                   'patience': args.patience, 'best_epoch': best_epoch + 1,
                   'best_' + args.monitor: best_metric, 'history': history},
                  f, ensure_ascii=False, indent=2)

    logger.info("finished | fold=%s seed=%s | best epoch %d | best %s = %.4f | checkpoint: %s",
                fold, times, best_epoch + 1, args.monitor, best_metric, output_fn)


if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    # Device selection: the CUDA device index is chosen once here; the model
    # module resolves 'cuda' against the current device.
    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda')
        logger.info("using GPU %d: %s", args.gpu, torch.cuda.get_device_name(args.gpu))
    else:
        device = torch.device('cpu')
        logger.info("using CPU")

    for idex in range(5):
        weight = torch.tensor(FOLD_CLASS_WEIGHTS[idex])
        fold = FOLD_DIRS[idex]
        id = FOLD_INDICES[idex]
        fold2 = FOLD_IDS[idex]
        for seed in SEEDS:
            set_seed(seed)
            epoch_train(fold=fold, times=str(seed), ind=id, weight=weight, fold2=fold2)
