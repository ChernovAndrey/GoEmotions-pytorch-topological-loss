import os
import random
import logging

import torch
import numpy as np
import sys
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, multilabel_confusion_matrix


def error_matrix(y_true, y_pred):
    n_labels = y_true.shape[1]

    # Initialize the misclassification matrix
    error_matrix = np.zeros((n_labels, n_labels), dtype=int)

    # Compute the error matrix
    for i in range(n_labels):
        for j in range(n_labels):
            # Count cases where label i is 1 in y_true but misclassified as j in y_pred
            if i == j:
                error_matrix[i, j] = np.sum((y_true[:, i] == 1) & (y_pred[:, j] == 1))
            else:
                error_matrix[i, j] = np.sum((y_true[:, i] == 1) & (y_pred[:, j] == 1) & (y_true[:, j] == 0))

    return error_matrix


def init_logger():
    logging.basicConfig(
        stream=sys.stdout,  # Ensure logs are sent to stdout
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(labels, preds):
    print('number of non predicted labels:')
    print(np.sum(preds.sum(axis=1) <= 1e-3))

    assert len(preds) == len(labels)
    results = dict()

    results["accuracy"] = accuracy_score(labels, preds)
    results["macro_precision"], results["macro_recall"], results[
        "macro_f1"], _ = precision_recall_fscore_support(
        labels, preds, average="macro")
    results["micro_precision"], results["micro_recall"], results[
        "micro_f1"], _ = precision_recall_fscore_support(
        labels, preds, average="micro")
    results["weighted_precision"], results["weighted_recall"], results[
        "weighted_f1"], _ = precision_recall_fscore_support(
        labels, preds, average="weighted")
    results['multilabel_confusion_matrix'] = multilabel_confusion_matrix(labels, preds)
    results['error_matrix'] = error_matrix(labels, preds)

    return results
