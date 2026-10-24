import numpy as np
from seqeval.metrics import f1_score
from transformers.trainer_utils import EvalPrediction

from src.dataset import IGNORE_INDEX, LABEL_NAMES


def decode_predictions(
    predictions: np.ndarray, labels: np.ndarray
) -> tuple[list[list[str]], list[list[str]]]:
    true_labels = [[LABEL_NAMES[label] for label in row if label != IGNORE_INDEX] for row in labels]
    true_predictions = [
        [LABEL_NAMES[pred] for pred, label in zip(pred_row, label_row) if label != IGNORE_INDEX]
        for pred_row, label_row in zip(predictions, labels)
    ]
    return true_labels, true_predictions


def compute_metrics(eval_preds: EvalPrediction) -> dict[str, float]:
    predictions = np.argmax(eval_preds.predictions, axis=-1)
    true_labels, true_predictions = decode_predictions(predictions, eval_preds.label_ids)  # ty:ignore[invalid-argument-type]
    return {'f1': f1_score(true_labels, true_predictions)}
