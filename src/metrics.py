"""Metrics beyond plain accuracy: precision/recall/F1 and a confusion matrix.

For a binary task these are cheap to compute and give a reviewer a much
better sense of failure modes than accuracy alone (e.g. is the model biased
toward predicting "positive"?).
"""

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred) -> dict:
    """y_true is always 0/1 (real dataset labels). y_pred CAN contain 2
    (see data.label2id) when the model generates something that isn't
    exactly "negative"/"positive" -- most common early in training before
    the model has learned the expected output format.

    sklearn's average="binary" requires a strictly 2-class target, so a
    stray 2 in y_pred makes it raise ValueError("Target is multiclass...").
    A malformed generation is, semantically, just a wrong answer -- so we
    map it to the incorrect label (relative to the true one) rather than
    letting it leak in as a third class. That keeps y_pred binary while
    still correctly counting it against accuracy/precision/recall.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)

    num_malformed = sum(1 for p in y_pred if p not in (0, 1))
    malformed_rate = num_malformed / len(y_pred) if y_pred else 0.0

    y_pred_clean = [p if p in (0, 1) else (1 - t) for p, t in zip(y_pred, y_true)]

    accuracy = accuracy_score(y_true, y_pred_clean)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_clean, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred_clean, labels=[0, 1]).tolist()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
        "malformed_generation_rate": malformed_rate,  # diagnostic: how often the model
                                                        # failed to output "negative"/"positive" verbatim
    }
