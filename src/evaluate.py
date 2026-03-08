from seqeval.metrics import classification_report
from transformers import Trainer

def compute_metrics(p):

    predictions, labels = p

    predictions = predictions.argmax(axis=2)

    true_labels = []
    true_predictions = []

    for prediction, label in zip(predictions, labels):

        true_labels.append(label)
        true_predictions.append(prediction)

    return classification_report(true_labels, true_predictions)