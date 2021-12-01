import numpy as np
import sklearn.metrics as metrics
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, labels, predictions):
        self.labels = np.array(labels)
        self.predictions = np.array(predictions)

    def get_accuracy(self, threshold):
        predicted_labels = np.where(self.predictions > threshold, 1, 0)
        return metrics.accuracy_score(self.labels, predicted_labels)

    def get_balanced_accuracy(self, threshold):
        predicted_labels = np.where(self.predictions > threshold, 1, 0)
        return metrics.balanced_accuracy_score(self.labels, predicted_labels)

    def get_auc_roc(self):
        return metrics.roc_auc_score(self.labels, self.predictions)

    def get_confusion_matrix(self, threshold):
        predicted_labels = np.where(self.predictions > threshold, 1, 0)
        return metrics.confusion_matrix(self.labels, predicted_labels)

    def get_best_threshold(self, target='f1-score'):
        precision, recall, thresholds = metrics.precision_recall_curve(
                self.labels, self.predictions)
        f1_scores = 2*recall*precision/(recall+precision)
        best_threshold = thresholds[np.argmax(f1_scores)]
        best_f1_score = np.max(f1_scores)
        logger.info(f"Best threshold is at {best_threshold} "
                    f"with f1-score {best_f1_score}")
        return best_threshold, best_f1_score

    def print_report(self, threshold='best'):
        if threshold == 'best':
            threshold, f1 = self.get_best_threshold()
        accuracy = self.get_accuracy(threshold)
        balanced_accuracy = self.get_balanced_accuracy(threshold)
        auc_roc = self.get_auc_roc()
        confusion_matrix = self.get_confusion_matrix(threshold)
        print(f"Accuracy at threshold {threshold}: {accuracy}")
        print(f"Balanced ccuracy at threshold {threshold}: "
              f"{balanced_accuracy}")
        print(f"Confusion matrix at threshold {threshold}: {confusion_matrix}")
        print(f"AUC-ROC: {auc_roc}")
