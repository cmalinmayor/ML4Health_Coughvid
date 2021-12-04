import numpy as np
import sklearn.metrics as metrics
import logging
import csv
import os
from datetime import datetime


logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, path_to_logs=None, model_name='resnet_18', threshold='best'):
        self.path_to_logs = path_to_logs
        self.str_date_time = datetime.now().strftime("%d%m%Y%H%M%S")
        if self.path_to_logs is None:
            self.path_to_logs = self.str_date_time + model_name
        os.makedirs(self.path_to_logs, exist_ok=True)
        self.model_name = model_name
        self.prc_file = 'prc.csv'
        self.roc_file = 'roc.csv'
        self.epoch_stats_file = 'stats.csv'
        self.threshold = threshold
        self.labels = None
        self.predictions = None
        self.f1 = None

    def get_accuracy(self):
        predicted_labels = np.where(self.predictions > self.threshold, 1, 0)
        return metrics.accuracy_score(self.labels, predicted_labels)

    def get_balanced_accuracy(self):
        predicted_labels = np.where(self.predictions > self.threshold, 1, 0)
        return metrics.balanced_accuracy_score(self.labels, predicted_labels)

    def get_auc_roc(self):
        return metrics.roc_auc_score(self.labels, self.predictions)

    def get_roc(self):
        return metrics.roc_curve(self.labels, self.predictions)

    def get_prc(self):
        return metrics.precision_recall_curve(self.labels, self.predictions)

    def get_confusion_matrix(self):
        predicted_labels = np.where(self.predictions > self.threshold, 1, 0)
        return metrics.confusion_matrix(self.labels, predicted_labels)

    def get_best_threshold(self):
        precision, recall, thresholds = metrics.precision_recall_curve(
                self.labels, self.predictions)
        f1_scores = 2*recall*precision/(recall+precision)
        best_threshold = thresholds[np.argmax(f1_scores)]
        best_f1_score = np.max(f1_scores)
        logger.info(f"Best threshold is at {best_threshold} "
                    f"with f1-score {best_f1_score}")
        return best_threshold, best_f1_score

    def evaluate_epoch(self, epoch, labels, predictions, train_acc):
        self.epoch = epoch
        self.train_acc = train_acc
        self.labels = np.array(labels)
        self.predictions = np.array(predictions)
        if self.threshold == 'best':
            self.threshold, self.f1 = self.get_best_threshold()
        self.test_acc = self.get_accuracy()
        self.test_balanced_acc = self.get_balanced_accuracy()
        self.auc_roc = self.get_auc_roc()
        self.confusion_matrix = self.get_confusion_matrix()
        self.tn = self.confusion_matrix[0][0]
        self.fp = self.confusion_matrix[0][1]
        self.fn = self.confusion_matrix[1][0]
        self.tp = self.confusion_matrix[1][1]
        self.print_report()
        self.save_epoch_stats()

    def print_report(self):
        print(f"Report for epoch {self.epoch}")
        print(f"Train accuracy: {self.train_acc}")
        print(f"Accuracy at threshold {self.threshold}: {self.test_acc}")
        print(f"Balanced accuracy at threshold {self.threshold}: "
              f"{self.test_balanced_acc}")
        print(f"Confusion matrix at threshold {self.threshold}: {self.confusion_matrix}")
        print(f"AUC-ROC: {self.auc_roc}")

    def save_epoch_stats(self):
        stats_rows = ['epoch', 'train_acc', 'threshold', 'test_acc', 'test_balanced_acc', 'auc_roc', 'tp', 'fp', 'tn', 'fn']
        row = [self.epoch, self.train_acc, self.threshold, self.test_acc, self.test_balanced_acc, self.auc_roc, self.tp, self.fp, self.tn, self.fn]
        fname = os.path.join(self.path_to_logs, self.epoch_stats_file)
        with open(fname, 'a') as f:
            csvwriter = csv.writer(f)
            if self.epoch == 1:
                csvwriter.writerow(stats_rows)
            csvwriter.writerow(row)
        self.save_roc()
        self.save_prc()

    def save_roc(self):
        fpr, tpr, threshold = self.get_roc()
        fname = os.path.join(self.path_to_logs, f"{self.epoch}_{self.roc_file}")

        with open(fname, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['FPR', 'TPR', 'threshold'])
            for (f, t, thr) in zip(fpr, tpr, threshold):
                csvwriter.writerow([f, t, thr])

    def save_prc(self):
        p, r, threshold = self.get_prc()
        fname = os.path.join(self.path_to_logs, f"{self.epoch}_{self.prc_file}")

        with open(fname, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Precision', 'Recall', "threshold"])
            for (f, t, thr) in zip(p, r, threshold):
                csvwriter.writerow([f, t, thr])
