import numpy as np


def get_precision(x):
    return x[1, 1]/(x[1, 1] + x[0, 1])


def get_recall(x):
    return x[1, 1] / (x[1, 0] + x[1, 1])


def get_f_score(class_table, beta=1):
    precision = get_precision(class_table)
    recall = get_recall(class_table)
    return ((1 + beta**2)*precision*recall)/(beta**2 * precision + recall)


class MetricsHandler:
    METRICS_LAMBDAS = {
        "Precision": lambda x: get_precision(x),
        "Recall": lambda x: get_recall(x),
        "F1-score": lambda x: get_f_score(x),
        "F0.5-score": lambda x: get_f_score(x, beta=0.5)
    }

    def __init__(self, classes):
        self.metrics_dict = {k: [] for k in MetricsHandler.METRICS_LAMBDAS.keys()}
        self.classes_prediction_table = {x: [np.zeros((2,2))] for x in classes}

    def _calculate_metrics(self):
        for metric, calc in MetricsHandler.METRICS_LAMBDAS.items():
            values = [calc(x[-1]) for _, x in self.classes_prediction_table.items()]
            self.metrics_dict[metric].append(np.nanmean(values))

    def collect(self):
        self._calculate_metrics()
        for key in self.classes_prediction_table.keys():
            self.classes_prediction_table[key].append(np.zeros((2,2)))

    @staticmethod
    def calculate_prediction_table(prediction, true_values, label):
        table = np.zeros((2, 2))
        for i in range(len(prediction)):
            if prediction[i] != label and true_values[i] != label:
                table[0][0] += 1
            if prediction[i] != label and true_values[i] == label:
                table[1, 0] += 1
            if prediction[i] == label and true_values[i] == label:
                table[1, 1] += 1
            if prediction[i] == label and true_values[i] != label:
                table[0, 1] += 1
        return table

    def update(self, prediction, true_values):
        for key in self.classes_prediction_table.keys():
            self.classes_prediction_table[key][-1] += MetricsHandler.calculate_prediction_table(prediction,
                                                                                                true_values,
                                                                                                key)

    def get_metrics(self):
        return self.metrics_dict
