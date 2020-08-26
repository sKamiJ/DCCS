# -*- coding: UTF-8 -*-

# scikit-learn: 0.21.3

from sklearn import metrics


def accuracy(y_pred, y_true, mask=None, percentage=True):
    """
    Calculate the multi-class classification accuracy.
    :param y_pred: Predicted labels of samples. ndarray, shape: [num_samples, ].
    :param y_true: Ground truth labels of samples. ndarray, shape: [num_samples, ].
    :param mask: Masks of samples. 1 means valid, 0 means invalid. Optional, ndarray, shape: [num_samples, ].
    :param percentage: Return accuracy as a percentage.
    :return: Accuracy.
    """
    acc = metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=mask)
    if percentage:
        acc *= 100
    return acc


def nmi(y_pred, y_true, average_method='geometric'):
    """
    Calculate the normalized mutual information.
    :param y_pred: Predicted labels of samples, do not need to map to the ground truth label.
                   ndarray, shape: [num_samples, ].
    :param y_true: Ground truth labels of samples. ndarray, shape: [num_samples, ].
    :param average_method: How to compute the normalizer in the denominator. Possible options
                           are 'min', 'geometric', 'arithmetic', and 'max'.
                           'min': min(U, V)
                           'geometric': np.sqrt(U * V)
                           'arithmetic': np.mean([U, V])
                           'max': max(U, V)
    :return: Normalized mutual information.
    """
    return metrics.normalized_mutual_info_score(y_true, y_pred, average_method=average_method)


def ari(y_pred, y_true):
    """
    Calculate the adjusted rand index.
    :param y_pred: Predicted labels of samples, do not need to map to the ground truth label.
                   ndarray, shape: [num_samples, ].
    :param y_true: Ground truth labels of samples. ndarray, shape: [num_samples, ].
    :return: Adjusted rand index.
    """
    return metrics.adjusted_rand_score(y_true, y_pred)
