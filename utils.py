# -*- coding: UTF-8 -*-

import logging
import numpy as np
from scipy import optimize
import torch
import torch.nn as nn
import torch.autograd as autograd


def hungarian_match(cluster_assignments, y_true, num_clusters):
    """
    Find the best match between the cluster assignments and the ground truth labels (one-to-one mapping).
    :param cluster_assignments: Cluster assignments of samples. ndarray, shape: [num_samples, ].
    :param y_true: Ground truth labels of samples. ndarray, shape: [num_samples, ].
    :param num_clusters: Number of the clusters.
    :return: A list of tuples, cluster assignment to ground truth label.
    """
    assert isinstance(cluster_assignments, np.ndarray)
    assert isinstance(y_true, np.ndarray)
    assert cluster_assignments.shape[0] == y_true.shape[0]

    num_samples = cluster_assignments.shape[0]
    num_correct = np.zeros((num_clusters, num_clusters), dtype=np.int32)

    for c1 in range(num_clusters):
        for c2 in range(num_clusters):
            # elementwise, so each sample contributes once
            votes = ((cluster_assignments == c1) * (y_true == c2)).sum(dtype=np.int32)
            num_correct[c1, c2] = votes

    # convert the maximization problem to a minimization problem, num_correct is small
    row_id, col_id = optimize.linear_sum_assignment(num_samples - num_correct)

    # return as list of tuples, cluster assignment to ground truth label
    match = list()
    for i in range(num_clusters):
        assignment = row_id[i]
        gt = col_id[i]
        match.append((assignment, gt))
    return match


def convert_cluster_assignment_to_ground_truth(cluster_assignments, match):
    """
    Convert the cluster assignments to the ground truth labels.
    :param cluster_assignments: Cluster assignments of samples. ndarray, shape: [num_samples, ].
    :param match: The match between the cluster assignments and the ground truth labels.
    :return: Mapped ground truth labels of samples. ndarray, shape: [num_samples, ].
    """
    assert isinstance(cluster_assignments, np.ndarray)

    mapped = np.empty_like(cluster_assignments)
    for assignment, gt in match:
        mapped[cluster_assignments == assignment] = gt
    return mapped


def save_model(model, path):
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def load_model(model, path, strict=True):
    state_dict = torch.load(path)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict, strict=strict)
    else:
        model.load_state_dict(state_dict, strict=strict)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # If `val` is an instance of torch.Tensor, then the variables generated when calculating `val`
        # during the forward propagation may be accumulated to `sum`, resulting in memory leaks.
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def sample_z(batch_size=64, dim_zs=30, dim_zc=10, fix_class=-1, zs_std=0.1):
    assert (fix_class == -1 or (fix_class >= 0 and fix_class < dim_zc))

    # Sample zs
    zs = zs_std * np.random.randn(batch_size, dim_zs)

    # Sample zc
    if fix_class == -1:
        zc_idx = np.random.randint(low=0, high=dim_zc, size=(batch_size,), dtype=np.int64)
    else:
        zc_idx = np.ones(shape=(batch_size,), dtype=np.int64) * fix_class
    zc = np.eye(dim_zc)[zc_idx]

    return zs, zc, zc_idx


def calc_gradient_penalty(critic, real_data, generated_data, lambda_gp):
    # Calculate interpolation
    b_size = real_data.size(0)
    shape = [b_size] + [1] * (real_data.dim() - 1)
    alpha = torch.rand(shape, dtype=torch.float32, device=real_data.device)

    interpolated = alpha * real_data.detach() + (1 - alpha) * generated_data.detach()
    interpolated.requires_grad_(True)

    # Calculate scores of interpolated examples
    score_interpolated = critic(interpolated)

    # Calculate gradients of scores with respect to examples
    gradients = autograd.grad(outputs=score_interpolated, inputs=interpolated,
                              grad_outputs=torch.ones_like(score_interpolated),
                              create_graph=True, retain_graph=True)[0]

    # Flatten to easily take norm per example in batch
    gradients = gradients.view(b_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return lambda_gp * ((gradients_norm - 1) ** 2).mean()


def create_logger(fp):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
    console_logger = logging.getLogger('ConsoleLoggoer')
    file_logger = logging.getLogger('FileLogger')

    file_handler = logging.FileHandler(fp, mode='a', encoding='utf-8')
    file_logger.addHandler(file_handler)
    return console_logger, file_logger
