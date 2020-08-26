# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import matplotlib.colors as colors
import matplotlib.cm as cmx
from sklearn import manifold


def plot_2d(x, y=None, label_dict=None, size=20, marker='o', show_ticks=False, show_legend=False, save_path='',
            show_fig=True):
    """
    Plot 2D data.
    :param x: 2D data, (x, y). ndarray, shape: [num_samples, 2].
    :param y: Labels of the data. Optional, if not provided, the color of all points will be black.
              ndarray, shape: [num_samples, ].
    :param label_dict: Strings for each label, shown in the legend. Optional.
    :param size: Size of the points.
    :param marker: Marker of the points.
    :param show_ticks: Whether to show the ticks.
    :param show_legend: Whether to show the legend.
    :param save_path: The path to save the figure.
    :param show_fig: Whether to show the figure.
    """
    x = np.asarray(x)
    assert len(x.shape) == 2
    if y is not None:
        y = np.asarray(y)
        assert len(y.shape) == 1

    fig = plt.figure()

    if not show_ticks:
        plt.xticks([])
        plt.yticks([])

    if y is None:
        plt.scatter(x[:, 0], x[:, 1], s=size, c='k', marker=marker)
    else:
        jet = plt.get_cmap('jet')
        y_mapping = {i: label_idx for i, label_idx in enumerate(set(y))}

        c_norm = colors.Normalize(vmin=0, vmax=len(y_mapping) - 1)
        scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=jet)

        for i, label_idx in y_mapping.items():
            color_val = scalar_map.to_rgba(i)
            selected = x[y == label_idx]
            label = label_idx if label_dict is None else label_dict[label_idx]
            plt.scatter(selected[:, 0], selected[:, 1], s=size, c=[color_val], marker=marker, label=label)
        if show_legend:
            plt.legend(loc='upper right')

    if save_path:
        plt.savefig(save_path, dpi=1000, bbox_inches='tight')

    if show_fig:
        plt.show()

    plt.close(fig)


def tsne(x, y=None, label_dict=None, size=20, marker='o', show_ticks=False, show_legend=False, save_path='',
         show_fig=True):
    """
    Visualize data with t-SNE.
    :param x: Data. ndarray, shape: [num_samples, num_features].
    :param y: Labels of the data. Optional, if not provided, the color of all points will be black.
              ndarray, shape: [num_samples, ].
    :param label_dict: Strings for each label, shown in the legend. Optional.
    :param size: Size of the points.
    :param marker: Marker of the points.
    :param show_ticks: Whether to show the ticks.
    :param show_legend: Whether to show the legend.
    :param save_path: The path to save the figure.
    :param show_fig: Whether to show the figure.
    :return: Embeddings of `x`.
    """
    embedding = manifold.TSNE(n_components=2).fit_transform(x)
    plot_2d(embedding, y=y, label_dict=label_dict, size=size, marker=marker, show_ticks=show_ticks,
            show_legend=show_legend, save_path=save_path, show_fig=show_fig)
    return embedding
