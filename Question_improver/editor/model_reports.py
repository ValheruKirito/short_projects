"""Functions to improve on iterations between model versions."""
import numpy as np
import pandas as pd
import umap
import seaborn as sns
import matplotlib.pyplot as plt


def get_k_examples(pred_prob: pd.Series, true_y: pd.Series, k: int = 5):
    """
    Return indexes of k worst, k best and k most confusing cases.

    Parameters
    ----------
    pred_prob : 1d array-like
        Predicted probabilities for individual cases.
    true_y : 1d array-like
        True class labels for individual cases.
    k : int, optional
        Number of cases to look for. The default is 5.

    Returns
    -------
    k_pos_best, k_pos_worst, k_neg_best, k_neg_worst, most_conf
        Returns three lists inlcuding indexes of the k positive cases that are
        classified best, k positive cases that are classified worst, k negative
        cases that are classified best, k negative cases that are classified
        worst, and k most confusing cases.

    """
    if isinstance(pred_prob, pd.Series) and isinstance(true_y, pd.Series):
        # actually positive cases predicted probs sorted high to low
        pos = pred_prob[true_y == 1].sort_values(ascending=False)
        # highest probs are the ones most certain
        k_pos_best = pos[:k].index
        # last postitions are the ones least certain
        k_pos_worst = pos[-k:].index

        # actually negative cases predicted probs sorted high to low
        neg = pred_prob[true_y == 0].sort_values(ascending=False)
        # highest probs are the ones least certain (prob ~ to 1 is labeled 1)
        k_neg_worst = neg[:k].index
        # last postitions are the ones most certain
        k_neg_best = neg[-k:].index

        # series of the most confusing cases (~ to 0.5) sorted low to high
        most_conf = (pred_prob - 0.5).abs(
            ).sort_values(ascending=True)[:k].index
        return k_pos_best, k_pos_worst, k_neg_best, k_neg_worst, most_conf

    else:
        raise TypeError(pd.Series)


def plot_mislabeled_data(df: pd.DataFrame,
                         y_pred_prob: pd.Series,
                         y_true: pd.Series
                         ):
    """
    Plot 2D embedding of the features categorised by their mis/labeling.

    Parameters
    ----------
    df : pd.DataFrame
        Vectorised features.
    y_pred_prob : pd.Series
        Predicted probabilities for individual cases.
    y_true : pd.Series
        True class labels for individual cases.

    Returns
    -------
    None.

    """
    sns.set_theme(style='ticks', palette='bright')
    umab_emb = umap.UMAP().fit_transform(df)
    plot = pd.DataFrame(columns=['umap_x', 'umap_y', 'conf'])
    plot['umap_x'], plot['umap_y'] = umab_emb[:, 0], umab_emb[:, 1]

    plot['conf'] = np.where((y_true == 1) & (y_pred_prob >= 0.5),
                            'TP', plot['conf'])
    plot['conf'] = np.where((y_true == 1) & (y_pred_prob < 0.5),
                            'FN', plot['conf'])
    plot['conf'] = np.where((y_true == 0) & (y_pred_prob >= 0.5),
                            'FP', plot['conf'])
    plot['conf'] = np.where((y_true == 0) & (y_pred_prob < 0.5),
                            'TN', plot['conf'])

    plt.figure(dpi=150)
    sns.scatterplot(plot, x='umap_x', y='umap_y', hue='conf',
                    palette=['r', 'g', 'b', 'orange'],
                    s=1, alpha=0.2)
    plt.show()
    plt.close()
