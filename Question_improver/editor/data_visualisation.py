"""Visualisations."""
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix,
    average_precision_score, accuracy_score, f1_score, brier_score_loss)


def plot_embeddings(df: pd.DataFrame,
                    dimen: tuple,
                    labels: str = None,
                    palette=['r', 'g', 'b', 'y', 'k', 'm'],
                    **kwargs):
    """
    Create and show 2d projection of the embedding.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data to display.
    dimen : (str, str)
        Tuple of column names to use as x and y dimension.
    labels : str, 1d-array like
        Column name with the labels, or a 1d-array containing labels.
    palette: dict, list
        Either a list of colors to be assigned to labels, or a dictionary dire-
        ctly mapping labels to specific colors.

    Returns
    -------
    None.
        Plots figures.

    """
    warnings.filterwarnings('ignore', category=UserWarning)
    plt.figure(dpi=150)
    sns.scatterplot(df,
                    x=dimen[0], y=dimen[1],
                    hue=labels,
                    alpha=0.4, s=1,
                    palette=palette,
                    **kwargs)
    plt.show()
    plt.close()


def plot_calibration_curves(train_prob_pred, val_prob_pred,
                            train_true, val_true):
    """
    Return plot of calibration curves for test set and train set.

    Parameters
    ----------
    train_prob_pred : 1-d array like
        Predicted probabilities for the train set.
    val_prob_pred : 1-d array like
        Predicted probabilities for the validation set.
    train_true : 1-d array like
        True class labels for the train set.
    val_true : 1-d array like
        True class labels for the validation set.

    Returns
    -------
    None.

    """
    sns.set_theme(style='darkgrid', palette='bright')
    fig = plt.figure(figsize=(4, 6), dpi=150, layout='constrained')
    subplots = fig.subfigures(2, 1, height_ratios=[2, 1])

    columns = ['Assigned probability', 'Fraction of positives']

    # create calibration data for train set
    mean_pred_value, frac_posit = calibration_curve(train_true,
                                                    train_prob_pred,
                                                    n_bins=20)
    train_plot_dat = pd.DataFrame(
        np.vstack((mean_pred_value,frac_posit)).transpose(),
        columns=columns
    )
    train_brier = brier_score_loss(train_true, train_prob_pred)
    del mean_pred_value, frac_posit
    train_plot_dat = train_plot_dat.sort_values(columns[0])

    # create calibration data for validation set
    mean_pred_value, frac_posit = calibration_curve(
        val_true, val_prob_pred, n_bins=50
    )
    val_plot_dat = pd.DataFrame(
        np.vstack((mean_pred_value, frac_posit)).transpose(),
        columns=columns
    )
    val_brier = brier_score_loss(val_true, val_prob_pred)
    del mean_pred_value, frac_posit
    val_plot_dat = val_plot_dat.sort_values(columns[0])
    # plot calibration curves
    ax = subplots[0].subplots(1, 1)
    ax.plot(val_plot_dat[columns[0]], val_plot_dat[columns[1]],
            'b', label=f'val\n br = {val_brier:.3f}'
            )
    ax.plot(train_plot_dat[columns[0]], train_plot_dat[columns[1]],
            'g', label=f'train\n br = {train_brier:.3f}'
            )
    ax.legend(loc='best')
    ax.set_xlim(-.02, 1.02)
    ax.set_ylim(-.02, 1.02)
    ax.vlines(0.5, -0.2, 1.02, 'r', linestyle=':')
    ax.set_xlabel('Assigned probability')
    ax.set_ylabel('Fraction of positives')
    ax.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), 'k:')

    # plot histograms
    ax = subplots[1].subplots(1, 1)
    sns.histplot(
        val_prob_pred,
        ax=ax,
        stat='percent',
        bins=np.linspace(0, 1, 50),
        alpha=0.4, color='b'
    )
    sns.histplot(
        train_prob_pred,
        ax=ax,
        stat='percent',
        bins=np.linspace(0, 1, 50),
        alpha=0.4, color='g'
    )
    ax.set_xlim(-.02, 1.02)
    ax.set_xlabel('Assigned probability')

    plt.show()
    plt.close(fig)


def plot_ROC_PRC_conf(
    predict_prob, true_y, thresh: float = 0.5, sup_tit: str = None):
    """
    Plot ROC curve, PR curve and confusion matrix along with some metrics.

    Parameters
    ----------
    predict_prob : 1d array like
        Predicted probabilities of categories.
    true_y : 1d array like
        True class labels.
    thresh : float
        Decision threshold for classification.

    Returns
    -------
    None.

    """
    sns.set_theme(style='darkgrid', palette='bright')
    fig, axs = plt.subplots(1, 3,
                            layout='constrained',
                            figsize=(10, 4),
                            dpi=150
                            )

    # Convert predictions to 0/1 labels
    predicted_ys = (predict_prob > thresh).astype(int)
    # accuracy, f1, conf_matrix
    acc = accuracy_score(true_y, predicted_ys)
    f1 = f1_score(true_y, predicted_ys)
    conf_matr = confusion_matrix(true_y, predicted_ys)
    conf_matr = pd.DataFrame(
        conf_matr,
        columns=['predicted\nbad', 'predicted\ngood'],
        index=['bad', 'good']
    )
    if isinstance(sup_tit, str):
        fig.suptitle(
            f'{sup_tit}   –   '
            f'threshold = {thresh};   acc = {acc:.3f};   '
            f'f1 = {f1:.3f}'
    )
    else:
        fig.suptitle(
            f'threshold = {thresh};   acc = {acc:.3f};   '
            f'f1 = {f1:.3f}'
    )

    # ROC curve
    axs[0].set_title('ROC curve')
    fpr, tpr, thresholds = roc_curve(true_y, predict_prob)
    auc = roc_auc_score(true_y, predict_prob)
    axs[0].plot(fpr, tpr, 'r', label='AUC   =  '+f'{auc:.3f}'.strip('0'))
    # optim_thresh
    thr_idx = np.argmin(np.abs(thresholds - thresh))
    axs[0].plot(
        fpr[thr_idx],
        tpr[thr_idx],
        'ro',
        markersize=6,
        label=(
            'TPR    = '+f'{tpr[thr_idx]:.3f}\n'.strip('0') +
            'FPR    = '+f'{fpr[thr_idx]:.3f}'.strip('0')
        )
    )
    axs[0].plot(np.linspace(0, 1, 20), np.linspace(0, 1, 20), 'k:')
    # visuals
    axs[0].legend(loc='lower right')
    axs[0].set_xlabel('FPR')
    axs[0].set_ylabel('TPR')

    # precision recall curve
    # f1 score contours
    array_0_1 = np.linspace(0.01, 1, 100)
    f1_grid = np.empty((100, 100))
    for i, a in enumerate(array_0_1):
        for ii, b in enumerate(array_0_1):
            f1_grid[i, ii] = 2*a*b/(a+b)
    grid = np.meshgrid(array_0_1, array_0_1)
    f1_sc = axs[1].contour(
        grid[0], grid[1], f1_grid,
        levels=np.arange(0, 1, 0.1),
        linestyles='dotted', colors='k'
    )
    axs[1].clabel(f1_sc, inline=True, fontsize=8)
    
    # PRC
    axs[1].set_title('PR curve')
    prec, rec, thresholds = precision_recall_curve(true_y, predict_prob)
    auc = average_precision_score(true_y, predict_prob)
    axs[1].plot(rec, prec, 'r',
                label='AUC       = '+f'{auc:.3f}'.strip('0'))
    # optim thresh
    thr_idx = np.argmin(np.abs(thresholds - thresh))
    axs[1].plot(rec[thr_idx], prec[thr_idx], 'ro', markersize=6,
                label='recall    = '+f'{rec[thr_idx]:.3f}\n'.strip('0') +
                'precision = '+f'{prec[thr_idx]:.3f}'.strip('0'))
    # visuals
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')
    axs[1].legend(loc='lower left')

    # plot confusion matrix
    axs[2].set_title('Confusion matrix')
    sns.heatmap(conf_matr, cmap=['w'],
                ax=axs[2], cbar=False, square=True,
                annot=True, fmt='.0f', annot_kws={'fontsize': 15})
    axs[2].vlines(1, 0, 2, 'k')
    axs[2].hlines(1, 0, 2, 'k')

    for ax in axs[0:1]:
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    plt.show()
    plt.close()


def plot_train_feature_correlation(train_data):
    """Plot correlation heatmap of training data.

    Args:
        train_data (2D array): Training data of vectorised features to be
        plotted.
    """
    plt.figure(figsize=(10, 10), dpi=150)
    sns.heatmap(train_data.corr())
    plt.show()
    plt.close()


def plot_model_feature_importances(model, feature_names):
    """Plot feature importance largest to smallest.

    Args:
        model (_type_): model, has to support sklearn's .feature_importances_
        method
        feature_names (_type_): list of names of the features
    """
    importances = pd.Series(
        model.feature_importances_,
        index=feature_names
    )
    importances = importances.sort_values(ascending=False)
    height = len(importances)/5
    plt.figure(figsize=(5,height), dpi=150)
    sns.barplot(
        importances,
        orient='h'
    )
    plt.show()
    plt.close()
