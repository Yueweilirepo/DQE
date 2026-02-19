#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.metrics import roc_curve, auc, precision_recall_curve,average_precision_score,roc_auc_score
import numpy as np
import matplotlib.pyplot as plt


def compute_auc(y_true, scores, plot_flag=False):
    auc_roc = roc_auc_score(y_true, scores)

    if plot_flag:
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        # fpr:0-1
        # tpr:0-1
        # thresholds:inf-0

        show_pic = True

        algorithm_str = "Algorithm"
        # algorithm_str = "Algorithm 1"
        # algorithm_str = "Algorithm 2"

        idx_first = np.argmax(tpr == 1.0)
        fpr_star = fpr[idx_first]
        tpr_star = tpr[idx_first]
        t_star = thresholds[idx_first]

        font_size = 20
        plt.figure(figsize=(6, 5))
        plt.fill_between(fpr, tpr, alpha=0.25, color='tab:orange')
        plt.plot(fpr, tpr, label=f'ROC curve (AUC={auc_roc:0.3f})')

        label_str = f'FPR={fpr_star:.3f}, TPR={tpr_star:.3f}, thresh={t_star:.3f}'
        plt.scatter(fpr_star, tpr_star, color='red', s=60, zorder=5,
                    label=label_str)

        plt.plot([0, 1], [0, 1], ls='--', c='grey')
        plt.xlabel('False Positive Rate', fontsize= font_size)
        plt.ylabel('True Positive Rate', fontsize= font_size)
        plt.title('ROC Curve ('+ algorithm_str +')', fontsize= font_size)
        plt.tick_params(axis='both', labelsize=font_size)

        plt.legend(fontsize= 12)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if show_pic:
            plt.show()

    return auc_roc

# Compute the AUPRC for the model
def compute_auprc(y_true, scores, plot_flag=False):

    ap = average_precision_score(y_true, scores)

    if plot_flag:
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        # precision：0->1
        # recall：1->0
        # thresh：0->1

        show_pic = True

        algorithm_str = "Algorithm"
        # algorithm_str = "Algorithm 1"
        # algorithm_str = "Algorithm 2"

        thresh_rev = np.r_[1.0, thresholds[::-1]]
        precision_rev = precision[::-1]
        recall_rev = recall[::-1]

        idx_first_one = np.argmax(recall_rev == 1.0)
        p_star = precision_rev[idx_first_one]
        r_star = recall_rev[idx_first_one]
        t_star = thresh_rev[idx_first_one]

        font_size = 20
        plt.figure(figsize=(6, 5))

        plt.fill_between(recall, precision, alpha=0.25, color='tab:blue')  # 关键行

        plt.plot(recall, precision, label=f'PR curve (AP={ap:0.3f})')

        plt.scatter(r_star, p_star, color='red', s=60, zorder=5,
                    label=f'P={p_star:.3f}, R={r_star:.3f}, thresh={t_star:.3f}')

        plt.xlabel('Recall', fontsize= font_size)
        plt.ylabel('Precision', fontsize= font_size)
        plt.title('Precision-Recall Curve ('+ algorithm_str +')', fontsize= font_size)
        plt.legend(fontsize= 12)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if show_pic:
            plt.show()

    return ap


def compute_auprc_backup(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    auc_pr_score = auc(recall, precision)
    return auc_pr_score
