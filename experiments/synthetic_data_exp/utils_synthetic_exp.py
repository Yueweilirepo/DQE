#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

from dqe.dqe_metric import DQE
from metrics.metrics_pa import PointAdjustKPercent
from metrics.pate.PATE_metric import PATE
from metrics.pate.PATE_utils import convert_events_to_array_PATE, categorize_predicted_ranges_with_ids, \
    convert_vector_to_events_PATE
from evaluation.metrics import basic_metricor, generate_curve

def create_path(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        dir_path = os.path.dirname(path) if os.path.splitext(path)[1] else path
        os.makedirs(dir_path, exist_ok=True)

def evaluate_all_metrics(pred, labels, vus_zone_size=20, e_buffer=20, d_buffer=20, near_single_side_range=125):
    window_length = len(labels)
    grader = basic_metricor()

    # Affliation
    Affiliation_F = grader.metric_Affiliation(labels, score=pred, preds=pred)


    # Vus
    _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(label=labels, score=pred, \
                                                       slidingWindow=vus_zone_size)

    # Auc
    AUC_ROC = grader.metric_ROC(labels, score=pred)
    AUC_PR = grader.metric_PR(labels, score=pred)

    # PATE
    pate = PATE(labels, pred, e_buffer, d_buffer, n_jobs=1, num_splits_MaxBuffer=1, include_zero=False,
                binary_scores=False)
    # R-based
    RF1 = grader.metric_RF1(labels, score=pred, preds=pred)

    # eTaPR
    eTaPR_F1 = grader.metric_eTaPR_F1(labels, score=pred, preds=pred)

    # Standard Original F1-Score
    PointF1 = grader.metric_PointF1(labels, score=pred, preds=pred)

    # DQE
    dqe_res_ts = DQE(labels,pred,
                     near_single_side_range=near_single_side_range,
                     cal_components=True)

    labels_ranges = convert_vector_to_events_PATE(labels)
    pred_ranges = convert_vector_to_events_PATE(pred)

    # %K-PA-F
    pa_k = PointAdjustKPercent(window_length, labels_ranges, pred_ranges)
    pa_k_score = pa_k.get_score()

    score_list_simple = {
        "original_F1Score": PointF1,
        "AUC": AUC_ROC,
        "AUC_PR": AUC_PR,

        "pa_k_score": pa_k_score,

        "VUS_ROC": VUS_ROC,
        "VUS_PR": VUS_PR,
        "PATE": pate,

        "Rbased_f1score": RF1,
        "eTaPR_f1_score": eTaPR_F1,
        "Affliation F1score": Affiliation_F,

        "dqe": dqe_res_ts["dqe"],
    }

    for key in score_list_simple:
        score_list_simple[key] = round(score_list_simple[key], 2)

    return score_list_simple


def synthetic_generator(label_anomaly_ranges, predicted_ranges, vus_zone_size=20, e_buffer=20, d_buffer=20,
                        time_series_length=500):
    """
    Runs a synthetic data experiment given label and prediction ranges.

    Parameters:
    - label_anomaly_ranges: List of [start, end] ranges for actual anomalies.
    - predicted_ranges: List of [start, end] ranges for detected anomalies.
    - time_series_length: Total length of the time series.
    - vus_zone_size: Size of the VUS method buffer zone.
    - e_buffer: Eaely prebuffer size.
    - d_buffer: Delayed postbuffer size.

    Returns:
    - A dictionary containing the categorized ranges with IDs, predicted array, and label array.
    """
    categorized_ranges_with_ids = categorize_predicted_ranges_with_ids(
        predicted_ranges, label_anomaly_ranges, e_buffer, d_buffer, time_series_length)

    predicted_array = convert_events_to_array_PATE(predicted_ranges, time_series_length)
    label_array = convert_events_to_array_PATE(label_anomaly_ranges, time_series_length)

    return {
        "categorized_ranges_with_ids": categorized_ranges_with_ids,
        "predicted_array": predicted_array,
        "label_array": label_array
    }




