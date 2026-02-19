#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###########################
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import json
import argparse
import numpy as np
import copy

from metrics.pate.PATE_utils import convert_events_to_array_PATE, convert_vector_to_events_PATE
from experiments.synthetic_data_exp.utils_synthetic_exp import evaluate_all_metrics


class CustomEncoder(json.JSONEncoder):
    def encode(self, obj):
        if isinstance(obj, list):
            items = [self.encode(item) for item in obj]
            return "[\n" + ",\n".join(items) + "\n]"
        elif isinstance(obj, dict):
            items = [f'"{key}":{self.encode(value)}' for key, value in obj.items()]
            return "{" + ",".join(items) + "}"
        return super().encode(obj)

if __name__ == '__main__':
    # ArgumentParser
    parser = argparse.ArgumentParser(description='Running DQE synthetic data experiments')
    parser.add_argument('--exp_name', type=str, default='anomaly_event_coverage')
    args = parser.parse_args()

    paint_name_list = [
        'original_F1Score',
        'AUC',
        'AUC_PR',

        'pa_k_score',

        'VUS_ROC',
        'VUS_PR',
        'PATE',

        'Rbased_f1score',
        'eTaPR_f1_score',
        'Affliation F1score',

        "dqe",
    ]

    plot_paper_pic_path = "paper/src/figures/single_prediction_figures/"

    name_dict = {
        "original_F1Score": "Original-F",
        "AUC": "AUC-ROC",
        "AUC_PR": "AUC-PR",

        "pa_k_score": "PA-K",

        "VUS_ROC": "VUS-ROC",
        "VUS_PR": "VUS-PR",
        "PATE": "PATE",

        "Rbased_f1score": "RF",
        "eTaPR_f1_score": "eTaF",
        "Affliation F1score": "AF",

        "dqe": "DQE",
    }


    choose_metric_name_order_list = [
            "Original-F",
            "AUC-ROC",
            "AUC-PR",

            "PA-K",

            "VUS-ROC", "VUS-PR", "PATE",

            "RF", "eTaF", "AF",

            "DQE",
        ]

    # anomaly event coverage
    if args.exp_name == "anomaly_event_coverage":
        json_file_name = "anomaly_event_coverage"

        anomaly_num = 5
        anomaly_point_len = 40
        anomaly_ratio = 0.1

        window_length = 2050
        label_ranges = [
            [[322, 361], [663, 702], [1004, 1043], [1345, 1384], [1686, 1725]],
            [[322, 361]],
            [[341, 341], [682, 682], [1023, 1023], [1364, 1364], [1705, 1705]]
        ]
        vus_zone_size = e_buffer = d_buffer = near_single_side_range = 10

    # near-miss proximity
    if args.exp_name == "near_miss_proximity":
        json_file_name = "near_miss_proximity"

        window_length = 300
        label_ranges = [[[100, 119]], [[120, 121]], [[125, 126]], [[130, 131]], [[135, 136]]]
        vus_zone_size = e_buffer = d_buffer = near_single_side_range = 20

    # proximity inconsistency
    if args.exp_name == "proximity_inconsistency":
        json_file_name = "proximity_inconsistency"

        window_length = 110
        label_ranges = [[[49, 51]], [[51, 51]], [[51, 52]], [[51, 53]], [[51, 56]], [[51, 59]]]
        vus_zone_size = e_buffer = d_buffer = near_single_side_range = 10

    # proximity inconsistency (af)
    if args.exp_name == "proximity_inconsistency_af":
        json_file_name = "proximity_inconsistency_af"
        vus_zone_size = e_buffer = d_buffer = near_single_side_range = 3

        window_length = 38
        label_ranges = [[[29, 30], [35, 36]], [[26, 27], [35, 36]], [[29, 30], [34, 35]]]

    # false alarm frequency
    if args.exp_name == "false_alarm_frequency":
        json_file_name = "false_alarm_frequency"

        window_length = 300
        label_ranges = [[[140, 159]], [[149, 150], [226, 233]], [[149, 150], [227, 227], [231, 231], [223, 223], [235, 235], [219, 219], [239, 239], [215, 215], [243, 243]]]
        vus_zone_size = e_buffer = d_buffer = near_single_side_range = 20

    # random case
    if args.exp_name == "random_case":
        json_file_name = "random_case"
        window_length = 1000
        label_ranges = [
            [[window_length // 2 - 10, window_length // 2 + 10]],
        ]
        vus_zone_size = e_buffer = d_buffer = near_single_side_range = 20

        np.random.seed(42)
        random_array = np.random.randint(0, 2, size=window_length)
        random_label_interval_ranges = convert_vector_to_events_PATE(random_array)
        label_ranges.append(random_label_interval_ranges)

    ordered_num = len(label_ranges)

    label_array_list = []
    res_data = []

    for i, single_range in enumerate(label_ranges):
        if i >= 0 and i < ordered_num:
            label_array = convert_events_to_array_PATE(single_range, time_series_length=window_length)

            label_array_copy = copy.deepcopy(label_array)
            label_array_list.append(label_array_copy)

            score_list_simple = evaluate_all_metrics(label_array,
                                                     label_array_list[0],
                                                     vus_zone_size,
                                                     e_buffer,
                                                     d_buffer,
                                                     near_single_side_range=near_single_side_range,
                                                     )

            selected_dict = {}

            for j, paint_name in enumerate(paint_name_list):
                value = score_list_simple[paint_name]
                selected_dict[paint_name] = value
            res_data.append(selected_dict)

    file_path = "../../../results/synthetic_exp_result/case_evaluation_res/synthetic_data_res_" + json_file_name + ".json"

    res_data_dict = {}

    new_res_data = []
    for i, pred_score_dict in enumerate(res_data):
        new_score_dict = {}
        for j, (metric_name, metric_score) in enumerate(pred_score_dict.items()):
            if name_dict[metric_name] in choose_metric_name_order_list:
                new_score_dict[name_dict[metric_name]] = metric_score
        new_res_data.append(new_score_dict)

    with open(file_path, "w", encoding="utf-8") as file:
        encoder = CustomEncoder()
        json_str = encoder.encode(new_res_data)
        file.write(json_str)


    reorder_new_res_data = []
    for i, pred_score_dict in enumerate(new_res_data):
        reorder_single_dict = {}
        for j, choose_metric_name in enumerate(choose_metric_name_order_list):
            find_pred = pred_score_dict[choose_metric_name]

            reorder_single_dict[choose_metric_name] = find_pred
        reorder_new_res_data.append(reorder_single_dict)

    new_res_data = reorder_new_res_data

    new_res_data = new_res_data[1:]

    df = pd.DataFrame(new_res_data)

    print("new_res_data", new_res_data)
