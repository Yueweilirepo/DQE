import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import json
import copy

from metrics.pate.PATE_utils import convert_events_to_array_PATE
from experiments.synthetic_data_exp.utils_synthetic_exp import evaluate_all_metrics, create_path


class CustomEncoder(json.JSONEncoder):
    def encode(self, obj):
        if isinstance(obj, list):
            items = [self.encode(item) for item in obj]
            return "[\n" + ",\n".join(items) + "\n]"
        elif isinstance(obj, dict):
            items = [f'"{key}":{self.encode(value)}' for key, value in obj.items()]
            return "{" + ",".join(items) + "}"
        return super().encode(obj)

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

dataset_dir = "dataset/"

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


upper_limit_res = []
upper_limit_x = []

score_gage_dict_res = []

anomaly_num_max = 1000+1

separate_point = 100

step_after = 100

anomaly_num_list = list(range(1, separate_point + 1)) \
       + list(range(separate_point + step_after, anomaly_num_max, step_after))

json_file_name = "event_anomaly_coverage_anomaly_num"

for anomaly_num in anomaly_num_list:

    vus_zone_size = e_buffer = d_buffer = near_single_side_range = 10

    anomaly_point_len = 10

    anomaly_ratio = 0.01

    window_length = int(anomaly_num*(anomaly_point_len+1)/anomaly_ratio)
    part_space = window_length//(anomaly_num+1)
    label_ranges = []
    gt_ranges = []
    for i in range(anomaly_num):
        gt_ranges.append([(i + 1) * part_space - (anomaly_point_len + 1) // 2 + 1,
                          (i + 1) * part_space - (anomaly_point_len + 1) // 2 + anomaly_point_len])
    label_ranges.append(gt_ranges)

    for i in range(anomaly_num):
        if not (i == 0 or i == anomaly_num - 1):
            continue
        pred_ranges = []
        for j in range(i+1):
            pred_ranges.append([(j+1)*part_space,(j+1)*part_space])
        label_ranges.append(pred_ranges)


    choose_metric_name_order_list = [
        "Original-F", "AUC-ROC", "AUC-PR",
        "PA-K",
        "VUS-ROC", "VUS-PR", "PATE",
        "RF", "eTaF", "AF",
        "DQE",
    ]

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

    file_path = dataset_dir + "synthetic_exp_result/single_res/anomaly_num/synthetic_data_res_" + str(anomaly_num) + "_" + json_file_name + ".json"
    create_path(file_path)

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

    first_detection_res = new_res_data[0]
    last_detection_res = new_res_data[-1]
    score_gage_dict = {}
    for metric_name in first_detection_res.keys():
        score_gage_dict[metric_name] = last_detection_res[metric_name] - first_detection_res[metric_name]

    score_gage_dict_res.append(score_gage_dict)


    upper_limit_line = new_res_data[-1]

    upper_limit_x.append(anomaly_num)
    upper_limit_res.append(upper_limit_line)


info = {
    "upper_limit_x":upper_limit_x,
    "upper_limit_res":upper_limit_res,
    "score_gage_dict_res":score_gage_dict_res,
}

res_save_dir = dataset_dir + "synthetic_exp_result/exp_res_data/" + "synthetic_exp_res_" + str(anomaly_num_max) + "_"+ json_file_name + ".json"
create_path(res_save_dir)
res_save_dir = res_save_dir.replace(" ", "_")


with open(res_save_dir, 'w', encoding='utf-8') as json_file:
    json.dump(info, json_file, indent=4, ensure_ascii=False)
print(f"write to {res_save_dir}")