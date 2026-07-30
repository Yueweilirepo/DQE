import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd
import argparse


from evaluation.metrics import get_metrics
from evaluation.slidingWindows import find_length_rank


def create_path(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        dir_path = os.path.dirname(path) if os.path.splitext(path)[1] else path
        os.makedirs(dir_path, exist_ok=True)

def convert_vector_to_events_dqe(vector):
    """
    Convert a binary anomaly vector into a list of half-open intervals.

    Each value `1` at index *i* is treated as an anomalous segment `[i, i+1)`.

    Parameters
    ----------
    vector : list[int] | np.ndarray
        Binary sequence containing only 0 or 1.

    Returns
    -------
    list[list[float]]
        List of `[start, end)` couples describing the detected events.
    """
    events = []
    event_start = None
    for i, val in enumerate(vector):
        if val == 1:
            if event_start is None:
                event_start = i
        else:
            if event_start is not None:
                events.append((event_start, i))
                event_start = None
    if event_start is not None:
        events.append((event_start, len(vector)))
    return events


if __name__ == '__main__':
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running DQE real-world experiments')
    parser.add_argument('--print', type=bool, default=True)
    parser.add_argument('--test_time', type=bool, default=True)
    args = parser.parse_args()


    ratio_list = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2]
    dataset_dir = "../../../dataset/"
    res_dir = "../../../results/TSB_AD/"
    method_pred_file_dir = dataset_dir + "methods_pred_res/"
    ori_data_dir = dataset_dir + "TSB-AD-U/"

    exp_list = ratio_list

    for param_vlue in exp_list:
        dataset_res_root_dir = res_dir + "dataset_nm_size_exp" + "_" + str(param_vlue).split(".")[0] + "_" + str(param_vlue).split(".")[1] + "/"

        create_path(dataset_res_root_dir)

        res_save_dir = dataset_res_root_dir + "metric_cal_res_windows/"
        create_path(res_save_dir)
        single_file_res_save_dir = dataset_res_root_dir + "single_file_evaluation_res/"
        create_path(single_file_res_save_dir)
        single_file_consume_time_save_dir = dataset_res_root_dir + "single_file_consume_time/"
        create_path(single_file_consume_time_save_dir)

        methods_list = ['KMeansAD_U', 'TimesNet', 'CNN', 'Sub_LOF', 'FFT']

        tuning_data = pd.read_csv("dataset/TSB-AD-U-Tuning.csv")
        tuning_data_list = tuning_data.to_numpy().flatten().tolist()
        tuning_data_index_list = []
        for item in tuning_data_list:
            tuning_data_index_list.append(item.split("_")[0])


        file_path_dict = {}
        full_dataset_methods_file_list = os.listdir(method_pred_file_dir)
        dataset_methods_file_list = []
        for file_name in full_dataset_methods_file_list:
            if file_name.split("_")[0] in tuning_data_index_list:
                continue
            dataset_methods_file_list.append(file_name)

        for file_name in dataset_methods_file_list:
            file_path_dict[file_name] = method_pred_file_dir + file_name

        method_num = len(methods_list)

        data_set_choose_file_list = []

        dataset_name_list = [
                             'WSD',
                             # 'YAHOO',
                             'UCR'
        ]

        file_method_metric_dict = {} # save
        file_method_metric_consume_time_dict = {} # save
        dqe_list = []
        for i, dataset_file_name in enumerate(dataset_methods_file_list):
            # dataset filter
            dataset_name = dataset_file_name.split("_")[1]
            if dataset_name not in dataset_name_list:
                continue

            single_file_res_save_path = single_file_res_save_dir + dataset_name + "/" + dataset_file_name
            single_file_consume_time_save_path = single_file_consume_time_save_dir + dataset_name + "/" + dataset_file_name

            file_method_metric_dict[dataset_file_name] = {}
            file_method_metric_consume_time_dict[dataset_file_name] = {}

            data_set_choose_file_path = file_path_dict[dataset_file_name]
            with open(data_set_choose_file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            gt_array = data["gt"]
            gt_range = data["gt_range"]

            for j, method_name in enumerate(methods_list):
                methods_choose_outputs = data[method_name]

                # cal score for all metric
                csv_file_name = dataset_file_name.split(".")[0].replace("_method_pred_scaled", "") + ".csv"

                ori_data_file_path = ori_data_dir + csv_file_name
                df = pd.read_csv(ori_data_file_path).dropna()

                train_index = dataset_file_name.split('.')[0].split('_')[-3]

                ori_data = df.iloc[:, 0:-1].values.astype(float)
                label = df['Label'].astype(int).to_numpy()
                slidingWindow = find_length_rank(ori_data, rank=1)

                output = methods_choose_outputs
                output_array = np.array(output)

                metric_list = [
                    'VUS-ROC',
                    'VUS-PR',
                    'PATE',
                    "DQE",
                ]

                metric_score_dict,metrics_consume_time_dict = get_metrics(output_array, label, slidingWindow=slidingWindow, thre=100, metric_list=metric_list, ratio=param_vlue)

                if args.print:
                    print(" metric_score_dict",metric_score_dict)

                file_method_metric_dict[dataset_file_name][method_name] = metric_score_dict
                file_method_metric_consume_time_dict[dataset_file_name][method_name] = metrics_consume_time_dict

                dqe_list.append(metric_score_dict["dqe"])

                # save one file evaluation result across all methods
                create_path(single_file_res_save_path)
                create_path(single_file_consume_time_save_path)

                # save single file evaluate res
                single_file_res = {}
                single_file_res[dataset_file_name] = file_method_metric_dict[dataset_file_name]
                single_file_time_res = {}
                single_file_time_res[dataset_file_name] = file_method_metric_consume_time_dict[dataset_file_name]


                # save
                if not os.path.isfile(single_file_res_save_path):
                    with open(single_file_res_save_path, 'w', encoding='utf-8') as json_file:
                        json.dump(single_file_res, json_file, indent=4, ensure_ascii=False)
                    with open(single_file_consume_time_save_path, 'w', encoding='utf-8') as json_file:
                        json.dump(single_file_time_res, json_file, indent=4, ensure_ascii=False)


        res_save_path = res_save_dir + "metric_cal_res_all_files" +".json"

        with open(res_save_path, 'w', encoding='utf-8') as json_file:
            json.dump(file_method_metric_dict, json_file, indent=4, ensure_ascii=False)
        if args.print:
            print(f"Results are saved to {res_save_path}")

        res_save_path = res_save_dir + "metric_cal_time_res_all_files" +".json"

        with open(res_save_path, 'w', encoding='utf-8') as json_file:
            json.dump(file_method_metric_consume_time_dict, json_file, indent=4, ensure_ascii=False)
        if args.print:
            print(f"Calculation time results are saved to {res_save_path}")
