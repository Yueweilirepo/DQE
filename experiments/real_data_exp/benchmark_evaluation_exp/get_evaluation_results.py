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


if __name__ == '__main__':
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running DQE real-world experiments')
    args = parser.parse_args()

    dataset_dir = "../../../dataset/"
    res_dir = "../../../results/TSB_AD/"

    method_pred_file_dir = res_dir + "methods_pred_res/"
    ori_data_dir = dataset_dir + "TSB-AD-U/"
    res_save_dir = res_dir + "metric_cal_res_windows/"
    single_file_res_save_dir = res_dir + "single_file_evaluation_res/"

    methods_list = ['SR', 'KMeansAD_U', 'Sub_KNN', 'TimesNet', 'CNN', 'Sub_LOF', 'FFT']

    file_path_dict = {}
    dataset_methods_file_list = os.listdir(method_pred_file_dir)
    for file_name in dataset_methods_file_list:
        file_path_dict[file_name] = method_pred_file_dir + file_name

    method_num = len(methods_list)

    data_set_choose_file_list = []

    dataset_name_list = [
                         'WSD',
                         'YAHOO',
                         'UCR'
    ]

    file_method_metric_dict = {} # save
    for i, dataset_file_name in enumerate(dataset_methods_file_list):

        # dataset filter
        dataset_name = dataset_file_name.split("_")[1]
        if dataset_name not in dataset_name_list:
            continue

        file_method_metric_dict[dataset_file_name] = {}

        data_set_choose_file_path = file_path_dict[dataset_file_name]
        with open(data_set_choose_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        for j, method_name in enumerate(methods_list):
            methods_choose_outputs = data[method_name]

            gt_array = data["gt"]
            gt_range = data["gt_range"]

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


            metric_score_dict,metrics_consume_time_dict = get_metrics(output_array, label, slidingWindow=slidingWindow, thre=100)

            file_method_metric_dict[dataset_file_name][method_name] = metric_score_dict

        # save single file evaluate res
        single_file_res = {}
        single_file_res[dataset_file_name] = file_method_metric_dict[dataset_file_name]


    res_save_path = res_save_dir + "metric_cal_res_all_files" +".json"

    with open(res_save_path, 'w', encoding='utf-8') as json_file:
        json.dump(file_method_metric_dict, json_file, indent=4, ensure_ascii=False)
    print(f"Results are saved to {res_save_path}")
