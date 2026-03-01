import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import argparse
import json

from dqe.dqe_metric import write_json, DQE_multi_data

if __name__ == '__main__':
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running DQE real-world experiments (anomaly event level)')
    parser.add_argument('--exp_name', type=str, default='WSD')
    parser.add_argument('--file_index', type=str, default='')
    parser.add_argument('--print', type=bool, default=True)
    parser.add_argument('--test_time', type=bool, default=True)
    args = parser.parse_args()


    methods_list = ['SR', 'CNN', 'Sub_LOF', 'FFT', 'KMeansAD_U', 'Sub_KNN', 'TimesNet']

    dataset_name_list = [
        # 'WSD',
        # 'YAHOO',
        # 'UCR',
        args.exp_name,
    ]

    dataset_dir = "../../../dataset/"
    res_dir = "../../../results/TSB_AD/"

    method_pred_file_dir = res_dir + "methods_pred_res/"
    ori_data_dir = dataset_dir + "TSB-AD-U/"

    # get multi ts for cal period
    ts_dict = {}
    # get all gt of multi ts from one method
    gt_dict = {}
    ori_file_list = os.listdir(ori_data_dir)

    method_dqe_res_dict = {}

    for ori_file in ori_file_list:
        file_index = ori_file.split("_")[0]
        dataset_name = ori_file.split("_")[1]
        # file filter
        if args.file_index != '':
            if file_index != args.file_index:
                continue

        # dataset filter
        dataset_name = ori_file.split("_")[1]
        if dataset_name not in dataset_name_list:
            continue

        ori_data_file_path = ori_data_dir + ori_file
        df = pd.read_csv(ori_data_file_path).dropna()

        ori_data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()
        ts_dict[ori_file] = ori_data
        gt_dict[ori_file] = label


    for idz,method_name in enumerate(methods_list):
        # get all output of multi ts from one method
        output_dict = {}
        for idx,(dataset_file,ori_ts) in enumerate(ts_dict.items()):
            method_pred_file_name = dataset_file.split(".")[0] + "_method_pred_scaled" + ".json"
            method_pred_file_path = method_pred_file_dir + method_pred_file_name
            with open(method_pred_file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            output_dict[dataset_file] = data[method_name]

        # cal dqe for multi ts
        if args.file_index != '':
            deq_res = DQE_multi_data(ts_dict, output_dict, gt_dict, thresh_num=100, cal_components=True, method_name=method_name, return_each_anomaly_score=True)
        else:
            deq_res = DQE_multi_data(ts_dict, output_dict, gt_dict, thresh_num=100, cal_components=True, method_name=method_name)
        method_dqe_res_dict[method_name] = deq_res

    dataset_save_str = dataset_name_list[0]
    for dataset_name in dataset_name_list[1:]:
        dataset_save_str+=("_"+dataset_name)
    multi_ts_res_save_path = res_dir + "multi_ts_res/" + "multi_ts_res_" + dataset_save_str + args.file_index+ ".json"
    write_json(multi_ts_res_save_path,method_dqe_res_dict)
