import json
import os
import numpy as np
import pandas as pd
import time
import argparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler


from evaluation.metrics import get_metrics
from evaluation.slidingWindows import find_length_rank


if __name__ == '__main__':
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running DQE real-world experiments (case analysis)')
    parser.add_argument('--exp_name', type=str, default='WSD case')
    args = parser.parse_args()

    dataset_dir = "../../../dataset/"
    res_dir = "../../../results/TSB_AD/"

    file_path = res_dir + "methods_pred_res/"

    ori_data_path = dataset_dir + "TSB-AD-U/"

    res_save_dir = res_dir + "metric_cal_res_windows/"

    dataset_methods_choose_name_list = ['CNN', 'Sub_LOF', 'TimesNet', 'FFT', 'KMeansAD_U']

    if args.exp_name == "WSD case":
        # WSD case
        json_file = "094_WSD_id_66_WebService_tr_3309_1st_3914_method_pred_scaled.json"
        dataset_methods_file_list = [json_file]
    elif args.exp_name == "UCR case":
        json_file = "452_UCR_id_150_Facility_tr_10000_1st_35774_method_pred_scaled.json"
        dataset_methods_file_list = [json_file]
    # elif args.exp_name == "AUC-ROC/AUC-PR issue case":
    else:
        json_file = "808_YAHOO_id_258_WebService_tr_500_1st_142_method_pred_scaled.json"

        dataset_methods_file_list = [json_file]
        dataset_methods_choose_name_list = ['KMeansAD_U', 'SR']

    file_msg = json_file.split("_")[0] + "_" +json_file.split("_")[1]+ "_case_analyze"

    file_path_dict = {}
    for file_name in dataset_methods_file_list:
        file_path_dict[file_name] = file_path+file_name

    file_method_metric_dict = {} # save

    thresh_num_list = [
                            25,
                            50,
                            100,
                            500,
                            1000,
                        ]



    sort_res = []
    thresh_num_str_list = []

    for thresh_num in thresh_num_list:
        thresh_num_str_list.append(str(thresh_num))
        for i, data_set_choose_file in enumerate(dataset_methods_file_list):
            # new data
            data_set_choose_file = data_set_choose_file.replace("_name_list","_method_pred_scaled")
            csv_file_name = data_set_choose_file.split(".")[0].replace("_method_pred_scaled", "") + ".csv"

            dataset_name= data_set_choose_file.split("_")[1]
            file_method_metric_dict[data_set_choose_file] = {}
            data_set_choose_file_path = file_path_dict[data_set_choose_file]
            with open(data_set_choose_file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            for j, dataset_methods_choose_name in enumerate(dataset_methods_choose_name_list):

                gt_array = data["gt"]
                gt_range = data["gt_range"]

                if dataset_methods_choose_name == "R":
                    methods_choose_outputs = np.random.uniform(0,1,len(gt_array))
                else:
                    methods_choose_outputs = data[dataset_methods_choose_name]

                # cal score for all metric
                ori_data_file_path = ori_data_path + csv_file_name.replace("_name_list.json",".csv")
                df = pd.read_csv(ori_data_file_path).dropna()

                train_index = data_set_choose_file.split('.')[0].split('_')[-3]


                ori_data = df.iloc[:, 0:-1].values.astype(float)
                label = df['Label'].astype(int).to_numpy()

                if args.exp_name == "AUC-ROC/AUC-PR issue case":
                    test_start = 900
                    test_end = 1050
                    label = label[test_start:test_end]
                    ori_data = ori_data[test_start:test_end]

                slidingWindow = find_length_rank(ori_data, rank=1)

                output = methods_choose_outputs
                output_array = np.array(output)

                if args.exp_name == "AUC-ROC/AUC-PR issue case":
                    output_array = output_array[test_start:test_end]
                    output_array = MinMaxScaler(feature_range=(0, 1)).fit_transform(
                        output_array.reshape(-1, 1)).ravel().tolist()

                metric_list = [
                                  "DQE",
                                ]
                metric_score_dict,_ = get_metrics(output_array, label, slidingWindow=slidingWindow, thre=100, exp_name=args.exp_name, metric_list=metric_list, thresh_num=thresh_num)

                file_method_metric_dict[data_set_choose_file][dataset_methods_choose_name] = metric_score_dict


        methods = next(iter(file_method_metric_dict.values()))

        ranking_results = {}

        metrics = next(iter(methods.values())).keys()

        for metric in metrics:
            ranking = sorted(
                methods.items(),
                key=lambda x: x[1][metric],
                reverse=True
            )

            ranking_results[metric] = [
                {
                    "method": method,
                    "score": scores[metric]
                }
                for method, scores in ranking
            ]


        metric_rank_list = [
            "dqe",
        ]

        name_dict = {
            "dqe":"DQE",
            }

        metric_rank_rename_list = [name_dict[item] for item in metric_rank_list]

        ranking_results = {
            metric: ranking_results[metric]
            for metric in metric_rank_list
        }

        table = []



        for metric, ranking in ranking_results.items():

            row = [
                f'{item["score"]:.2f} ({item["method"]})'
                for item in ranking
            ]
            table.append(row)

        dqe_methods = [
            item["method"]
            for item in ranking_results["dqe"]
        ]

        new_metric_name_list = dqe_methods


        sort_res.append(table[0])

    print(sort_res)
