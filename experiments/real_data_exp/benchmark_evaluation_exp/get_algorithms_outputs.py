import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import torch
import random, argparse
import json


from sklearn.preprocessing import MinMaxScaler
from experiments.TSB_AD.model_wrapper import *
from experiments.TSB_AD.HP_list import Optimal_Uni_algo_HP_dict, Optimal_Multi_algo_HP_dict
from experiments.TSB_AD.utils.plot_utils import *


def custom_json_dumps(obj, indent=2):
    if isinstance(obj, list):
        items = []
        for item in obj:
            if isinstance(item, list):
                item_str = json.dumps(item, separators=(',', ':'))
            else:
                item_str = custom_json_dumps(item, indent)
            items.append(item_str)
        indent_str = ' ' * indent
        items_str = ',\n'.join([f'{indent_str}{item}' for item in items])
        return f'[\n{items_str}\n]'
    elif isinstance(obj, dict):
        items = []
        for key, value in obj.items():
            value_str = custom_json_dumps(value, indent)
            items.append(f'"{key}": {value_str}')
        indent_str = ' ' * indent
        items_str = ',\n'.join([f'{indent_str}{item}' for item in items])
        return f'{{\n{items_str}\n}}'
    else:
        return json.dumps(obj)


def custom_json_dumps_only_list(obj, indent=2):
    if isinstance(obj, dict):
        items = []
        for key, value in obj.items():
            value_str = custom_json_dumps(value, indent)
            if isinstance(value, list):
                value_str = json.dumps(value, separators=(',', ':'))
            items.append(f'"{key}": {value_str}')
        indent_str = ' ' * indent
        items_str = ',\n'.join([f'{indent_str}{item}' for item in items])
        return f'{{\n{items_str}\n}}'
    else:
        return json.dumps(obj)


# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("CUDA Available: ", torch.cuda.is_available())
print("cuDNN Version: ", torch.backends.cudnn.version())

if __name__ == '__main__':
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running TSB-AD')
    args = parser.parse_args()

    filter_ad_pool = ['SR', 'KMeansAD_U', 'Sub_KNN', 'TimesNet', 'CNN', 'Sub_LOF', 'FFT']

    dataset_choose_list = [
        "WSD",
        "YAHOO",
        "UCR",
    ]

    filename_list = []

    df_exp_files_all = pd.read_csv("../../../dataset/File_List/exp_file_list_all.csv").dropna()

    all_list = df_exp_files_all.squeeze().tolist()
    add_list = []
    for file in all_list:
        add_info_file = file.split(".")[0] + "_method_pred_scaled.json"
        add_list.append(file)

    df_exp_files_add = pd.DataFrame(add_list, columns=["FileName"])

    df_exp_files_array = df_exp_files_add.iloc[:].values
    df_exp_files_list = df_exp_files_array.reshape(df_exp_files_array.shape[0])

    directory_path = "../../../dataset/TSB-AD-U/"
    file_list_ndarray = df_exp_files_add[:].values
    file_array = file_list_ndarray.reshape(file_list_ndarray.shape[0])

    dataset_choose_list = [
        "WSD",
        "YAHOO",
        "UCR",
    ]

    cal_model_num = len(filter_ad_pool)

    json_file_save_dir = "../../../results/TSB_AD/methods_pred_res/"

    for i, file_name in enumerate(file_array):
        add_info_file_name = file_name.split(".")[0] + "_method_pred_scaled.json"

        df = pd.read_csv(directory_path + file_name).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()
        slidingWindow = find_length_rank(data, rank=1)
        train_index = file_name.split('.')[0].split('_')[-3]
        data_train = data[:int(train_index), :]

        range_anomaly = range_convers_new(label)

        label_ranges = [range_anomaly]
        label_array_list = [label]

        for idx, ad_name in enumerate(filter_ad_pool[:cal_model_num]):
            if ad_name in Optimal_Uni_algo_HP_dict.keys():
                Optimal_Det_HP = Optimal_Uni_algo_HP_dict[ad_name]
            else:
                Optimal_Det_HP = Optimal_Multi_algo_HP_dict[ad_name]

            if ad_name in Semisupervise_AD_Pool:
                output = run_Semisupervise_AD(ad_name, data_train, data, **Optimal_Det_HP)
            elif ad_name in Unsupervise_AD_Pool:
                output = run_Unsupervise_AD(ad_name, data, **Optimal_Det_HP)
            else:
                raise Exception(f"{ad_name} is not defined")

            output = MinMaxScaler(feature_range=(0, 1)).fit_transform(output.reshape(-1, 1)).ravel()
            label_array_list.append(output)

        json_file_save_path = json_file_save_dir + file_name.split('.')[0] + "_method_pred_scaled" + ".json"

        label_array_list_to_list = []
        for label_array in label_array_list:
            label_array_list_to_list.append(label_array.tolist())

        data_res_only_list = {}

        method_name_list = ["gt"] + filter_ad_pool

        data_res_only_list["gt_range"] = label_ranges

        choose_num = cal_model_num + 1
        for j, method_name in enumerate(method_name_list[:choose_num]):
            data_res_only_list[method_name] = label_array_list_to_list[j]

        json_str_only_list = custom_json_dumps_only_list(data_res_only_list, indent=2)

        with open(json_file_save_path, "w") as json_file:
            json_file.write(json_str_only_list)

    print(f"Results are saved to {json_file_save_path}")
