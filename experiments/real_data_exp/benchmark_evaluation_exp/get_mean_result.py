import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import copy


dataset_dir = "../../../dataset/"
res_dir = "../../../results/TSB_AD/"

file_path = res_dir + "methods_pred_res/"
ori_data_path = dataset_dir + "TSB-AD-U/"
res_save_dir = res_dir + "metric_cal_res_windows/"

dataset_methods_name_list = ['SR', 'KMeansAD_U', 'Sub_KNN', 'TimesNet', 'CNN', 'Sub_LOF', 'FFT']

file_path_dict = {}
dataset_methods_file_list = os.listdir(file_path)
for file_name in dataset_methods_file_list:
    file_path_dict[file_name] = file_path + file_name

choose_method_num = len(dataset_methods_name_list)

dataset_methods_choose_name_list = dataset_methods_name_list

data_set_choose_file_list = []

dataset_name_list = [
    'WSD',
    'YAHOO',
    'UCR',
]
choose_method_num = str(len(dataset_methods_name_list))

dataset_methods_choose_name_list = ['KMeansAD_U', 'TimesNet', 'CNN', 'Sub_LOF', 'FFT']

method_name_index_dict = {}
for i, method_name in enumerate(dataset_methods_choose_name_list):
    method_name_index_dict[method_name] = i

print("method_name_index_dict")
print(method_name_index_dict)

# integral every dataset
dataset_name_score_list_dict = {
    'YAHOO': {},
    'WSD': {},
    'UCR': {},
}

dataset_name_list = list(dataset_name_score_list_dict.keys())

metric_res_file_path = res_dir+"metric_cal_res_windows/metric_cal_res_all_files.json"

with open(metric_res_file_path, "r", encoding="utf-8") as file:
    metric_res_data = json.load(file)

file_method_metric_dict = metric_res_data
print("len(file_method_metric_dict)",len(file_method_metric_dict))

# cal mean res
# cal number each dataset
dataset_count_dict = {}
for id_dataset_name, dataset_name in enumerate(file_method_metric_dict.keys()):
    dataset_name_to_find = dataset_name.split("_")[1]
    if dataset_name_to_find not in dataset_count_dict:
        dataset_count_dict[dataset_name_to_find] =0

    dataset_count_dict[dataset_name_to_find] +=1

print("dataset_count_dict",dataset_count_dict)

metric_name_list = [
    'Standard-F1',
    'AUC-ROC',
    'AUC-PR',

    'PA-K',

    'VUS-ROC',
    'VUS-PR',
    'PATE',

    'R-based-F1',
    'eTaPR_F1',
    'Affiliation-F',

    "dqe", # DQE
]

# create mean score data structure for single dataset(dataset, metric, method)
for id_dataset_name, dataset_name in enumerate(dataset_name_score_list_dict.keys()):
    metric_score_list_dict = {}

    for id_metric_name, metric_name in enumerate(metric_name_list):

        method_score_list_dict = {}

        for id_method_name, method_name in enumerate(dataset_methods_choose_name_list):
            method_score_list_dict[method_name] = {"score_list":[],"mean_score":None}

        metric_score_list_dict[metric_name] = method_score_list_dict

    dataset_name_score_list_dict[dataset_name] = metric_score_list_dict

# create mean score data structure for all datasets(metric, method)
metric_score_list_dict = {}

for id_metric_name, metric_name in enumerate(metric_name_list):

    method_score_list_dict = {}

    for id_method_name, method_name in enumerate(dataset_methods_choose_name_list):
        method_score_list_dict[method_name] = {"score_list":[],"mean_score":None}

    metric_score_list_dict[metric_name] = method_score_list_dict

# cal dataset mean score for every metric
# add to list
# target order, dataset,metric, method
for id_dataset_name_need, dataset_name_need in enumerate(dataset_name_list):
    for id_metric_name_need, metric_name_need in enumerate(metric_name_list):
        for id_method_choose_need, method_name_need in enumerate(dataset_methods_choose_name_list):
            # have order, dataset, method, metric
            for id_file_to_find, (dataset_to_find,dataset_dict) in enumerate(file_method_metric_dict.items()):
                dataset_name_to_find = dataset_to_find.split("_")[1]
                if dataset_name_need == dataset_name_to_find:
                    for id_method_to_find, (method_name_to_find, method_dict) in enumerate(dataset_dict.items()):
                        if method_name_need == method_name_to_find:
                            for id_metric_to_find, (metric_name_to_find,metric_dict) in enumerate(method_dict.items()):
                                if metric_name_need == metric_name_to_find:
                                    # find
                                    find_score = method_dict[metric_name_to_find]
                                    import math
                                    if isinstance(find_score, float) and math.isnan(find_score):
                                        find_score = 0
                                    dataset_name_score_list_dict[dataset_name_need][metric_name_need][method_name_need]["score_list"].append(find_score)

for id_metric_name_need, metric_name_need in enumerate(metric_name_list):
    for id_method_choose_need, method_name_need in enumerate(dataset_methods_choose_name_list):
        # have order, dataset, method, metric
        for id_file_to_find, (dataset_file_name,dataset_dict) in enumerate(file_method_metric_dict.items()):
            for id_method_to_find, (method_name_to_find, method_dict) in enumerate(dataset_dict.items()):
                if method_name_need == method_name_to_find:
                    for id_metric_to_find, (metric_name_to_find,metric_dict) in enumerate(method_dict.items()):
                        if metric_name_need == metric_name_to_find:
                            # find
                            find_score = method_dict[metric_name_to_find]
                            import math

                            if isinstance(find_score, float) and math.isnan(find_score):
                                find_score = 0
                            metric_score_list_dict[metric_name_need][method_name_need]["score_list"].append(find_score)



# cal mean score for single dataset
for id_dateset_name, (dateset_name, dateset_score_dict) in enumerate(dataset_name_score_list_dict.items()):
    for id_metric_name, (metric_name,metric_score_dict) in enumerate(dateset_score_dict.items()):
        for id_method_name, (method_name,method_score_info_dict) in enumerate(metric_score_dict.items()):
            score_list = dataset_name_score_list_dict[dateset_name][metric_name][method_name]["score_list"]

            mean_score = np.array(score_list).mean()
            dataset_name_score_list_dict[dateset_name][metric_name][method_name]["mean_score"] = mean_score

            # cal range
            max_score = np.array(score_list).max()
            min_score = np.array(score_list).min()
            median_score = np.median(score_list)

            dataset_name_score_list_dict[dateset_name][metric_name][method_name]["max_score"] = float(max_score)
            dataset_name_score_list_dict[dateset_name][metric_name][method_name]["min_score"] = float(min_score)
            dataset_name_score_list_dict[dateset_name][metric_name][method_name]["median_score"] = float(median_score)
            dataset_name_score_list_dict[dateset_name][metric_name][method_name]["score_range"] = float(max_score - min_score)


# cal mean score for all datasets
for id_metric_name, (metric_name,metric_score_dict) in enumerate(metric_score_list_dict.items()):
    for id_method_name, (method_name,method_score_info_dict) in enumerate(metric_score_dict.items()):
        score_list = metric_score_list_dict[metric_name][method_name]["score_list"]
        mean_score = np.array(score_list).mean()
        metric_score_list_dict[metric_name][method_name]["mean_score"] = mean_score


# creating rankings by sort for single dataset
dataset_name_score_list_dict_sort = copy.deepcopy(dataset_name_score_list_dict)  # copy structure and change dict item
for id_dateset_name, (dateset_name, dateset_score_dict) in enumerate(dataset_name_score_list_dict_sort.items()):
    # for a dataset
    for id_metric_name, (metric_name,metric_score_dict) in enumerate(dateset_score_dict.items()):
        # for a metric,sort method
        method_score_dict_list= []
        for id_method_name, (method_name,method_score_res_dict) in enumerate(metric_score_dict.items()):
            method_score_dict_list.append({"method_name":method_name,
                                           "method_index":method_name_index_dict[method_name],
                                           "mean_score":method_score_res_dict["mean_score"],
                                           })
        method_score_dict_list_sorted = sorted(method_score_dict_list, key=lambda x: x["mean_score"], reverse=True)

        # dict->list
        dataset_name_score_list_dict_sort[dateset_name][metric_name] = method_score_dict_list_sorted


# creating rankings by sort for all datasets
metric_score_list_dict_sort = copy.deepcopy(metric_score_list_dict)  # copy structure and change dict item
for id_metric_name, (metric_name,metric_score_dict) in enumerate(metric_score_list_dict_sort.items()):
    # for a metric,sort method
    method_score_dict_list= []
    for id_method_name, (method_name,method_score_res_dict) in enumerate(metric_score_dict.items()):
        method_score_dict_list.append({"method_name":method_name,
                                       "method_index":method_name_index_dict[method_name],
                                       "mean_score":method_score_res_dict["mean_score"],
                                       })
    method_score_dict_list_sorted = sorted(method_score_dict_list, key=lambda x: x["mean_score"], reverse=True)

    # dict->list
    metric_score_list_dict_sort[metric_name] = method_score_dict_list_sorted


# save rankings
dataset_name_score_list_dict_copy_index_dict_sort = copy.deepcopy(dataset_name_score_list_dict_sort)  # copy structure and simplify ranking info item
metric_score_list_dict_copy_index_dict_sort = copy.deepcopy(metric_score_list_dict_sort)

print("single dataset rankings result")

for id_dateset_name, (dateset_name, dateset_score_dict) in enumerate(dataset_name_score_list_dict_sort.items()):
    # for a dataset
    # find case

    for id_metric_name, (metric_name,metric_score_dict_list) in enumerate(dateset_score_dict.items()):
        add_info_list = []
        sort_index_info_list = []
        for idx, metric_score_dict in enumerate(metric_score_dict_list):
            metric_score_dict["sort_id"] = idx
            add_info_list.append(metric_score_dict)

            sort_index_info_list.append(
                str(metric_score_dict["sort_id"]) \
                + " " + "m:" + str(metric_score_dict["method_index"]) \
                + " " + str(metric_score_dict["method_name"]) \
                + " " + str(round(metric_score_dict["mean_score"],2)) \
                )

        dataset_name_score_list_dict_sort[dateset_name][metric_name] = add_info_list

        dataset_name_score_list_dict_copy_index_dict_sort[dateset_name][metric_name] = sort_index_info_list


for id_metric_name, (metric_name,metric_score_dict_list) in enumerate(metric_score_list_dict_sort.items()):
    add_info_list = []
    sort_index_info_list = []

    for idx, metric_score_dict in enumerate(metric_score_dict_list):
        metric_score_dict["sort_id"] = idx
        add_info_list.append(metric_score_dict)

        sort_index_info_list.append(
            str(metric_score_dict["sort_id"]) \
            + " " + "m:" + str(metric_score_dict["method_index"]) \
            + " " + str(metric_score_dict["method_name"]) \
            + " " + str(round(metric_score_dict["mean_score"],2)) \
            )

    metric_score_list_dict_sort[metric_name] = add_info_list

    metric_score_list_dict_copy_index_dict_sort[metric_name] = sort_index_info_list


# patch all datasets result
dataset_name_score_list_dict_sort["all_dataset_mean"] = metric_score_list_dict_sort
dataset_name_score_list_dict_copy_index_dict_sort["all_dataset_mean"] = metric_score_list_dict_copy_index_dict_sort


def custom_json_formatter(data, indent=4):
    def _format(value, level=0):
        if isinstance(value, list):
            elements = [json.dumps(item, ensure_ascii=False) for item in value]
            max_length = max(len(item) for item in elements)
            formatted_elements = ", ".join(item.ljust(max_length) for item in elements)
            return f"[{formatted_elements}]"
        elif isinstance(value, dict):
            lines = []
            for key, val in value.items():
                lines.append(" " * (indent * level) + f'"{key}": {(_format(val, level + 1))}')
            return "{\n" + ",\n".join(lines) + "\n" + " " * (indent * (level - 1)) + "}"
        else:
            return json.dumps(value, ensure_ascii=False)

    return _format(data)

res_save_dir = res_dir + "metric_mean_res/"

res_seve_path = res_save_dir + "metric_mean_res" + choose_method_num+".json"

with open(res_seve_path, "w", encoding="utf-8") as file:
    write_data = custom_json_formatter(dataset_name_score_list_dict_sort)
    file.write(write_data)
print(f"Results are saved to {res_seve_path}")