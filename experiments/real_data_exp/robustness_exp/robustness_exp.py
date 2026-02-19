import json

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import time
import random
from random import shuffle
from matplotlib import cm
from multiprocessing import Pool
import pickle
from copy import deepcopy

import os
import sys


module_path = os.path.abspath(os.path.join('evaluation_metrics'))
if module_path not in sys.path:
    sys.path.append(module_path)

from experiments.TSB_AD.utils.slidingWindows import find_length_rank,find_length
from experiments.TSB_AD.HP_list import Optimal_Uni_algo_HP_dict, Optimal_Multi_algo_HP_dict
from experiments.TSB_AD.model_wrapper import *
from evaluation.metrics import get_metrics

def find_section_length(label,length):
    best_i = None
    best_sum = None
    current_subseq = False
    for i in range(len(label)):
        changed = False
        if label[i] == 1:
            if current_subseq == False:
                current_subseq = True
                if best_i is None:
                    changed = True
                    best_i = i
                    best_sum = np.sum(label[max(0,i-200):min(len(label),i+9800)])
                else:
                    if np.sum(label[max(0,i-200):min(len(label),i+9800)]) < best_sum:
                        changed = True
                        best_i = i
                        best_sum = np.sum(label[max(0,i-200):min(len(label),i+9800)])
                    else:
                        changed = False
                if changed:
                    diff = i+9800 - len(label)
            
                    pos1 = max(0,i-200 - max(0,diff))
                    pos2 = min(i+9800,len(label))
        else:
            current_subseq = False
    if best_i is not None:
        return best_i-pos1,(pos1,pos2)
    else:
        return None,None

def generate_data(filepath, init_pos, max_length,dataset_name):
    df = pd.read_csv(filepath).dropna()
    data_all = df.iloc[:, 0:-1].values.astype(float)
    label_all = df.iloc[:, -1].values.astype(int)

    pos_first_anom, pos = find_section_length(label_all, max_length)
    if pos is None:
        return None, None, None, None, None, None, None, None, None
    data = data_all[pos[0]:pos[1]].astype(float).reshape(-1, 1)
    label = label_all[pos[0]:pos[1]]

    slidingWindow = find_length_rank(data, rank=1)

    # train ratio
    if dataset_name == "YAHOO":
        data_train = data[:int(0.2 * len(data))]
    else:
        data_train = data[:int(0.1 * len(data))]

    data_test = data

    return pos_first_anom, slidingWindow, data, data_train, data_test, label


def create_path(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        dir_path = os.path.dirname(path) if os.path.splitext(path)[1] else path
        os.makedirs(dir_path, exist_ok=True)

def write_json(save_path, single_dict):
    create_path(save_path)
    with open(save_path, 'w', encoding='utf-8') as json_file:
        json.dump(single_dict, json_file, indent=4, ensure_ascii=False)
    print(f"write to {save_path}")

def compute_score(data,data_train,dataset_name,file_name):

    filter_ad_pool = ['SR', 'CNN', 'Sub_LOF', 'FFT', 'KMeansAD_U', 'Sub_KNN', 'TimesNet']

    methods_scores_scaled = {}
    for ad_name in filter_ad_pool:
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

        output_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(output.reshape(-1, 1)).ravel()

        methods_scores_scaled[ad_name] = output_scaled

    return methods_scores_scaled



def generate_new_label(label,lag):
    if lag < 0:
        return np.array(list(label[-lag:]) + [0]*(-lag))
    elif lag > 0:
        return np.array([0]*lag + list(label[:-lag]))
    elif lag == 0:
        return label

def bounded_random_walk(length, lower_bound,  upper_bound, start, end, std):
    assert (lower_bound <= start and lower_bound <= end)
    assert (start <= upper_bound and end <= upper_bound)

    bounds = upper_bound - lower_bound

    rand = (std * (np.random.random(length) - 0.5)).cumsum()
    rand_trend = np.linspace(rand[0], rand[-1], length)
    rand_deltas = (rand - rand_trend)
    rand_deltas /= np.max([1, (rand_deltas.max()-rand_deltas.min())/bounds])

    trend_line = np.linspace(start, end, length)
    upper_bound_delta = upper_bound - trend_line
    lower_bound_delta = lower_bound - trend_line

    upper_slips_mask = (rand_deltas-upper_bound_delta) >= 0
    upper_deltas =  rand_deltas - upper_bound_delta
    rand_deltas[upper_slips_mask] = (upper_bound_delta - upper_deltas)[upper_slips_mask]

    lower_slips_mask = (lower_bound_delta-rand_deltas) >= 0
    lower_deltas =  lower_bound_delta - rand_deltas
    rand_deltas[lower_slips_mask] = (lower_bound_delta + lower_deltas)[lower_slips_mask]

    return trend_line + rand_deltas


dict_acc_global = {
            'Standard-F1': [],
            'AUC-ROC': [],
            'AUC-PR': [],

            'PA-K': [],

            'VUS-ROC': [],
            'VUS-PR': [],
            'PATE': [],

            'R-based-F1': [],
            'eTaPR_F1': [],
            'Affiliation-F': [],

            "dqe": [],
        }

methods_keys_global = ['SR', 'CNN', 'Sub_LOF', 'FFT', 'KMeansAD_U', 'Sub_KNN', 'TimesNet']


def compute_anomaly_acc_lag(methods_scores, label, slidingWindow, methods_keys, dataset_name, file_name):
    robust_type = "lag"
    lag_num = 10
    lag_range = list(range(-slidingWindow // 4, slidingWindow // 4, max(1, (slidingWindow // 2) // lag_num)))
    methods_acc = {}

    for i, method_name in enumerate(tqdm(methods_keys)):
        dict_acc = deepcopy(dict_acc_global)

        for idy,lag in enumerate(lag_range):
            eva_save_path = "../../../results/robustness_result/evaluation_res_robust_exp/" + robust_type \
                            + "/" + dataset_name \
                            + "/" + file_name.split(".")[0] \
                            + "/" + method_name \
                            + "/" + robust_type + "_" +  str(idy) \
                            + "/" + "evaluation_res.json"

            new_label = generate_new_label(label, lag)
            metric_score_dict,_ = get_metrics(methods_scores[method_name], new_label, slidingWindow=slidingWindow,
                                            thre=100)

            # save robust variation
            evaluation_res_dict = {"evaluation_res": metric_score_dict}
            write_json(eva_save_path, evaluation_res_dict)

            # add value list(concat)
            for idx, (metric_name, eva_score) in enumerate(metric_score_dict.items()):
                # metric = [[lag1],[lag2]],one file,one method,all lags
                dict_acc[metric_name] += [eva_score]

        # one file,all method,all lags
        methods_acc[method_name] = dict_acc
    return methods_acc


def compute_anomaly_acc_noise(methods_scores, label, slidingWindow, methods_keys, dataset_name, file_name):
    robust_type = "noise"
    noise_times = 10
    methods_acc = {}
    for i, method_name in enumerate(tqdm(methods_keys)):
        dict_acc = deepcopy(dict_acc_global)

        for noise_idx in range(noise_times):
            eva_save_path = "../../../results/robustness_result/evaluation_res_robust_exp/" + robust_type \
                            + "/" + dataset_name \
                            + "/" + file_name.split(".")[0] \
                            + "/" + method_name \
                            + "/" + robust_type + "_" + str(noise_idx) \
                            + "/" + "evaluation_res.json"

            new_label = label

            delta = methods_scores[method_name].max() - methods_scores[method_name].min()
            noise = np.random.uniform(-0.05 * delta, 0.05 * delta, len(methods_scores[method_name]))
            perturbed_score = np.array(methods_scores[method_name]) + noise
            perturbed_score_range = (perturbed_score.max() - perturbed_score.min())
            new_score = (perturbed_score - perturbed_score.min()) / perturbed_score_range if perturbed_score_range != 0 else (perturbed_score - perturbed_score.min())

            metric_score_dict,_ = get_metrics(new_score, new_label, slidingWindow=slidingWindow,
                                            thre=100)


            # add value list(concat)
            for idx, (metric_name, eva_score) in enumerate(metric_score_dict.items()):
                # metric = [[noise1],[noise2]],one file,one method,all noises
                dict_acc[metric_name] += [eva_score]

        methods_acc[method_name] = dict_acc
    return methods_acc


def compute_anomaly_acc_percentage(methods_scores, label, slidingWindow, methods_keys, pos_first_anom, dataset_name,
                                   file_name):
    robust_type = "percentage"
    extend_times = 10
    list_pos = []
    step_a = max(0, (len(label) - pos_first_anom - 200)) // extend_times
    step_b = max(0, pos_first_anom - 200) // extend_times
    pos_a = min(len(label), pos_first_anom + 200)
    pos_b = max(0, pos_first_anom - 200)
    list_pos.append((pos_b, pos_a))
    for pos_iter in range(extend_times):
        pos_a = min(len(label), pos_a + step_a)
        pos_b = max(0, pos_b - step_b)
        list_pos.append((pos_b, pos_a))

    methods_acc = {}
    for i, method_name in enumerate(tqdm(methods_keys)):
        dict_acc = deepcopy(dict_acc_global)

        for idy,end_pos in enumerate(list_pos):
            eva_save_path = "../../../results/robustness_result/evaluation_res_robust_exp/" + robust_type \
                            + "/" + dataset_name \
                            + "/" + file_name.split(".")[0] \
                            + "/" + method_name \
                            + "/" + robust_type + "_" + str(idy) \
                            + "/" + "evaluation_res.json"

            new_label = label[end_pos[0]:end_pos[1]]
            new_score = np.array(methods_scores[method_name])[end_pos[0]:end_pos[1]]
            metric_score_dict,_ = get_metrics(new_score, new_label, slidingWindow=slidingWindow,
                                            thre=100)

            for idx, (metric_name, eva_score) in enumerate(metric_score_dict.items()):
                # metric = [[percentage1],[percentage2]],one file,one method,all percentages
                dict_acc[metric_name] += [eva_score]

        methods_acc[method_name] = dict_acc
    return methods_acc

def group_dict(methods_acc_lag):
    key_metrics = deepcopy(list(dict_acc_global.keys()))
    methods_keys = deepcopy(methods_keys_global)

    norm_methods_acc_lag = {key:[] for key in key_metrics}
    for key in methods_keys:
        for key_metric in key_metrics:
            ts = list(methods_acc_lag[key][key_metric])
            new_ts = list(np.array(ts) -  np.mean(ts))
            norm_methods_acc_lag[key_metric] += new_ts
    return norm_methods_acc_lag

def sanitize_float(x, eps=1e-12):
    if abs(x) < eps:
        return 0.0
    return x

def sanitize_dict(d):
    return {k: sanitize_float(v) for k, v in d.items()}


def run_method_analysis(filepath):
    methods_keys = deepcopy(methods_keys_global)

    aggregated_lag_res_path = '../../../results/robustness_result/result_data_aggregated_lag/{}_robustness.json'.format(filepath.split('/')[-1])
    aggregated_noise_res_path = '../../../results/robustness_result/result_data_aggregated_noise/{}_robustness.json'.format(filepath.split('/')[-1])
    aggregated_percentage_res_path = '../../../results/robustness_result/result_data_aggregated_percentage/{}_robustness.json'.format(filepath.split('/')[-1])

    all_aggregated_res_path = '../../../results/robustness_result/result_data_aggregated/{}_robustness.json'.format(filepath.split('/')[-1])
    create_path(all_aggregated_res_path)

    res_path_list = [aggregated_lag_res_path,
                     aggregated_noise_res_path,
                     aggregated_percentage_res_path]
    for res_path in res_path_list:
        dir_path = os.path.dirname(res_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    dataset_name = filepath.split("/")[-1].split("_")[1]
    file_name = filepath.split("/")[-1]

    if dataset_name == "YAHOO":
        max_length = 1400
    else:
        max_length = 10000

    pos_first_anom, slidingWindow, data, data_train, data_test, label = generate_data(filepath,0,max_length,dataset_name)

    # get slidingWindow for robust variation
    if slidingWindow is None:
        return None

    # step1
    methods_scores =  compute_score(data,data_train,dataset_name,file_name)

    # step2
    methods_acc_lag = compute_anomaly_acc_lag(methods_scores,label,slidingWindow,methods_keys,dataset_name, file_name)
    methods_acc_noise = compute_anomaly_acc_noise(methods_scores,label,slidingWindow,methods_keys,dataset_name, file_name)
    methods_acc_percentage = compute_anomaly_acc_percentage(methods_scores,label,slidingWindow,methods_keys,pos_first_anom,dataset_name, file_name)

    # step3
    group_norm_methods_acc_lag = group_dict(methods_acc_lag)
    group_norm_methods_acc_noise = group_dict(methods_acc_noise)
    group_norm_methods_acc_percentage = group_dict(methods_acc_percentage)

    all_res_robust = {}
    all_res_robust_lag = {}
    all_res_robust_noise = {}
    all_res_robust_percentage = {}
    for key in group_norm_methods_acc_lag.keys():
        std_1 = np.std(group_norm_methods_acc_lag[key])
        std_2 = np.std(group_norm_methods_acc_noise[key])
        std_3 = np.std(group_norm_methods_acc_percentage[key])

        all_res_robust_lag[key] = std_1
        all_res_robust_noise[key] = std_2
        all_res_robust_percentage[key] = std_3
        all_res_robust[key] = np.mean([std_1,std_2,std_3])

    write_json(aggregated_lag_res_path,all_res_robust_lag)
    write_json(aggregated_noise_res_path,all_res_robust_noise)
    write_json(aggregated_percentage_res_path,all_res_robust_percentage)
    write_json(all_aggregated_res_path,all_res_robust)

    return all_res_robust

def multi_run_wrapper(args):
   return run_method_analysis(*args)

def main():
    dataset_name_list = [
        'WSD',
        'YAHOO',
        'UCR'
    ]

    all_files = []

    ori_file_dir = "../../../dataset/TSB-AD-U/"
    for ori_file in os.listdir(ori_file_dir):
        dataset_name = ori_file.split("_")[1]
        if dataset_name not in dataset_name_list:
            continue
        all_files.append([ori_file_dir+ori_file])


    with Pool(processes=16) as pool:
        results = pool.map(multi_run_wrapper,all_files)


if __name__ == '__main__':
        main()



