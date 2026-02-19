import json
import ast
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm_notebook as tqdm
import time
import matplotlib.ticker as mticker

from robustness_exp import create_path

# Loading the robustness results

all_file_folder = {"UCR":[],
                   "WSD":[],
                   # "YAHOO":[]
                   }
name_dict = {
    "Standard-F1": "Original-F",
    "AUC-ROC": "AUC-ROC",
    "AUC-PR": "AUC-PR",

    "PA-K": "PA-K",

    "VUS-ROC": "VUS-ROC",
    "VUS-PR": "VUS-PR",
    "PATE": "PATE",

    "R-based-F1": "RF",
    "eTaPR_F1": "eTaF",
    "Affiliation-F": "AF",

    "dqe": "DQE",
}

for file_name in os.listdir('../../../dataset/TSB-AD-U/'):
    dataset_name = file_name.split("_")[1]
    if dataset_name not in all_file_folder.keys():
        continue
    all_file_folder[dataset_name].append('{}_robustness.json'.format(file_name))


def generate_dict(list_file):
    all_dict_lag = []
    file_list = os.listdir(dirname_lag)
    for file_s in file_list:
        if file_s in list_file:
            with open(dirname_lag + file_s, "r", encoding="utf-8") as file:
                f = json.load(file)

            mapped_dict = {
                name_dict[k]: v
                for k, v in f.items()
                if k in name_dict.keys()
            }
            all_dict_lag.append(mapped_dict)

    all_dict_noise = []
    for file_s in os.listdir(dirname_noise):
        if file_s in list_file:
            with open(dirname_noise + file_s, "r", encoding="utf-8") as file:
                f = json.load(file)

            mapped_dict = {
                name_dict[k]: v
                for k, v in f.items()
                if k in name_dict.keys()
            }
            all_dict_noise.append(mapped_dict)

    all_dict_percentage = []
    for file_s in os.listdir(dirname_percentage):
        if file_s in list_file:
            with open(dirname_percentage + file_s, "r", encoding="utf-8") as file:
                f = json.load(file)

            mapped_dict = {
                name_dict[k]: v
                for k, v in f.items()
                if k in name_dict.keys()
            }
            all_dict_percentage.append(mapped_dict)
    return all_dict_lag, all_dict_noise, all_dict_percentage


def group_dict(all_dict_lag):
    d_lag = {}
    for k in measures_name:
        d_lag[k] = tuple(d[k] for d in all_dict_lag)
    return d_lag


dirname_lag = '../../../results/robustness_result/result_data_aggregated_lag/'
create_path(dirname_lag)
dirname_noise = '../../../results/robustness_result/result_data_aggregated_noise/'
create_path(dirname_noise)
dirname_percentage = '../../../results/robustness_result/result_data_aggregated_percentage/'
create_path(dirname_percentage)

measures_name = {
    "Original-F",
    "PA-K",
    "RF",
    "eTaF",
     "AF",
    "PATE",
    "AUC-ROC",
    "AUC-PR",
    "VUS-ROC",
    "VUS-PR",
    "DQE",
}

all_dict = {}
for list_file_key in all_file_folder.keys():
    all_dict_lag, all_dict_noise, all_dict_percentage = generate_dict(all_file_folder[list_file_key])
    if len(all_dict_lag) > 1:
        all_dict[list_file_key] = {}
        all_dict[list_file_key]['lag'] = group_dict(all_dict_lag)
        all_dict[list_file_key]['noise'] = group_dict(all_dict_noise)
        all_dict[list_file_key]['percentage'] = group_dict(all_dict_percentage)

overall_dict, all_dict_mean_norm, all_dict_mean = {}, {}, {}
all_dict_mean['lag'], all_dict_mean['noise'], all_dict_mean['percentage'] = {}, {}, {}
all_dict_mean_norm['lag'], all_dict_mean_norm['noise'], all_dict_mean_norm['percentage'] = {}, {}, {}
for key_metric in measures_name:
    for key_exp in all_dict_mean.keys():
        overall_dict[key_metric] = []
        all_dict_mean[key_exp][key_metric] = []
        all_dict_mean_norm[key_exp][key_metric] = []

for list_file_key in all_file_folder.keys():
    all_dict_lag, all_dict_noise, all_dict_percentage = generate_dict(all_file_folder[list_file_key])
    if len(all_dict_lag) > 1:
        for key_metric in measures_name:
            all_dict_mean['lag'][key_metric] += [val for val in group_dict(all_dict_lag)[key_metric]]
            all_dict_mean['noise'][key_metric] += [val for val in group_dict(all_dict_noise)[key_metric]]
            all_dict_mean['percentage'][key_metric] += [val for val in group_dict(all_dict_percentage)[key_metric]]

            all_dict_mean_norm['lag'][key_metric] += [np.mean(group_dict(all_dict_lag)[key_metric])]
            all_dict_mean_norm['noise'][key_metric] += [np.mean(group_dict(all_dict_noise)[key_metric])]
            percentage_temp_res = group_dict(all_dict_percentage)[key_metric]
            all_dict_mean_norm['percentage'][key_metric] += [np.mean(percentage_temp_res)]

for key_metric in measures_name:
    overall_dict[key_metric] = [np.mean([
        np.mean(all_dict_mean_norm['lag'][key_metric]),
        np.mean(all_dict_mean_norm['noise'][key_metric]),
    ])]

create_path("../../../results/robustness_result/results_figures/")
plot_single_dataset_single_robust = True

plot_all_dataset_single_robust = True

show_flag = True

desired_order = [
    "Original-F",
    "PA-K",
    "RF",
    "eTaF",
    "PATE",
    "AUC-ROC",
    "AUC-PR",
    "VUS-ROC",
    "VUS-PR",
    "AF",
    "DQE",
]

# Plotting the overall robustness results for each dataset in the benchmark

if plot_single_dataset_single_robust:
    fontsize = 50
    fontsize_tick = 50
    plt.figure(figsize=(40, 15))

    measures = ['lag', 'noise', 'percentage']

    for i, key in enumerate(['WSD', 'UCR']):
        for j, measure in enumerate(measures):
            ax = plt.subplot(len(all_dict.keys()), 3, 1 + j + i * 3)

            data = pd.DataFrame.from_dict(all_dict[key][measure]).mean()
            data = data.reindex(desired_order)
            data.plot.bar(ax=ax)

            ax.set_yscale('log')

            ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

            ax.yaxis.set_minor_formatter(mticker.LogFormatterSciNotation())

            ax.tick_params(axis='y', which='both', labelsize=fontsize_tick)
            ax.tick_params(axis='x', which='both', labelsize=fontsize_tick)

            if measure == "percentage":
                measure = "ratio"
            ax.set_title("Averaged " + measure + " sensitivity (" + key + ")", fontsize=fontsize)

            ax.grid(which="both")

            ax.yaxis.set_tick_params(labelsize=fontsize_tick)
            ax.xaxis.set_tick_params(labelsize=fontsize_tick)

            if i == len(all_dict.keys()) - 1:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            else:
                ax.set_xticklabels([''] * len(ax.get_xticklabels()))

    plt.tight_layout()
    if show_flag:
        plt.show()

# Plotting the overall robustness results across all datasets

if plot_all_dataset_single_robust:
    plt.rcParams.update({'font.size': 50})
    plt.figure(figsize=(40, 10.5))
    plt.subplot(1, 3, 1)
    plt.title('Averaged Lag sensitivity')

    data = pd.DataFrame.from_dict(all_dict_mean_norm['lag']).mean()
    data = data.reindex(desired_order)
    data.plot.bar()
    plt.yscale('log')
    plt.xticks(rotation=90)
    plt.grid(which="both")

    plt.subplot(1, 3, 2)
    plt.title('Averaged Noise sensitivity')

    data = pd.DataFrame.from_dict(all_dict_mean_norm['noise']).mean()
    data = data.reindex(desired_order)
    data.plot.bar()
    plt.yscale('log')
    plt.yscale('log')
    plt.xticks(rotation=90)
    plt.grid(which="both")

    plt.subplot(1, 3, 3)
    plt.title('Averaged ratio sensitivity')

    data = pd.DataFrame.from_dict(all_dict_mean_norm['percentage']).mean()
    data = data.reindex(desired_order)
    data.plot.bar()
    plt.yscale('log')
    plt.xticks(rotation=90)
    plt.grid(which="both")
    plt.tight_layout()
    if show_flag:
        plt.show()