#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###########################
import os
import numpy as np
import math
import json

from sortedcontainers import SortedSet
from copy import deepcopy
from evaluation.slidingWindows import find_length_rank


def split_intervals(intervals, split_points):
    """
        Split every interval in `intervals` at the given `split_points` that fall
        strictly inside the interval.

        Parameters
        ----------
        intervals : list[list[float]]
            List of [start, end] pairs to be subdivided.
        split_points : list[float]
            Points at which to split.  They must be provided in ascending order
            and without duplicates.

        Returns
        -------
        list[list[float]]
            Flat list of [start, end] sub-intervals produced by the splits.
        """

    # Initialize the result list
    result = []

    # Iterate through each interval in the 2D list a
    for interval in intervals:
        start, end = interval
        # Get the split points within the current interval
        current_splits = [start] + [point for point in split_points if start < point < end] + [end]

        # Generate new sub-intervals based on the split points
        for i in range(len(current_splits) - 1):
            result.append([current_splits[i], current_splits[i + 1]])

    return result


def pred_in_area(pred, area):
    # Check whether a prediction interval lies entirely within a given area.
    if area == [] or pred == []:
        return False
    return True if pred[0] >= area[0] and pred[1] <= area[1] else False


def ddl_func(x, max_area_len, gama=1):
    """
        Compute a decaying-linear or power-based score that drops from 1 to 0
        as `x` approaches `max_area_len`.

        The function implements:
            score = (1 / max_area_len^gama) * (max_area_len - x)^gama
                  = [(max_area_len - x) / max_area_len]^gama

        When gama = 1 the curve is a straight line; gama > 1 yields concave decay.

        Parameters
        ----------
        x : float
            Independent variable (must be <= max_area_len).
        max_area_len : float
            Length or upper bound at which the score becomes zero.
        gama : float, optional
            Exponent controlling the shape of decay (default 1).

        Returns
        -------
        float
            Score in [0, 1]; equals 1 at x = 0 and 0 at x = max_area_len.
        """
    parameter_a = 1 / max_area_len ** gama
    score = parameter_a * (max_area_len - x) ** gama
    return score


def false_alarm_func_liner(x, dis_range=100):
    """
        Compute a linear decay score for false alarm based on total duration.

        The score decays linearly from 1 to 0 as the total duration increases from 0 to
        `dis_range`. Distances beyond `dis_range` receive zero penalty.

        Parameters
        ----------
        x : float
            Total duration of false alarms.
        dis_range : float, optional
            Threshold on total duration.
            If $x$ exceeds this threshold, the false alarm penalty reaches its maximum (i.e., the score is 0).

        Returns
        -------
        float
            False alarm score in [0, 1]:
        """
    if x > dis_range:
        score = 0
    else:
        score = ddl_func(x, dis_range)
    return score


def randomness_penalty_coefficient(distances, a, b):
    """
    Randomness Penalty Coefficient (distance range -a ~ b)

    Parameters
    ----------
    distances : 1-D array_like
        Distances from midpoints of detection events to local anomaly event.
    a, b : float > 0
        Left and right boundaries of temporal differences in the false alarm subregion.

    Returns
    -------
    float
        Randomness penalty coefficient P ∈ [0,1];
        Low randomness → P approaches 1,
        High randomness → P approaches 0
    """
    distances = np.asarray(distances, dtype=float)
    if distances.size == 0 or (a + b) <= 1:
        return 1.0

    # 1. Clip to [-a, b]
    distances = np.clip(distances, -a, b)

    # 2. Histogram: bin length = 1, total K bins
    K = int(np.ceil(a + b))
    # Divide [-a, b] into K equal segments
    bins = np.linspace(-a, b, K + 1)  # left-closed, right-closed, K+1 boundaries
    ori_counts, _ = np.histogram(distances, bins=bins)
    counts = (ori_counts >= 1).astype(int)

    # 3. Entropy
    total = counts.sum()
    if total == 0:
        return 1.0
    p_nonzero = counts[counts > 0] / total
    H = -np.sum(p_nonzero * np.log2(p_nonzero))

    # 4. Randomness score S and penalty coefficient P
    S = H / np.log2(K)
    P = 1.0 - S
    return P


def DQE_section(tq_section_list, prediction_section_list, ts_len, gt_num=None, near_single_side_range=125,
                cal_components=False, partition_res=None):
    """
    Evaluate binary detection results and compute the local DQE score for each anomaly event.

    The function completes the full DQE pipeline:
    1. Section partitioning
    2. Local detection-event grouping
    3. Single-threshold DQE-cap, DQE-nm and DQE-fa calculation
    4. DQE-nm and DQE-fa adjustment
    5. Aggregation of local DQE

    Parameters
    ----------
    tq_section_list : list[list[float]]
        Ground-truth anomaly intervals.
    prediction_section_list : list[list[float]]
        Detection intervals produced by the algorithm after thresholding.
    ts_len : int
        Total length of the time series.
    gt_num : int
        Number of ground-truth anomalous events.
    near_single_side_range : float
        Half-width of the temporal window used to define near-miss regions;
        detections within this window are treated as relevant near-misses.
    cal_components : bool
        If True, also return the three local DQE-component scores (DQE-cap, DQE-nm, DQE-fa);
        otherwise return only the aggregated local DQE.
    partition_res : dict
        Threshold-free DQE: partition result pre-computed and passed in.
        Threshold-dependent DQE: partition computed inside this call.

    Returns
    -------
    dqe_res_ts : dict
        seq_dqe_local_list : list[float]
            Final local DQE score for each anomaly event.
        seq_cap_local_list : list[float], optional
            Local DQE-cap for each anomaly event (returned only when cal_components=True).
        seq_near_miss_local_list : list[float], optional
            Local DQE-nm for each anomaly event (returned only when cal_components=True).
        seq_false_alarm_local_list : list[float], optional
            Local DQE-fa for each anomaly event (returned only when cal_components=True).
    """

    if not gt_num:
        gt_num = len(tq_section_list)

    # section partition
    if partition_res is not None:
        # area list
        fq_dis_e_section_list = partition_res["fq_dis_e_section_list"]  # index:0-(n-1)
        fq_dis_d_section_list = partition_res["fq_dis_d_section_list"]  # index:1-n
        fq_near_e_section_list = partition_res["fq_near_e_section_list"]  # index:0-(n-1)
        fq_near_d_section_list = partition_res["fq_near_d_section_list"]  # index:1-n

        split_line_set = partition_res["split_line_set"]
    else:
        # area list
        fq_dis_e_section_list = [[] for _ in range(gt_num)]  # index:0-(n-1)
        fq_dis_d_section_list = [[] for _ in range(gt_num + 1)]  # index:1-n
        fq_near_e_section_list = [[] for _ in range(gt_num)]  # index:0-(n-1)
        fq_near_d_section_list = [[] for _ in range(gt_num + 1)]  # index:1-n

        split_line_set = SortedSet()

        for i, tq_section_i in enumerate(tq_section_list):
            # tq_i
            tq_section_i_start, tq_section_i_end = tq_section_i

            # fq_distant_i_mid
            fq_dis_section_i_mid = None
            if i == 0:
                if gt_num == 1:
                    # get position
                    # fq_near_early_i position
                    fq_near_e_section_i_start = max(tq_section_i_start - near_single_side_range, 0)
                    fq_near_e_section_i_end = tq_section_i_start

                    # fq_near_delay_i_next position
                    fq_near_d_section_i_next_start = tq_section_i_end
                    fq_near_d_section_i_next_end = min(tq_section_i_end + near_single_side_range, ts_len)

                    # fq_near
                    # fq_near_early_i
                    fq_near_e_section_list[i] = [fq_near_e_section_i_start, fq_near_e_section_i_end]

                    # next section
                    # fq_near_delay_i_next
                    fq_near_d_section_list[i + 1] = [fq_near_d_section_i_next_start,
                                                     fq_near_d_section_i_next_end]

                    # fq_distant
                    # fq_distant_early_i
                    fq_dis_e_section_list[i] = [0, fq_near_e_section_i_start]

                    # next section
                    # fq_distant_delay_i_next
                    fq_dis_d_section_list[i + 1] = [fq_near_d_section_i_next_end, ts_len]
                else:
                    # get position
                    fq_near_e_section_i_start = max(tq_section_i_start - near_single_side_range, 0)
                    fq_near_e_section_i_end = tq_section_i_start

                    tq_section_i_next_start, tq_section_i_next_end = tq_section_list[i + 1]

                    fq_near_d_section_i_next_start = tq_section_i_end
                    fq_near_d_section_i_next_end = min(tq_section_i_end + near_single_side_range,
                                                       tq_section_i_next_start)

                    # fq_near
                    fq_near_e_section_list[i] = [fq_near_e_section_i_start, fq_near_e_section_i_end]

                    # next section
                    fq_near_d_section_list[i + 1] = [fq_near_d_section_i_next_start, fq_near_d_section_i_next_end]

                    # fq_distant
                    # fq_distant_early_i
                    fq_dis_e_section_list[i] = [0, fq_near_e_section_i_start]
            elif i == gt_num - 1:
                # get position
                tq_section_i_last_start, tq_section_i_last_end = tq_section_list[i - 1]

                fq_near_e_section_i_start = max(tq_section_i_start - near_single_side_range, tq_section_i_last_end)
                fq_near_e_section_i_end = tq_section_i_start

                fq_near_d_section_i_next_start = tq_section_i_end
                fq_near_d_section_i_next_end = min(tq_section_i_end + near_single_side_range, ts_len)

                fq_near_d_i_end = fq_near_d_section_list[i][1]

                # fq_near
                fq_near_e_section_list[i] = [fq_near_e_section_i_start, fq_near_e_section_i_end]
                fq_near_d_section_list[i + 1] = [fq_near_d_section_i_next_start, fq_near_d_section_i_next_end]

                # fq_distant
                fq_dis_section_i_mid = (fq_near_d_i_end + fq_near_e_section_i_start) / 2
                fq_dis_d_section_list[i] = [fq_near_d_i_end, fq_dis_section_i_mid]
                fq_dis_e_section_list[i] = [fq_dis_section_i_mid, fq_near_e_section_i_start]

                # next section
                fq_dis_d_section_list[i + 1] = [fq_near_d_section_i_next_end, ts_len]
            else:
                # get position
                tq_section_i_last_start, tq_section_i_last_end = tq_section_list[i - 1]
                tq_section_i_next_start, tq_section_i_next_end = tq_section_list[i + 1]

                fq_near_e_section_i_start = max(tq_section_i_start - near_single_side_range, tq_section_i_last_end)
                fq_near_e_section_i_end = tq_section_i_start

                fq_near_d_section_i_next_start = tq_section_i_end
                fq_near_d_section_i_next_end = min(tq_section_i_end + near_single_side_range, tq_section_i_next_start)

                fq_near_d_i_end = fq_near_d_section_list[i][1]

                # fq_near
                fq_near_e_section_list[i] = [fq_near_e_section_i_start, fq_near_e_section_i_end]
                fq_near_d_section_list[i + 1] = [fq_near_d_section_i_next_start, fq_near_d_section_i_next_end]

                # fq_distant
                fq_dis_section_i_mid = (fq_near_d_i_end + fq_near_e_section_i_start) / 2
                fq_dis_d_section_list[i] = [fq_near_d_i_end, fq_dis_section_i_mid]
                fq_dis_e_section_list[i] = [fq_dis_section_i_mid, fq_near_e_section_i_start]

            # add split line
            split_line_set.add(fq_near_e_section_i_start)

            split_line_set.add(tq_section_i_start)
            split_line_set.add(tq_section_i_end)

            split_line_set.add(fq_near_d_section_i_next_end)

            if i > 0:
                if fq_dis_section_i_mid != None:
                    split_line_set.add(fq_dis_section_i_mid)

    # split prediction from segments to events (second circle,prediction_interval_num*gt_num)
    prediction_events = split_intervals(prediction_section_list, split_line_set)

    # detection event group
    tq_prediction_group_list = [[] for _ in range(gt_num)]

    fq_near_e_prediction_group_list = [[] for _ in range(gt_num)]
    fq_near_d_prediction_group_list = [[] for _ in range(gt_num + 1)]

    precision_prediction_group_list = [[] for _ in range(gt_num)]

    fq_dis_e_prediction_group_list = [[] for _ in range(gt_num)]
    fq_dis_d_prediction_group_list = [[] for _ in range(gt_num + 1)]

    # get local prediction event group (third circle,prediction_event_num*gt_num*4)
    for i, basic_interval in enumerate(prediction_events):
        # tq_group
        for j, area in enumerate(tq_section_list):
            if pred_in_area(basic_interval, area):
                tq_prediction_group_list[j].append(basic_interval)

        # fq_near_early_group
        for j, area in enumerate(fq_near_e_section_list):
            if pred_in_area(basic_interval, area):
                fq_near_e_prediction_group_list[j].append(basic_interval)

        # fq_near_delay_group
        for j, area in enumerate(fq_near_d_section_list):
            if pred_in_area(basic_interval, area):
                fq_near_d_prediction_group_list[j].append(basic_interval)

            # precision_group
            if j >= 1:
                area_id_last = j - 1
                precision_prediction_group_list[j - 1] = fq_near_e_prediction_group_list[area_id_last] + \
                                                         tq_prediction_group_list[area_id_last] + \
                                                         fq_near_d_prediction_group_list[j]
        # fq_distant_group
        for j, area in enumerate(fq_dis_e_section_list):
            if pred_in_area(basic_interval, area):
                fq_dis_e_prediction_group_list[j].append(basic_interval)

        for j, area in enumerate(fq_dis_d_section_list):
            if pred_in_area(basic_interval, area):
                fq_dis_d_prediction_group_list[j].append(basic_interval)

    # deal the mid one (not need here)
    for i in range(gt_num):
        if i >= 1 and i <= gt_num:
            if fq_dis_d_prediction_group_list[i] != [] and fq_dis_e_prediction_group_list[i] != []:
                f_dis_d_last_pred = fq_dis_d_prediction_group_list[i][-1]
                f_dis_e_first_pred = fq_dis_e_prediction_group_list[i][0]
                f_dis_d_last_pred_len = f_dis_d_last_pred[1] - f_dis_d_last_pred[0]
                f_dis_e_first_pred_len = f_dis_e_first_pred[1] - f_dis_e_first_pred[0]
                if f_dis_d_last_pred_len >= f_dis_e_first_pred_len:
                    fq_dis_d_prediction_group_list[i][-1] = [f_dis_d_last_pred[0], f_dis_e_first_pred[1]]
                    fq_dis_e_prediction_group_list[i].pop(0)
                else:
                    fq_dis_d_prediction_group_list[i].pop()
                    fq_dis_e_prediction_group_list[i][0] = [f_dis_d_last_pred[0], f_dis_e_first_pred[1]]

    # fq_near_early
    fq_near_e_score_list = np.arange(gt_num, dtype=np.float64)

    # fq_near_delay
    fq_near_d_score_list = np.arange(gt_num + 1, dtype=np.float64)

    mp_fq_near_list = [[] for _ in range(gt_num)]  # item num = pred num
    co_fq_near_list = [[] for _ in range(gt_num)]  # item num = 2
    td_fq_near_list = [[] for _ in range(gt_num)]  # item num = 2
    fq_near_score_list = np.arange(gt_num, dtype=np.float64)

    # cal score function,tq_near score
    for i, area in enumerate(tq_section_list):
        area_id_next = i + 1
        # fq_near_early

        fq_near_e_area = fq_near_e_section_list[i]

        fq_near_e_end = fq_near_e_area[1]

        fq_near_e_pred_group = fq_near_e_prediction_group_list[i]

        fq_near_e_num = len(fq_near_e_pred_group)
        for interval_idx, basic_interval in enumerate(fq_near_e_pred_group):
            single_duration = basic_interval[1] - basic_interval[0]
            td_fq_near_list[i].append(single_duration)

            single_p = abs(fq_near_e_end - (basic_interval[1] + basic_interval[0]) / 2)
            mp_fq_near_list[i].append(single_p)

            if interval_idx == fq_near_e_num - 1:
                single_r = abs(fq_near_e_end - basic_interval[1])
                co_fq_near_list[i].append(single_r)

        # fq_near_delay
        fq_near_d_area = fq_near_d_section_list[area_id_next]
        fq_near_d_start = fq_near_d_area[0]

        fq_near_d_pred_group = fq_near_d_prediction_group_list[area_id_next]

        for interval_idx, basic_interval in enumerate(fq_near_d_pred_group):
            single_duration = basic_interval[1] - basic_interval[0]
            td_fq_near_list[i].append(single_duration)

            single_p = abs((basic_interval[1] + basic_interval[0]) / 2 - fq_near_d_start)
            mp_fq_near_list[i].append(single_p)

            if interval_idx == 0:
                single_r = abs(fq_near_d_start - basic_interval[0])
                co_fq_near_list[i].append(single_r)

        mp_fq_near = np.mean(np.array(mp_fq_near_list[i])) if len(mp_fq_near_list[i]) > 0 else 0
        co_fq_near = np.min(np.array(co_fq_near_list[i])) if len(co_fq_near_list[i]) > 0 else 0
        td_fq_near = np.sum(np.array(td_fq_near_list[i])) if len(td_fq_near_list[i]) > 0 else 0

        score_fq_near_mp = ddl_func(mp_fq_near, near_single_side_range,
                                    gama=1) if near_single_side_range != 0 else 1
        score_fq_near_co = ddl_func(co_fq_near, near_single_side_range,
                                    gama=1) if near_single_side_range != 0 else 1
        score_fq_near_td = ddl_func(td_fq_near, 2 * near_single_side_range,
                                    gama=1) if near_single_side_range != 0 else 1

        score_fq_near = score_fq_near_mp * \
                        score_fq_near_co * \
                        score_fq_near_td

        fq_near_score_list[i] = score_fq_near

    # adjust near-miss score,see both sides
    adjust_score_fq_near_list = deepcopy(fq_near_score_list)

    for i in range(gt_num):
        area_id_next = i + 1
        fq_near_prediction_group_now = fq_near_e_prediction_group_list[i] + fq_near_d_prediction_group_list[
            area_id_next]
        fq_dis_prediction_group_now = fq_dis_e_prediction_group_list[i] + fq_dis_d_prediction_group_list[area_id_next]

        if fq_near_prediction_group_now == [] \
                and (tq_prediction_group_list[i] == [] or \
                     (tq_prediction_group_list[i] != [] and \
                      fq_dis_prediction_group_now != [])):
            adjust_score_fq_near_list[i] = 0

    # false alarm score
    score_fq_dis_list = np.arange(gt_num, dtype=np.float64)

    adjust_score_fq_dis_list = np.arange(gt_num, dtype=np.float64)

    td_fq_dis_e_list = np.arange(gt_num + 1, dtype=np.float64)

    td_fq_dis_d_list = np.arange(gt_num + 1, dtype=np.float64)

    p_direction_dis_all_list = [[] for _ in range(gt_num)]

    for i, area in enumerate(tq_section_list):
        area_id_next = i + 1

        fq_dis_e_area = fq_dis_e_section_list[i]
        fq_dis_e_end = fq_dis_e_area[1]

        fq_dis_d_area = fq_dis_d_section_list[area_id_next]
        fq_dis_d_start = fq_dis_d_area[0]

        fq_dis_e_pred_group = fq_dis_e_prediction_group_list[i]
        fq_dis_d_next_pred_group = fq_dis_d_prediction_group_list[area_id_next]

        td_fq_dis_e_sum = 0
        p_direction_dis_list = []

        for interval_idx, basic_interval in enumerate(fq_dis_e_pred_group):
            td_fq_dis_e_sum += basic_interval[1] - basic_interval[0]

            p_dis_e = abs(fq_dis_e_end - (basic_interval[1] + basic_interval[0]) / 2)
            p_direction_dis_list.append(-1 * p_dis_e)

        td_fq_dis_e_list[i] = td_fq_dis_e_sum

        td_fq_dis_d_sum = 0

        for interval_idx, basic_interval in enumerate(fq_dis_d_next_pred_group):
            td_fq_dis_d_sum += basic_interval[1] - basic_interval[0]

            p_dis_d = abs((basic_interval[1] + basic_interval[0]) / 2 - fq_dis_d_start)
            p_direction_dis_list.append(p_dis_d)

        td_fq_dis_d_list[area_id_next] = td_fq_dis_d_sum

        p_direction_dis_all_list[i] = p_direction_dis_list

    for i in range(gt_num):
        area_id_next = i + 1

        fq_dis_e_section = fq_dis_e_section_list[i]
        fq_dis_d_next_section = fq_dis_d_section_list[area_id_next]
        if fq_dis_e_section[0] <= fq_dis_e_section[1]:
            fq_dis_e_section_len = fq_dis_e_section[1] - fq_dis_e_section[0]
        else:
            fq_dis_e_section_len = 0
        if fq_dis_d_next_section[0] <= fq_dis_d_next_section[1]:
            fq_dis_d_next_section_len = fq_dis_d_next_section[1] - fq_dis_d_next_section[0]
        else:
            fq_dis_d_next_section_len = 0

        p_direction_dis_list_now = p_direction_dis_all_list[i]

        randomness_penalty_score = randomness_penalty_coefficient(p_direction_dis_list_now, fq_dis_e_section_len,
                                                                  fq_dis_d_next_section_len)

        fq_dis_e_pred_group = fq_dis_e_prediction_group_list[i]
        fq_dis_d_next_pred_group = fq_dis_d_prediction_group_list[area_id_next]

        fq_dis_pred_group_td_around = td_fq_dis_e_list[i] + td_fq_dis_d_list[i + 1]

        dis_section_len = fq_dis_e_section_len + fq_dis_d_next_section_len
        dis_section_scaled = dis_section_len / 2

        false_alarm_score = false_alarm_func_liner(fq_dis_pred_group_td_around,
                                                   dis_section_scaled) if dis_section_len != 0 else 1
        score_fq_dis_td = randomness_penalty_score * false_alarm_score

        score_fq_dis_list[i] = score_fq_dis_td

        # adjust false alarm
        if fq_dis_e_pred_group == [] \
                and fq_dis_d_next_pred_group == [] \
                and fq_near_e_prediction_group_list[i] == [] \
                and tq_prediction_group_list[i] == [] \
                and fq_near_d_prediction_group_list[area_id_next] == []:
            adjust_score_fq_dis_list[i] = 0
        else:
            adjust_score_fq_dis_list[i] = score_fq_dis_list[i]

    # integrate

    local_tqe_list = np.arange(gt_num, dtype=np.float64)

    local_cap_list = np.arange(gt_num, dtype=np.float64)
    local_near_detection_list = np.arange(gt_num, dtype=np.float64)
    local_false_alarm_list = np.arange(gt_num, dtype=np.float64)

    gt_detected_num = 0

    # 7th circle
    for i in range(gt_num):
        precision_tq_pred_group = tq_prediction_group_list[i]

        pred_group_integral_recall_tq = 0

        if precision_tq_pred_group != []:
            gt_detected_num += 1
        for j, basic_interval in enumerate(precision_tq_pred_group):
            if basic_interval != []:
                cal_integral_basic_interval_gt_recall = (basic_interval[1] - basic_interval[0])
                pred_group_integral_recall_tq += cal_integral_basic_interval_gt_recall


        if pred_group_integral_recall_tq > 0:
            detected_score = 1
        else:
            detected_score = 0

        tq_recall = detected_score

        adjust_score_fq_near = adjust_score_fq_near_list[i]

        adjust_score_fq_dis = adjust_score_fq_dis_list[i]

        local_tqe = cal_local_dqe(tq_recall, adjust_score_fq_near, adjust_score_fq_dis)

        local_tqe_list[i] = local_tqe

        if cal_components:
            local_cap_list[i] = tq_recall

            local_near_detection_list[i] = adjust_score_fq_near
            local_false_alarm_list[i] = adjust_score_fq_dis

    if cal_components:
        return {
            "seq_dqe_local_list": local_tqe_list,

            "seq_cap_local_list": local_cap_list,
            "seq_near_miss_local_list": local_near_detection_list,
            "seq_false_alarm_local_list": local_false_alarm_list,
        }
    else:
        return {
            "seq_dqe_local_list": local_tqe_list,
        }


def SDQE(y_true, binary_predicted, near_single_side_range=125, cal_components=False):
    """
    Evaluate binary detection results and compute the final single-threshold DQE score.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels for the time series.
    binary_predicted : np.ndarray
        Binary detections of an algorithm by applying a threshold.
    near_single_side_range : float
        Half-width of the temporal window used to define near-miss regions;
        detections within this window are treated as relevant near-misses.
    cal_components : bool
        If True, also return the three DQE component scores (DQE-cap, DQE-nm, DQE-fa);
        otherwise return only the aggregated single-threshold DQE.

    Returns
    -------
    dqe_res_ts : dict
            dqe : float
                Final threshold-free DQE score.
            dqe_cap : float, optional
                DQE-cap (returned only when cal_components=True).
            dqe_near_miss : float, optional
                DQE-nm (returned only when cal_components=True).
            dqe_false_alarm : float, optional
                DQE-fa (returned only when cal_components=True).
    """


    ts_len = len(y_true)
    gt_interval_ranges = convert_vector_to_events_dqe(y_true)
    gt_num = len(gt_interval_ranges)

    pred_interval_ranges = convert_vector_to_events_dqe(binary_predicted)
    dqe_local_list_res_ts_single_thresh = DQE_section(gt_interval_ranges,
                                                      pred_interval_ranges,
                                                      ts_len,
                                                      gt_num,
                                                      near_single_side_range=near_single_side_range,
                                                      cal_components=cal_components)

    dqe_single = np.array(dqe_local_list_res_ts_single_thresh["seq_dqe_local_list"]).mean()

    if cal_components:
        dqe_cap_single = np.array(dqe_local_list_res_ts_single_thresh["seq_cap_local_list"]).mean()
        dqe_near_miss_single = np.array(dqe_local_list_res_ts_single_thresh["seq_near_miss_local_list"]).mean()
        dqe_false_alarm_single = np.array(dqe_local_list_res_ts_single_thresh["seq_false_alarm_local_list"]).mean()

    if cal_components:
        # dqe in single ts
        dqe_res_ts = {
            "dqe": dqe_single,

            "dqe_cap": dqe_cap_single,
            "dqe_near_miss": dqe_near_miss_single,
            "dqe_false_alarm": dqe_false_alarm_single,
        }
    else:
        dqe_res_ts = {
            "dqe": dqe_single,
        }
    return dqe_res_ts


def DQE(y_true, y_score, near_single_side_range=125, thresh_num=100, cal_components=False, cal_multi_ts=False):
    """
    Evaluate detection quality evaluation score in a threshold-free manner.

    The function performs all prerequisite steps for DQE:
    1. Region partitioning.
    2. Getting local detection event groups.
    3. Calculating threshold-dependent DQE (quality score for each anomaly).
    4. Averaging over all thresholds to produce the final threshold-free DQE score.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels for the time series.
    y_score : np.ndarray
        Algorithm output scores normalized to [0, 1], indicating anomaly likelihood.
    near_single_side_range : float
        Half-width of the temporal window used to define near-miss regions;
        detections within this window are treated as relevant near-misses.
    thresh_num : int
        Number of thresholds to evaluate. Larger values yield finer-grained results
        at the cost of higher runtime.
    cal_components : bool
        If True, also return the three DQE component scores (DQE-cap, DQE-nm, DQE-fa);
        otherwise return only the aggregated DQE.

    Returns
    -------
    dqe_res_ts : dict
        dqe : float
            Final threshold-free DQE score.
        dqe_cap : float, optional
            DQE-cap (returned only when cal_components=True).
        dqe_near_miss : float, optional
            DQE-nm (returned only when cal_components=True).
        dqe_false_alarm : float, optional
            DQE-fa (returned only when cal_components=True).
    """

    ts_len = len(y_true)

    thresholds = np.linspace(1, 0, thresh_num + 1)[:-1]

    # array -> interval_ranges
    gt_interval_ranges = convert_vector_to_events_dqe(y_true)

    gt_num = len(gt_interval_ranges)

    dqe_matrix = []
    if cal_components:
        dqe_matrix_cap = []
        dqe_matrix_near_miss = []
        dqe_matrix_false_alarm = []

    for idx, threshold in enumerate(thresholds):
        # score->binary array
        binary_predicted = (y_score >= threshold).astype(int)
        pred_interval_ranges = convert_vector_to_events_dqe(binary_predicted)

        # area list
        fq_dis_section_list = [[] for _ in range(gt_num + 1)]  # index:0-n
        fq_dis_e_section_list = [[] for _ in range(gt_num)]  # index:0-(n-1)
        fq_dis_d_section_list = [[] for _ in range(gt_num + 1)]  # index:1-n
        fq_near_e_section_list = [[] for _ in range(gt_num)]  # index:0-(n-1)
        fq_near_d_section_list = [[] for _ in range(gt_num + 1)]  # index:1-n

        split_line_set = SortedSet()
        tq_section_list = gt_interval_ranges
        for i, tq_section_i in enumerate(tq_section_list):
            # tq_i
            tq_section_i_start, tq_section_i_end = tq_section_i

            # fq_distant_i_mid
            fq_dis_section_i_mid = None
            if i == 0:
                if gt_num == 1:
                    # get position
                    # fq_near_early_i position
                    fq_near_e_section_i_start = max(tq_section_i_start - near_single_side_range, 0)
                    fq_near_e_section_i_end = tq_section_i_start

                    # fq_near_delay_i_next position
                    fq_near_d_section_i_next_start = tq_section_i_end
                    fq_near_d_section_i_next_end = min(tq_section_i_end + near_single_side_range, ts_len)

                    # fq_near
                    # fq_near_early_i
                    fq_near_e_section_list[i] = [fq_near_e_section_i_start, fq_near_e_section_i_end]


                    # next section
                    # fq_near_delay_i_next
                    fq_near_d_section_list[i + 1] = [fq_near_d_section_i_next_start,
                                                     fq_near_d_section_i_next_end]

                    # fq_distant
                    # fq_distant_early_i
                    fq_dis_e_section_list[i] = [0, fq_near_e_section_i_start]

                    # next section
                    # fq_distant_delay_i_next
                    fq_dis_d_section_list[i + 1] = [fq_near_d_section_i_next_end, ts_len]
                else:
                    # get position
                    fq_near_e_section_i_start = max(tq_section_i_start - near_single_side_range, 0)
                    fq_near_e_section_i_end = tq_section_i_start

                    tq_section_i_next_start, tq_section_i_next_end = tq_section_list[i + 1]

                    fq_near_d_section_i_next_start = tq_section_i_end
                    fq_near_d_section_i_next_end = min(tq_section_i_end + near_single_side_range,
                                                       tq_section_i_next_start)

                    # fq_near
                    fq_near_e_section_list[i] = [fq_near_e_section_i_start, fq_near_e_section_i_end]

                    # next section
                    fq_near_d_section_list[i + 1] = [fq_near_d_section_i_next_start, fq_near_d_section_i_next_end]

                    # fq_distant
                    # fq_distant_early_i
                    fq_dis_e_section_list[i] = [0, fq_near_e_section_i_start]
            elif i == gt_num - 1:
                # get position
                tq_section_i_last_start, tq_section_i_last_end = tq_section_list[i - 1]

                fq_near_e_section_i_start = max(tq_section_i_start - near_single_side_range, tq_section_i_last_end)
                fq_near_e_section_i_end = tq_section_i_start

                fq_near_d_section_i_next_start = tq_section_i_end
                fq_near_d_section_i_next_end = min(tq_section_i_end + near_single_side_range, ts_len)

                fq_near_d_i_end = fq_near_d_section_list[i][1]

                # fq_near
                fq_near_e_section_list[i] = [fq_near_e_section_i_start, fq_near_e_section_i_end]
                fq_near_d_section_list[i + 1] = [fq_near_d_section_i_next_start, fq_near_d_section_i_next_end]

                # fq_distant
                fq_dis_section_i_mid = (fq_near_d_i_end + fq_near_e_section_i_start) / 2
                fq_dis_d_section_list[i] = [fq_near_d_i_end, fq_dis_section_i_mid]
                fq_dis_e_section_list[i] = [fq_dis_section_i_mid, fq_near_e_section_i_start]

                # next section
                fq_dis_d_section_list[i + 1] = [fq_near_d_section_i_next_end, ts_len]
            else:
                # get position
                tq_section_i_last_start, tq_section_i_last_end = tq_section_list[i - 1]
                tq_section_i_next_start, tq_section_i_next_end = tq_section_list[i + 1]

                fq_near_e_section_i_start = max(tq_section_i_start - near_single_side_range, tq_section_i_last_end)
                fq_near_e_section_i_end = tq_section_i_start

                fq_near_d_section_i_next_start = tq_section_i_end
                fq_near_d_section_i_next_end = min(tq_section_i_end + near_single_side_range, tq_section_i_next_start)

                fq_near_d_i_end = fq_near_d_section_list[i][1]

                # fq_near
                fq_near_e_section_list[i] = [fq_near_e_section_i_start, fq_near_e_section_i_end]
                fq_near_d_section_list[i + 1] = [fq_near_d_section_i_next_start, fq_near_d_section_i_next_end]

                # fq_distant
                fq_dis_section_i_mid = (fq_near_d_i_end + fq_near_e_section_i_start) / 2
                fq_dis_d_section_list[i] = [fq_near_d_i_end, fq_dis_section_i_mid]
                fq_dis_e_section_list[i] = [fq_dis_section_i_mid, fq_near_e_section_i_start]

            # add split line
            split_line_set.add(fq_near_e_section_i_start)

            split_line_set.add(tq_section_i_start)
            split_line_set.add(tq_section_i_end)

            split_line_set.add(fq_near_d_section_i_next_end)

            if i > 0:
                if fq_dis_section_i_mid != None:
                    split_line_set.add(fq_dis_section_i_mid)

        partition_res = {
            "fq_dis_section_list": fq_dis_section_list,
            "fq_dis_e_section_list": fq_dis_e_section_list,
            "fq_dis_d_section_list": fq_dis_d_section_list,
            "fq_near_e_section_list": fq_near_e_section_list,
            "fq_near_d_section_list": fq_near_d_section_list,

            "split_line_set": split_line_set
        }

        # add single thresh
        dqe_local_list_res_ts_single_thresh = DQE_section(gt_interval_ranges,
                                                          pred_interval_ranges,
                                                          ts_len,
                                                          gt_num,
                                                          near_single_side_range=near_single_side_range,
                                                          cal_components=cal_components,
                                                          partition_res=partition_res,
                                                          )

        dqe_matrix.append(dqe_local_list_res_ts_single_thresh["seq_dqe_local_list"])
        if cal_components:
            dqe_matrix_cap.append(dqe_local_list_res_ts_single_thresh["seq_cap_local_list"])
            dqe_matrix_near_miss.append(dqe_local_list_res_ts_single_thresh["seq_near_miss_local_list"])
            dqe_matrix_false_alarm.append(dqe_local_list_res_ts_single_thresh["seq_false_alarm_local_list"])

    # return
    if cal_multi_ts:
        if cal_components:
            return {
                "dqe_matrix": dqe_matrix,

                "matrix_cap": dqe_matrix_cap,
                "matrix_near_miss": dqe_matrix_near_miss,
                "matrix_false_alarm": dqe_matrix_false_alarm,
            }
        else:
            return {
                "dqe_matrix": dqe_matrix,
            }
    else:
        # weight sum
        # v_h
        dqe_matrix_v = np.array(dqe_matrix).mean(axis=0)

        if cal_components:
            dqe_cap_v = np.array(dqe_matrix_cap).mean(axis=0)
            dqe_near_miss_v = np.array(dqe_matrix_near_miss).mean(axis=0)
            dqe_false_alarm_v = np.array(dqe_matrix_false_alarm).mean(axis=0)

        dqe = dqe_matrix_v.mean()

        if cal_components:
            dqe_cap = dqe_cap_v.mean()
            dqe_near_miss = dqe_near_miss_v.mean()
            dqe_false_alarm = dqe_false_alarm_v.mean()

        if cal_components:
            # dqe in single ts
            dqe_res_ts = {
                "dqe": dqe,

                "dqe_cap": dqe_cap,
                "dqe_near_miss": dqe_near_miss,
                "dqe_false_alarm": dqe_false_alarm,
            }
        else:
            dqe_res_ts = {
                "dqe": dqe,
            }
        return dqe_res_ts


def cal_dqe_matrix(ts_dict: dict, output_dict: dict, gt_dict: dict, thresh_num=100, cal_components=False,
                   method_name=None, single_slidingWindow=None):
    # cal local dqe for each anomaly event in a single ts

    dqe_chunks_global = []

    cap_chunks_global = []
    proximity_chunks_global = []
    false_alarm_chunks_global = []

    for i, (ts_file_name, ts_data) in enumerate(ts_dict.items()):
        single_ts = np.array(ts_data)
        single_gt = np.array(gt_dict[ts_file_name])
        single_output = np.array(output_dict[ts_file_name])

        # if calculated, pass here
        if single_slidingWindow == None:
            single_slidingWindow = find_length_rank(single_ts, rank=1)

        dqe_matrix_res = DQE(single_gt, single_output,
                             near_single_side_range=single_slidingWindow / 2,
                             cal_multi_ts=True,
                             cal_components=True,
                             thresh_num=thresh_num)
        dqe_matrix = np.array(dqe_matrix_res["dqe_matrix"])

        if cal_components:
            matrix_cap = np.array(dqe_matrix_res["matrix_cap"])
            matrix_proximity = np.array(dqe_matrix_res["matrix_near_miss"])
            matrix_false_alarm = np.array(dqe_matrix_res["matrix_false_alarm"])

        dqe_chunks_global.append(dqe_matrix)

        if cal_components:
            cap_chunks_global.append(matrix_cap)
            proximity_chunks_global.append(matrix_proximity)
            false_alarm_chunks_global.append(matrix_false_alarm)

    if cal_components:
        chunks_global_dict = {
            "dqe_chunks_global": dqe_chunks_global,
            "cap_chunks_global": cap_chunks_global,
            "proximity_chunks_global": proximity_chunks_global,
            "false_alarm_chunks_global": false_alarm_chunks_global,
        }
    else:
        chunks_global_dict = {
            "dqe_chunks_global": dqe_chunks_global,
        }

    return chunks_global_dict


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


def DQE_multi_data(ts_dict: dict, output_dict: dict, gt_dict: dict, thresh_num=100, cal_components=False,
                   method_name=None, return_each_anomaly_score=False):
    # merge local dqe for each ts

    # cal matrix
    chunks_global_dict = cal_dqe_matrix(ts_dict, output_dict, gt_dict, thresh_num=thresh_num,
                                        cal_components=cal_components, method_name=method_name)

    # cal multi data dqe
    dqe_chunks_global = chunks_global_dict["dqe_chunks_global"]

    if cal_components:
        cap_chunks_global = chunks_global_dict["cap_chunks_global"]
        proximity_chunks_global = chunks_global_dict["proximity_chunks_global"]
        false_alarm_chunks_global = chunks_global_dict["false_alarm_chunks_global"]

    # concat
    dqe_matrix_global = np.concatenate(dqe_chunks_global, axis=1)

    if cal_components:
        cap_matrix_global = np.concatenate(cap_chunks_global, axis=1)
        proximity_matrix_global = np.concatenate(proximity_chunks_global, axis=1)
        false_alarm_matrix_global = np.concatenate(false_alarm_chunks_global, axis=1)

    # local threshold-dependent dqe across all thresholds
    dqe_matrix_global_mean_v = np.mean(dqe_matrix_global, axis=0)

    if cal_components:
        cap_matrix_global_mean_v = np.mean(cap_matrix_global, axis=0)
        proximity_matrix_global_mean_v = np.mean(proximity_matrix_global, axis=0)
        false_alarm_matrix_global_mean_v = np.mean(false_alarm_matrix_global, axis=0)

    # dqe across all anomaly events
    dqe = dqe_matrix_global_mean_v.mean()

    if cal_components:
        cap = cap_matrix_global_mean_v.mean()
        proximity = proximity_matrix_global_mean_v.mean()
        false_alarm = false_alarm_matrix_global_mean_v.mean()

    if cal_components:
        if return_each_anomaly_score:
            return {"dqe": dqe,

                    "cap": cap,
                    "near_miss": proximity,
                    "false_alarm": false_alarm,

                    "dqe_list": dqe_matrix_global_mean_v.tolist(),

                    "cap_list": cap_matrix_global_mean_v.tolist(),
                    "near_miss_list": proximity_matrix_global_mean_v.tolist(),
                    "false_alarm_list": false_alarm_matrix_global_mean_v.tolist(),
                    }
        else:
            return {"dqe": dqe,

                    "cap": cap,
                    "near_miss": proximity,
                    "false_alarm": false_alarm,
                    }
    else:
        if return_each_anomaly_score:
            return {"dqe": dqe,

                    "dqe_list": dqe_matrix_global_mean_v,
                    }
        else:
            return {"dqe": dqe,
                    }


def cal_local_dqe(row_mean_real_detection,
                  row_mean_near_detection,
                  row_mean_false_alarm):
    # Integrating dqe-cap, dqe-nm, and dqe-fa into a unified evaluation metric ( threshold-dependent local dqe).
    local_dqe_value = (row_mean_near_detection
                       + row_mean_real_detection) / 2 * row_mean_false_alarm
    local_dqe_value = math.sqrt(local_dqe_value)
    return local_dqe_value


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