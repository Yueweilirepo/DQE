#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###########################
import numpy as np
import scipy.integrate as spi

from sortedcontainers import SortedSet
from copy import deepcopy
from evaluation.slidingWindows import find_length_rank

# DQE
w_tq = 1 / 3  # tq weight
w_fq_near = 1 / 3  # fq_near weight
w_fq_dis = 1 - w_tq - w_fq_near  # fq_distant weight
w_fq_near_delay = 1 / 2  # fq_near_delayed weight
w_fq_near_early = 1 - w_fq_near_delay  # fq_near_early weight

beta = 2  # beta in paper

fq_near_mean_proximity_gama = 1  # gama of mean proximity in near fq
fq_near_closest_onset_gama = 1  # gama of closest onset response in near fq
fq_near_total_duration_gama = 1  # gama of total duration in near fq

fq_distant_mean_proximity_gama = 1  # gama of mean proximity in distant fq
fq_distant_closest_onset_gama = 1  # gama of closest onset response in distant fq
fq_distant_total_duration_gama = 2  # gama of total duration in distant fq

distant_method = "whole"  # fq_distant strategy_1 whole

distant_direction = "both"  # fq_distant direction

use_detection_rate = True

weight_sum_method = "triangle"

parameter_dict = {
    # weight
    "w_tq": w_tq,
    "w_fq_near": w_fq_near,
    "w_fq_near_early": w_fq_near_early,
    "w_fq_near_delay": w_fq_near_delay,

    "beta": beta,
    "distant_direction": distant_direction,

    # strategy configuration
    "distant_method": distant_method,
    "use_detection_rate": use_detection_rate,
    "weight_sum_method": weight_sum_method,

    # gama
    "fq_near_mean_proximity_gama": fq_near_mean_proximity_gama,
    "fq_near_closest_onset_gama": fq_near_closest_onset_gama,
    "fq_near_total_duration_gama": fq_near_total_duration_gama,
    "fq_distant_mean_proximity_gama": fq_distant_mean_proximity_gama,
    "fq_distant_closest_onset_gama": fq_distant_closest_onset_gama,
    "fq_distant_total_duration_gama": fq_distant_total_duration_gama,

}


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
    # Convert the set b to a sorted list

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


def cal_integral_fq_near_e_section_power_func(area, parameter_a, parameter_beta, parameter_w_tq=0,
                                              parameter_w_fq_near=0,
                                              parameter_near_single_side_range=None, area_end=None):
    """
        Integrate a power-law weight function over a given interval.

        The integrand is
            f(x) = |a * ((area_end - x) - near_range)^beta|,
        where `near_range` is provided via `parameter_near_single_side_range`.
        All other parameters are passed through but **not used** in the current
        integrand definition (they are kept for API consistency).

        Parameters
        ----------
        area : list or tuple of float
            Integration limits [x_start, x_end].
        parameter_a : float
            Leading coefficient of the power-law term.
        parameter_beta : float
            Exponent of the power-law term.
        parameter_w_tq : float, optional
            Placeholder parameter (unused in integrand).
        parameter_w_fq_near : float, optional
            Placeholder parameter (unused in integrand).
        parameter_near_single_side_range : float, optional
            Offset subtracted inside the power term.
        area_end : float, optional
            Reference position used to construct the term (area_end - x).

        Returns
        -------
        float
            Definite integral of the weight function over `area`.
        """
    f = lambda x, parameter_a, parameter_beta, parameter_w_tq, parameter_w_fq_near, \
               parameter_near_single_side_range, area_end: abs(
        parameter_a * pow((area_end - x) - parameter_near_single_side_range, parameter_beta))

    result, error = spi.quad(f, area[0], area[1],
                             args=(parameter_a, parameter_beta, parameter_w_tq, parameter_w_fq_near,
                                   parameter_near_single_side_range, area_end))
    return result


def cal_integral_fq_near_d_section_power_func(area, parameter_a, parameter_beta, parameter_w_tq=0,
                                              parameter_near_single_side_range=None, area_start=None):
    """
        Integrate a power-law weight function over a given interval (after some
        reference point).

        The integrand is
            f(x) = |a * ((x - area_start) - near_range)^beta|,
        where `near_range` is supplied via `parameter_near_single_side_range`.
        The parameter `parameter_w_tq` is accepted for API consistency but is
        **not used** inside the integrand.

        Parameters
        ----------
        area : list or tuple of float
            Integration limits [x_start, x_end].
        parameter_a : float
            Leading coefficient of the power-law term.
        parameter_beta : float
            Exponent of the power-law term.
        parameter_w_tq : float, optional
            Placeholder parameter (unused in integrand).
        parameter_near_single_side_range : float, optional
            Offset subtracted inside the power term.
        area_start : float, optional
            Reference position used to construct the term (x - area_start).

        Returns
        -------
        float
            Definite integral of the weight function over `area`.
        """
    f = lambda x, parameter_a, parameter_beta, parameter_w_tq, \
               parameter_near_single_side_range, area_start: abs(
        parameter_a * pow((x - area_start) - parameter_near_single_side_range, parameter_beta))

    result, error = spi.quad(f, area[0], area[1], args=(parameter_a, parameter_beta, parameter_w_tq,
                                                        parameter_near_single_side_range, area_start))
    return result


def cal_integral_tq_section_power_func(area, parameter_coefficient, parameter_beta, parameter_w_tq=0, area_end=None):
    """
    Integrate a shifted power-law function over a given interval.

    Integrand:
        f(x) = coefficient * (area_end - x)^beta  +  (1 - w_tq)

    The second term (1 - w_tq) acts as a constant offset that is independent
    of x; the power-law part vanishes as x approaches area_end.

    Parameters
    ----------
    area : list or tuple of float
        Integration limits [x_start, x_end].
    parameter_coefficient : float
        Leading coefficient for the power-law term.
    parameter_beta : float
        Exponent of the power-law term.
    parameter_w_tq : float, optional
        Weight factor subtracted from 1 to form the constant offset.
    area_end : float, optional
        Reference position used to construct (area_end - x).

    Returns
    -------
    float
        Definite integral of the function over `area`.
    """
    f = lambda x, parameter_coefficient, parameter_beta, area_end: parameter_coefficient * pow(area_end - x,
                                                                                               parameter_beta) + (
                                                                               1 - parameter_w_tq)
    result, error = spi.quad(f, area[0], area[1], args=(parameter_coefficient, parameter_beta, area_end))
    return result


def cal_power_function_parameter(parameter_gama, parameter_w_tq, parameter_w_fq_near, gt_len):
    """
        Compute the leading coefficient (parameter_b) for a power-law weight function.

        Formula:
            b = w_tq / (gt_len ^ gamma)

        The parameters `parameter_w_fq_near` and `parameter_gama` are accepted
        for API consistency; only `parameter_w_tq` and `gt_len` are used in the
        actual calculation.

        Parameters
        ----------
        parameter_gama : float
            Exponent denominator (unused in computation but kept for interface uniformity).
        parameter_w_tq : float
            Weight factor assigned to the true-quality component.
        parameter_w_fq_near : float
            Placeholder parameter (not used in this calculation).
        gt_len : float
            Length or scale factor of the ground-truth interval.

        Returns
        -------
        float
            Computed coefficient `parameter_b` for the power-law term.
        """
    parameter_b = parameter_w_tq / pow(gt_len, parameter_gama)
    return parameter_b


def compute_f1_score(precision, recall):
    # Calculate the F1 score from precision and recall
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


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
    # gama = 2
    parameter_a = 1 / max_area_len ** gama
    score = parameter_a * (max_area_len - x) ** gama
    return score


def DQE_F1_section(tq_section_list, prediction_section_list, ts_len, gt_num=None, parameter_dict=parameter_dict,
                   near_single_side_range=125, max_fq_distant_len=None, cal_components=False, partition_res=None,
                   integral_tq_section_res=None):
    """
    Evaluate binary anomaly-detection predictions on time-series data and compute the final DQE-F1 score.

    The function completes the full DQE pipeline:
    1. Section partitioning
    2. Local detection-event grouping
    3. Single-threshold Precision-TQ, Recall-TQ, near-FQ and distant-FQ calculation
    4. Near-FQ and distant-FQ adjustment
    5. Weighted-sum DQE-F1 aggregation

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
    parameter_dict : dict
        Application-specific configuration parameters.
    near_single_side_range : int
        Half-width of the temporal window used to define near-FQ regions;
        detections within this window are treated as relevant near-misses.
    max_fq_distant_len : int
        Maximum length among all distant-FQ sections.
    cal_components : bool
        If True, also return the three DQE-F1 sub-scores (TQ, near-FQ, distant-FQ);
        otherwise return only the aggregated DQE-F1.
    partition_res : dict
        Threshold-free DQE: partition result pre-computed and passed in.
        Threshold-dependent DQE-F1: partition computed inside this call.
    integral_tq_section_res : dict
        Threshold-free DQE: integral over the whole TQ section (constant for all
        thresholds) pre-computed and passed in.
        Threshold-dependent DQE-F1: integral computed inside this call.

    Returns
    -------
    dqe_f1 : float
        Final DQE-F1 score for the given threshold.
    dqe_f1_w_tq : float, optional
        TQ-weighted DQE-F1 sub-score (returned only when cal_components=True).
    dqe_f1_w_fq_near : float, optional
        Near-FQ DQE-F1 sub-score (returned only when cal_components=True).
    dqe_f1_w_fq_dis : float, optional
        Distant-FQ DQE-F1 sub-score (returned only when cal_components=True).
    """

    if not gt_num:
        gt_num = len(tq_section_list)

    weight_fq_near_early = parameter_dict["w_fq_near_early"]
    weight_fq_near_delay = parameter_dict["w_fq_near_delay"]

    if max_fq_distant_len is None:
        # o(m)
        max_fq_distant_len = find_max_fq_distant_length_in_single_ts(tq_section_list, ts_len, gt_num,
                                                                     parameter_dict=parameter_dict,
                                                                     near_single_side_range=near_single_side_range)

    distant_method = parameter_dict["distant_method"]
    distant_direction = parameter_dict["distant_direction"]

    use_detection_rate = parameter_dict["use_detection_rate"]

    beta = parameter_dict["beta"]

    fq_near_total_duration_gama = parameter_dict["fq_near_total_duration_gama"]
    fq_near_mean_proximity_gama = parameter_dict["fq_near_mean_proximity_gama"]
    fq_near_closest_onset_gama = parameter_dict["fq_near_closest_onset_gama"]
    fq_distant_total_duration_gama = parameter_dict["fq_distant_total_duration_gama"]
    fq_distant_mean_proximity_gama = parameter_dict["fq_distant_mean_proximity_gama"]
    fq_distant_closest_onset_gama = parameter_dict["fq_distant_closest_onset_gama"]

    # weight relation check
    # section partition (first)
    if partition_res is not None:
        # area list
        fq_dis_section_list = partition_res["fq_dis_section_list"]  # index:0-n
        fq_dis_e_section_list = partition_res["fq_dis_e_section_list"]  # index:0-(n-1)
        fq_dis_d_section_list = partition_res["fq_dis_d_section_list"]  # index:1-n
        fq_near_e_section_list = partition_res["fq_near_e_section_list"]  # index:0-(n-1)
        fq_near_d_section_list = partition_res["fq_near_d_section_list"]  # index:1-n

        split_line_set = partition_res["split_line_set"]

    else:
        # area list
        fq_dis_section_list = [[] for _ in range(gt_num + 1)]  # index:0-n
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
                    if distant_method == "whole":
                        # fq_distant_i
                        fq_dis_section_list[i] = [0, fq_near_e_section_i_start]

                        # next section
                        # fq_distant_i_next
                        fq_dis_section_list[i + 1] = [fq_near_d_section_i_next_end, ts_len]
                    else:
                        # fq_distant_early_i
                        fq_dis_e_section_list[i] = [0, fq_near_e_section_i_start]

                        # next section
                        # fq_distant_delay_i_next
                        fq_dis_d_section_list[i + 1] = [fq_near_d_section_i_next_end, ts_len]

                    # merge section
                    if weight_fq_near_early <= 0:  # merge fq_near_early to fq_distant
                        # fq_near_early_i = empty
                        fq_near_e_section_list[i] = [tq_section_i_start, tq_section_i_start]
                        if distant_method == "whole":
                            # fq_distant_i
                            fq_dis_section_list[i] = [0, tq_section_i_start]
                        else:
                            # fq_distant_early_i
                            fq_dis_e_section_list[i] = [0, tq_section_i_start]
                    if weight_fq_near_delay <= 0:  # merge fq_near_delay to fq_distant
                        # fq_near_delay_i_next = empty
                        fq_near_d_section_list[i + 1] = [tq_section_i_end, tq_section_i_end]

                        if distant_method == "whole":
                            # next section
                            fq_dis_section_list[i + 1] = [tq_section_i_end, ts_len]
                        else:
                            # next section
                            # fq_distant_delay_i_next
                            fq_dis_d_section_list[i + 1] = [tq_section_i_end, ts_len]
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
                    if distant_method == "whole":
                        # fq_distant_i
                        fq_dis_section_list[i] = [0, fq_near_e_section_i_start]
                    else:
                        # fq_distant_early_i
                        fq_dis_e_section_list[i] = [0, fq_near_e_section_i_start]

                    # merge section
                    if weight_fq_near_early <= 0:
                        fq_near_e_section_list[i] = [tq_section_i_start, tq_section_i_start]
                        if distant_method == "whole":
                            fq_dis_section_list[i] = [0, tq_section_i_start]
                        else:
                            fq_dis_e_section_list[i] = [0, tq_section_i_start]
                    if weight_fq_near_delay <= 0:
                        fq_near_d_section_list[i + 1] = [tq_section_i_end, tq_section_i_end]
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
                if distant_method == "whole":
                    fq_dis_section_list[i] = [fq_near_d_i_end, fq_near_e_section_i_start]

                    # next section
                    fq_dis_section_list[i + 1] = [fq_near_d_section_i_next_end, ts_len]
                else:
                    fq_dis_section_i_mid = (fq_near_d_i_end + fq_near_e_section_i_start) / 2
                    fq_dis_d_section_list[i] = [fq_dis_section_i_mid, fq_near_e_section_i_start]
                    fq_dis_e_section_list[i] = [fq_near_d_i_end, fq_dis_section_i_mid]

                    # next section
                    fq_dis_d_section_list[i + 1] = [fq_near_d_section_i_next_end, ts_len]

                # merge section
                if weight_fq_near_early <= 0:
                    fq_near_e_section_list[i] = [tq_section_i_start, tq_section_i_start]
                    if distant_method == "whole":
                        fq_dis_section_list[i] = [fq_near_d_i_end, tq_section_i_start]
                    else:
                        fq_dis_section_i_mid = (fq_near_d_i_end + tq_section_i_start) / 2
                        fq_dis_d_section_list[i] = [fq_dis_section_i_mid, tq_section_i_start]
                        fq_dis_e_section_list[i] = [fq_near_d_i_end, fq_dis_section_i_mid]
                if weight_fq_near_delay <= 0:
                    fq_near_d_section_list[i + 1] = [tq_section_i_end, tq_section_i_end]
                    if distant_method == "whole":
                        fq_dis_section_list[i] = [tq_section_i_last_end, fq_near_e_section_i_start]
                        # next section
                        fq_dis_section_list[i + 1] = [tq_section_i_end, ts_len]
                    else:
                        fq_dis_section_i_mid = (tq_section_i_last_end + fq_near_e_section_i_start) / 2
                        fq_dis_d_section_list[i] = [fq_dis_section_i_mid, fq_near_e_section_i_start]
                        fq_dis_e_section_list[i] = [tq_section_i_last_end, fq_dis_section_i_mid]

                        # next section
                        fq_dis_d_section_list[i + 1] = [tq_section_i_end, ts_len]
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
                if distant_method == "whole":
                    fq_dis_section_list[i] = [fq_near_d_i_end, fq_near_e_section_i_start]
                else:
                    fq_dis_section_i_mid = (fq_near_d_i_end + fq_near_e_section_i_start) / 2
                    fq_dis_d_section_list[i] = [fq_dis_section_i_mid, fq_near_e_section_i_start]
                    fq_dis_e_section_list[i] = [fq_near_d_i_end, fq_dis_section_i_mid]

                # merge section
                if weight_fq_near_early <= 0:
                    fq_near_e_section_list[i] = [tq_section_i_start, tq_section_i_start]
                    if distant_method == "whole":
                        fq_dis_section_list[i] = [fq_near_d_i_end, tq_section_i_start]
                    else:
                        fq_dis_section_i_mid = (fq_near_d_i_end + tq_section_i_start) / 2
                        fq_dis_d_section_list[i] = [fq_dis_section_i_mid, tq_section_i_start]
                        fq_dis_e_section_list[i] = [fq_near_d_i_end, fq_dis_section_i_mid]
                if weight_fq_near_delay <= 0:
                    fq_near_d_section_list[i + 1] = [tq_section_i_end, tq_section_i_end]
                    if distant_method == "whole":
                        fq_dis_section_list[i] = [tq_section_i_last_end, fq_near_e_section_i_start]
                    else:
                        fq_dis_section_i_mid = (tq_section_i_last_end + fq_near_e_section_i_start) / 2
                        fq_dis_d_section_list[i] = [fq_dis_section_i_mid, fq_dis_section_list[i][1]]
                        fq_dis_e_section_list[i] = [fq_dis_section_list[i][0], fq_dis_section_i_mid]

            # add split line
            if weight_fq_near_early > 0:
                split_line_set.add(fq_near_e_section_i_start)

            split_line_set.add(tq_section_i_start)
            split_line_set.add(tq_section_i_end)

            if weight_fq_near_delay > 0:
                split_line_set.add(fq_near_d_section_i_next_end)

            if distant_method == "split":
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
    if distant_method == "whole":
        fq_dis_prediction_group_list = [[] for _ in range(gt_num + 1)]
    else:
        fq_dis_e_prediction_group_list = [[] for _ in range(gt_num)]
        fq_dis_d_prediction_group_list = [[] for _ in range(gt_num + 1)]

    # get local prediction event group (third circle,prediction_event_num*gt_num*4)
    # for i in
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
        if distant_method == "whole":
            for j, area in enumerate(fq_dis_section_list):
                if pred_in_area(basic_interval, area):
                    fq_dis_prediction_group_list[j].append(basic_interval)
        else:
            for j, area in enumerate(fq_dis_e_section_list):
                if pred_in_area(basic_interval, area):
                    fq_dis_e_prediction_group_list[j].append(basic_interval)

            for j, area in enumerate(fq_dis_d_section_list):
                if pred_in_area(basic_interval, area):
                    fq_dis_d_prediction_group_list[j].append(basic_interval)


    # fq_near_early
    # total duration
    td_fq_near_e_list = np.arange(gt_num, dtype=np.float64)
    # mean proximity
    mp_fq_near_e_list = np.arange(gt_num, dtype=np.float64)
    # closest_onset
    co_fq_near_e_list = np.arange(gt_num, dtype=np.float64)
    fq_near_e_score_list = np.arange(gt_num, dtype=np.float64)

    # fq_near_delay
    td_fq_near_d_list = np.arange(gt_num + 1, dtype=np.float64)
    mp_fq_near_d_list = np.arange(gt_num + 1, dtype=np.float64)
    co_fq_near_d_list = np.arange(gt_num + 1, dtype=np.float64)
    fq_near_d_score_list = np.arange(gt_num + 1, dtype=np.float64)

    # cal score function,tq_near score
    if integral_tq_section_res is not None:
        tq_recall_integral_list = integral_tq_section_res["tq_recall_integral_list"]

        func_parameter_list_precision_tq = integral_tq_section_res["func_parameter_list_precision_tq"]
        func_parameter_list_recall = integral_tq_section_res["func_parameter_list_recall"]
        func_parameter_list_precision_fq_near = integral_tq_section_res["func_parameter_list_precision_fq_near"]

        for i, area in enumerate(tq_section_list):
            area_id_next = i + 1
            # fq_near_early

            fq_near_e_area = fq_near_e_section_list[i]

            fq_near_e_end = fq_near_e_area[1]

            fq_near_e_pred_group = fq_near_e_prediction_group_list[i]

            # proximity
            p_fq_near_e_sum = 0
            # onset
            co_fq_near_e = 0
            # duration
            td_fq_near_e = 0
            fq_near_e_num = len(fq_near_e_pred_group)
            for interval_idx, basic_interval in enumerate(fq_near_e_pred_group):
                td_fq_near_e += basic_interval[1] - basic_interval[0]
                p_fq_near_e_sum += abs(fq_near_e_end - (basic_interval[1] + basic_interval[0]) / 2)
                if interval_idx == fq_near_e_num - 1:
                    co_fq_near_e = abs(fq_near_e_end - basic_interval[1])

            td_fq_near_e_list[i] = td_fq_near_e

            mp_fq_near_e = p_fq_near_e_sum / fq_near_e_num if fq_near_e_num != 0 else 0
            mp_fq_near_e_list[i] = mp_fq_near_e

            co_fq_near_e_list[i] = co_fq_near_e

            score_fq_near_e_mp = ddl_func(mp_fq_near_e, near_single_side_range,
                                          gama=fq_near_mean_proximity_gama) if near_single_side_range != 0 else 1
            score_fq_near_e_co = ddl_func(co_fq_near_e, near_single_side_range,
                                          gama=fq_near_closest_onset_gama) if near_single_side_range != 0 else 1
            score_fq_near_e_td = ddl_func(td_fq_near_e, near_single_side_range,
                                          gama=fq_near_total_duration_gama) if near_single_side_range != 0 else 1

            score_fq_near_e = score_fq_near_e_mp * \
                              score_fq_near_e_co * \
                              score_fq_near_e_td

            fq_near_e_score_list[i] = score_fq_near_e

            # fq_near_delay
            fq_near_d_area = fq_near_d_section_list[area_id_next]
            fq_near_d_start = fq_near_d_area[0]

            fq_near_d_pred_group = fq_near_d_prediction_group_list[area_id_next]

            p_fq_near_d_sum = 0
            co_fq_near_d = 0
            td_fq_near_d = 0

            fq_near_d_num = len(fq_near_d_pred_group)

            for interval_idx, basic_interval in enumerate(fq_near_d_pred_group):
                td_fq_near_d += basic_interval[1] - basic_interval[0]
                p_fq_near_d_sum += abs((basic_interval[1] + basic_interval[0]) / 2 - fq_near_d_start)

                if interval_idx == 0:
                    co_fq_near_d = abs(fq_near_d_start - basic_interval[0])

            td_fq_near_d_list[area_id_next] = td_fq_near_d

            mp_fq_near_d = p_fq_near_d_sum / fq_near_d_num if fq_near_d_num != 0 else 0
            mp_fq_near_d_list[area_id_next] = mp_fq_near_d

            co_fq_near_d_list[area_id_next] = co_fq_near_d

            score_fq_near_d_mp = ddl_func(mp_fq_near_d, near_single_side_range,
                                          gama=fq_near_mean_proximity_gama) if near_single_side_range != 0 else 1
            score_fq_near_d_co = ddl_func(co_fq_near_d, near_single_side_range,
                                          gama=fq_near_closest_onset_gama) if near_single_side_range != 0 else 1
            score_fq_near_d_td = ddl_func(td_fq_near_d, near_single_side_range,
                                          gama=fq_near_total_duration_gama) if near_single_side_range != 0 else 1

            score_fq_near_d = score_fq_near_d_mp * \
                              score_fq_near_d_co * \
                              score_fq_near_d_td

            fq_near_d_score_list[area_id_next] = score_fq_near_d
    else:
        tq_recall_integral_list = [[] for _ in range(gt_num)]

        func_parameter_list_precision_tq = [{} for _ in range(gt_num)]
        func_parameter_list_recall = [{} for _ in range(gt_num)]
        func_parameter_list_precision_fq_near = [{} for _ in range(gt_num)]

        for i, area in enumerate(tq_section_list):
            area_id_next = i + 1
            tq_section_len = area[1] - area[0]

            w_tq_recall = 1
            a_recall = cal_power_function_parameter(beta, w_tq_recall, 0, tq_section_len)
            w_tq_precision = 1 / 2

            c_precision = cal_power_function_parameter(beta, w_tq_precision, 0, tq_section_len)

            entire_section_integral_tq_recall = cal_integral_tq_section_power_func(area, a_recall, beta, \
                                                                                   parameter_w_tq=w_tq_recall, \
                                                                                   area_end=area[1])

            func_parameter_list_precision_tq[i]["coefficient"] = c_precision
            func_parameter_list_precision_tq[i]["beta"] = beta
            func_parameter_list_precision_tq[i]["w_tq"] = w_tq_precision

            func_parameter_list_recall[i]["coefficient"] = a_recall
            func_parameter_list_recall[i]["beta"] = beta
            func_parameter_list_recall[i]["w_tq"] = w_tq_recall

            # fq_near_early
            fq_near_e_area = fq_near_e_section_list[i]

            beta_early = beta
            coefficient_early = (1 - w_tq_precision) / pow(near_single_side_range, beta)

            # fq_near_delay
            fq_near_d_area = fq_near_d_section_list[area_id_next]

            func_parameter_list_precision_fq_near[i]["coefficient_early"] = coefficient_early
            func_parameter_list_precision_fq_near[i]["beta_early"] = beta_early

            tq_recall_integral_list[i] = entire_section_integral_tq_recall

            # fq_near_early

            fq_near_e_area = fq_near_e_section_list[i]

            fq_near_e_end = fq_near_e_area[1]

            fq_near_e_pred_group = fq_near_e_prediction_group_list[i]

            # proximity
            p_fq_near_e_sum = 0
            # onset
            co_fq_near_e = 0
            # duration
            td_fq_near_e = 0
            fq_near_e_num = len(fq_near_e_pred_group)
            for interval_idx, basic_interval in enumerate(fq_near_e_pred_group):
                td_fq_near_e += basic_interval[1] - basic_interval[0]
                p_fq_near_e_sum += abs(fq_near_e_end - (basic_interval[1] + basic_interval[0]) / 2)
                if interval_idx == fq_near_e_num - 1:
                    co_fq_near_e = abs(fq_near_e_end - basic_interval[1])

            td_fq_near_e_list[i] = td_fq_near_e

            mp_fq_near_e = p_fq_near_e_sum / fq_near_e_num if fq_near_e_num != 0 else 0
            mp_fq_near_e_list[i] = mp_fq_near_e

            co_fq_near_e_list[i] = co_fq_near_e

            score_fq_near_e_mp = ddl_func(mp_fq_near_e, near_single_side_range,
                                          gama=fq_near_mean_proximity_gama) if near_single_side_range != 0 else 1
            score_fq_near_e_co = ddl_func(co_fq_near_e, near_single_side_range,
                                          gama=fq_near_closest_onset_gama) if near_single_side_range != 0 else 1
            score_fq_near_e_td = ddl_func(td_fq_near_e, near_single_side_range,
                                          gama=fq_near_total_duration_gama) if near_single_side_range != 0 else 1

            score_fq_near_e = score_fq_near_e_mp * \
                              score_fq_near_e_co * \
                              score_fq_near_e_td

            fq_near_e_score_list[i] = score_fq_near_e

            # fq_near_delay
            fq_near_d_area = fq_near_d_section_list[area_id_next]
            fq_near_d_start = fq_near_d_area[0]

            fq_near_d_pred_group = fq_near_d_prediction_group_list[area_id_next]

            p_fq_near_d_sum = 0
            co_fq_near_d = 0
            td_fq_near_d = 0

            fq_near_d_num = len(fq_near_d_pred_group)

            for interval_idx, basic_interval in enumerate(fq_near_d_pred_group):
                td_fq_near_d += basic_interval[1] - basic_interval[0]
                p_fq_near_d_sum += abs((basic_interval[1] + basic_interval[0]) / 2 - fq_near_d_start)

                if interval_idx == 0:
                    co_fq_near_d = abs(fq_near_d_start - basic_interval[0])

            td_fq_near_d_list[area_id_next] = td_fq_near_d

            mp_fq_near_d = p_fq_near_d_sum / fq_near_d_num if fq_near_d_num != 0 else 0
            mp_fq_near_d_list[area_id_next] = mp_fq_near_d

            co_fq_near_d_list[area_id_next] = co_fq_near_d

            score_fq_near_d_mp = ddl_func(mp_fq_near_d, near_single_side_range,
                                          gama=fq_near_mean_proximity_gama) if near_single_side_range != 0 else 1
            score_fq_near_d_co = ddl_func(co_fq_near_d, near_single_side_range,
                                          gama=fq_near_closest_onset_gama) if near_single_side_range != 0 else 1
            score_fq_near_d_td = ddl_func(td_fq_near_d, near_single_side_range,
                                          gama=fq_near_total_duration_gama) if near_single_side_range != 0 else 1

            score_fq_near_d = score_fq_near_d_mp * \
                              score_fq_near_d_co * \
                              score_fq_near_d_td

            fq_near_d_score_list[area_id_next] = score_fq_near_d

    # distant ia method1
    score_fq_dis_list = [[] for _ in range(gt_num + 1)]
    if distant_method == "whole":

        td_fq_dis_list = np.arange(gt_num + 1, dtype=np.float64)
        mp_fq_dis_list = np.arange(gt_num + 1, dtype=np.float64)
        co_fq_dis_list = np.arange(gt_num + 1, dtype=np.float64)

        # 5th circle
        for i, basic_interval_list in enumerate(fq_dis_prediction_group_list):
            td_fq_dis = 0
            mp_fq_dis_sum = 0
            co_fq_dis = 0
            fq_dis_num = len(basic_interval_list)

            fq_dis_section_start = fq_dis_section_list[i][0]
            fq_dis_section_end = fq_dis_section_list[i][1]

            for interval_idx, basic_interval in enumerate(basic_interval_list):
                basic_interval_mid = (basic_interval[1] + basic_interval[0]) / 2
                if distant_direction == "both":
                    td_fq_dis += basic_interval[1] - basic_interval[0]
                    if i == 0:
                        base_position = fq_dis_section_list[i][1]
                        mp_fq_dis_sum += abs(basic_interval_mid - base_position)
                    elif i == gt_num:
                        base_position = fq_dis_section_list[i][0]
                        mp_fq_dis_sum += abs(basic_interval_mid - base_position)
                    else:
                        mp_fq_dis_sum += min(abs(basic_interval_mid - fq_dis_section_list[i][0]), \
                                             abs(basic_interval_mid - fq_dis_section_list[i][1]))

                    if i == 0:
                        if interval_idx == fq_dis_num - 1:
                            co_fq_dis = abs(fq_dis_section_end - basic_interval[1])
                    elif i == gt_num:
                        if interval_idx == 0:
                            co_fq_dis = abs(basic_interval[0] - fq_dis_section_start)
                    else:
                        if interval_idx == 0:
                            co_fq_dis = min(co_fq_dis, abs(basic_interval[0] - fq_dis_section_start))
                        if interval_idx == fq_dis_num - 1:
                            co_fq_dis = min(co_fq_dis, abs(fq_dis_section_end - basic_interval[1]))
                else:  # delay
                    td_fq_dis += basic_interval[1] - basic_interval[0]
                    mp_fq_dis_sum += abs(basic_interval_mid - fq_dis_section_list[i][0])
                    if interval_idx == 0:
                        co_fq_dis = abs(basic_interval[0] - fq_dis_section_start)

            td_fq_dis_list[i] = td_fq_dis

            d_ia_dis_mean = mp_fq_dis_sum / fq_dis_num if fq_dis_num != 0 else 0
            mp_fq_dis_list[i] = d_ia_dis_mean

            co_fq_dis_list[i] = co_fq_dis

            score_fq_dis_mp = ddl_func(d_ia_dis_mean, max_fq_distant_len,
                                       gama=fq_distant_mean_proximity_gama) if max_fq_distant_len != 0 else 1
            score_fq_dis_e_co = ddl_func(co_fq_dis, max_fq_distant_len,
                                         gama=fq_distant_closest_onset_gama) if max_fq_distant_len != 0 else 1
            score_ia_dis_td = ddl_func(td_fq_dis, max_fq_distant_len,
                                       gama=fq_distant_total_duration_gama) if max_fq_distant_len != 0 else 1

            dl_score_ia_distant = score_fq_dis_mp * \
                                  score_fq_dis_e_co * \
                                  score_ia_dis_td

            score_fq_dis_list[i] = dl_score_ia_distant
    else:
        td_fq_dis_e_list = np.arange(gt_num + 1, dtype=np.float64)
        mp_fq_dis_e_list = np.arange(gt_num + 1, dtype=np.float64)
        co_fq_dis_e_list = np.arange(gt_num + 1, dtype=np.float64)
        score_fq_dis_e_list = np.arange(gt_num + 1, dtype=np.float64)

        td_fq_dis_d_list = np.arange(gt_num + 1, dtype=np.float64)
        mp_fq_dis_d_list = np.arange(gt_num + 1, dtype=np.float64)
        co_fq_dis_d_list = np.arange(gt_num + 1, dtype=np.float64)
        score_fq_dis_d_list = np.arange(gt_num + 1, dtype=np.float64)

        for i, area in enumerate(tq_section_list):
            area_id_next = i + 1

            fq_dis_e_area = fq_dis_e_section_list[i]
            fq_dis_e_len = fq_dis_e_area[1] - fq_dis_e_area[0]
            fq_dis_e_end = fq_dis_e_area[1]

            fq_dis_d_area = fq_dis_d_section_list[area_id_next]
            fq_dis_d_len = fq_dis_d_area[1] - fq_dis_d_area[0]
            fq_dis_d_start = fq_dis_d_area[0]

            fq_dis_e_pred_group = fq_dis_e_prediction_group_list[i]
            fq_dis_d_next_pred_group = fq_dis_d_prediction_group_list[area_id_next]

            td_fq_dis_e_sum = 0
            p_fq_dis_e_sum = 0
            co_fq_dis_e = 0
            fq_dis_e_num = len(fq_dis_e_pred_group)

            for interval_idx, basic_interval in enumerate(fq_dis_e_pred_group):
                td_fq_dis_e_sum += basic_interval[1] - basic_interval[0]
                p_fq_dis_e_sum += abs(fq_dis_e_end - (basic_interval[1] + basic_interval[0]) / 2)
                if interval_idx == fq_dis_e_num - 1:
                    co_fq_dis_e += abs(fq_dis_e_end - basic_interval[1])

            td_fq_dis_e_list[i] = td_fq_dis_e_sum

            mp_fq_dis_e = p_fq_dis_e_sum / fq_dis_e_num if fq_dis_e_num != 0 else 0
            mp_fq_dis_e_list[i] = mp_fq_dis_e

            co_fq_dis_e_list[i] = co_fq_dis_e

            score_fq_dis_e_mp = ddl_func(mp_fq_dis_e, max_fq_distant_len,
                                         gama=fq_distant_mean_proximity_gama) if max_fq_distant_len != 0 else 1
            score_fq_dis_e_co = ddl_func(co_fq_dis_e, max_fq_distant_len,
                                         gama=fq_distant_closest_onset_gama) if max_fq_distant_len != 0 else 1
            score_fq_dis_e_td = ddl_func(td_fq_dis_e_sum, fq_dis_e_len,
                                         gama=fq_distant_total_duration_gama) if fq_dis_e_len != 0 else 1

            score_fq_dis_e = score_fq_dis_e_mp * \
                             score_fq_dis_e_co * \
                             score_fq_dis_e_td

            score_fq_dis_e_list[i] = score_fq_dis_e

            td_fq_dis_d_sum = 0
            p_fq_dis_d_sum = 0
            co_fq_dis_d = 0
            fq_dis_d_num = len(fq_dis_d_next_pred_group)

            for interval_idx, basic_interval in enumerate(fq_dis_d_next_pred_group):
                td_fq_dis_d_sum += basic_interval[1] - basic_interval[0]
                p_fq_dis_d_sum += abs((basic_interval[1] + basic_interval[0]) / 2 - fq_dis_d_start)
                if interval_idx == 0:
                    co_fq_dis_d = abs(fq_dis_d_start - basic_interval[0])

            td_fq_dis_d_list[area_id_next] = td_fq_dis_d_sum

            mp_fq_dis_d = p_fq_dis_d_sum / fq_dis_d_num if fq_dis_d_num != 0 else 0
            mp_fq_dis_d_list[area_id_next] = mp_fq_dis_d

            co_fq_dis_d_list[area_id_next] = co_fq_dis_d

            score_fq_dis_d_mp = ddl_func(mp_fq_dis_d, max_fq_distant_len,
                                         gama=fq_distant_mean_proximity_gama) if max_fq_distant_len != 0 else 1
            score_fq_dis_d_co = ddl_func(co_fq_dis_d, max_fq_distant_len,
                                         gama=fq_distant_closest_onset_gama) if max_fq_distant_len != 0 else 1
            score_fq_dis_d_td = ddl_func(td_fq_dis_d_sum, fq_dis_d_len,
                                         gama=fq_distant_total_duration_gama) if fq_dis_d_len != 0 else 1

            dl_score_ia_distant_after = score_fq_dis_d_mp * \
                                        score_fq_dis_d_co * \
                                        score_fq_dis_d_td

            score_fq_dis_d_list[area_id_next] = dl_score_ia_distant_after

    # adjust
    # adjust fq_near

    adjust_score_fq_near_e_list = deepcopy(fq_near_e_score_list)
    adjust_score_fq_near_d_list = deepcopy(fq_near_d_score_list)
    adjust_ratio_fq_near_e_list = [[] for _ in range(gt_num + 1)]

    for i in range(gt_num + 1):
        area_id_front = i - 1
        area_id_next = i + 1

        fq_near_ratio_delay = weight_fq_near_delay
        fq_near_ratio_early = weight_fq_near_early

        # adjust fq_near weight
        # fq_near_early:0-gt_num-1
        # fq_near_delay:1-gt_num
        if weight_fq_near_early > 0 and weight_fq_near_delay > 0:
            if i == 0:
                fq_near_ratio_delay = 0
                if fq_near_e_section_list[i][0] >= fq_near_e_section_list[i][1] and \
                        fq_near_d_section_list[area_id_next][0] < fq_near_d_section_list[area_id_next][1]:
                    fq_near_ratio_early = 0
            elif i == gt_num:
                fq_near_ratio_early = 0
                if fq_near_d_section_list[i][0] >= fq_near_d_section_list[i][1] and \
                        fq_near_e_section_list[area_id_front][0] < fq_near_e_section_list[area_id_front][1]:
                    fq_near_ratio_delay = 0
            else:
                if i == 1:
                    if fq_near_d_section_list[i][0] < fq_near_d_section_list[i][1] and \
                            fq_near_e_section_list[area_id_front][0] >= fq_near_e_section_list[area_id_front][1]:
                        fq_near_ratio_delay = 1
                if i == gt_num - 1:
                    if fq_near_e_section_list[i][0] < fq_near_e_section_list[i][1] and \
                            fq_near_d_section_list[area_id_next][0] >= fq_near_d_section_list[area_id_next][
                        1]:
                        fq_near_ratio_early = 1

        adjust_ratio_fq_near_e_list[i] = [fq_near_ratio_delay, fq_near_ratio_early]

    # adjust fq near score
    for i in range(gt_num):
        area_id_next = i + 1
        # adjust early
        if weight_fq_near_early > 0:
            if fq_near_e_prediction_group_list[i] == [] \
                    and tq_prediction_group_list[i] == [] and fq_near_d_prediction_group_list[area_id_next] == []:
                adjust_score_fq_near_e_list[i] = 0
        else:
            adjust_score_fq_near_e_list[i] = 0

        # adjust delay
        if weight_fq_near_delay > 0:
            if fq_near_d_prediction_group_list[area_id_next] == [] \
                    and tq_prediction_group_list[i] == [] and fq_near_e_prediction_group_list[i] == []:
                adjust_score_fq_near_d_list[area_id_next] = 0
        else:
            adjust_score_fq_near_d_list[area_id_next] = 0

    if distant_method == "whole":
        adjust_score_fq_dis_list = deepcopy(score_fq_dis_list)
        adjust_score_contribution_fq_dis_list = [[] for _ in range(gt_num + 1)]
        adjust_ratio_fq_dis_list = [[] for _ in range(gt_num + 1)]
    else:
        adjust_fq_dis_e_list = deepcopy(score_fq_dis_e_list)  # distant early
        adjust_adjust_fq_dis_d_list = deepcopy(score_fq_dis_d_list)  # distant delay
        adjust_score_fq_dis_e_d_list = [[] for _ in range(gt_num + 1)]

    # adjust fq distant ratio and score
    if distant_method == "whole":
        for i in range(gt_num + 1):
            area_id_front = i - 1
            area_id_next = i + 1

            fq_dis_ratio_delay = 1 / 2
            fq_dis_ratio_early = 1 / 2

            if i == 0:
                # default
                fq_dis_ratio_delay = 0
                # adjust score
                if precision_prediction_group_list[i] == [] and fq_dis_prediction_group_list[i] == []:
                    adjust_score_fq_dis_list[i] = 0

                # adjust rate
                if fq_dis_section_list[i][0] >= fq_dis_section_list[i][1] and fq_dis_section_list[area_id_next][0] < \
                        fq_dis_section_list[area_id_next][1]:
                    fq_dis_ratio_early = 0

            elif i == gt_num:
                # default
                fq_dis_ratio_early = 0
                # adjust score
                if precision_prediction_group_list[area_id_front] == [] and fq_dis_prediction_group_list[i] == []:
                    adjust_score_fq_dis_list[i] = 0
                # adjust rate
                if fq_dis_section_list[i][0] >= fq_dis_section_list[i][1] and fq_dis_section_list[area_id_front][0] < \
                        fq_dis_section_list[area_id_front][1]:
                    fq_dis_ratio_delay = 0
            else:
                if fq_dis_prediction_group_list[i] == [] and \
                        precision_prediction_group_list[area_id_front] == [] and precision_prediction_group_list[
                    i] == []:
                    adjust_score_fq_dis_list[i] = 0

                if i == 1:
                    if fq_dis_section_list[i][0] < fq_dis_section_list[i][1] and fq_dis_section_list[area_id_front][
                        0] >= fq_dis_section_list[area_id_front][1]:
                        fq_dis_ratio_delay = 1
                if i == gt_num - 1:
                    if fq_dis_section_list[i][0] < fq_dis_section_list[i][1] and fq_dis_section_list[area_id_next][0] >= \
                            fq_dis_section_list[area_id_next][1]:
                        fq_dis_ratio_early = 1

            adjust_ratio_fq_dis_list[i] = [fq_dis_ratio_delay, fq_dis_ratio_early]
            adjust_score_contribution_fq_dis_list[i] = [adjust_score_fq_dis_list[i] * fq_dis_ratio_delay,
                                                        adjust_score_fq_dis_list[i] * fq_dis_ratio_early]
    else:
        # adjust ratio
        for i in range(gt_num + 1):
            area_id_front = i - 1
            area_id_next = i + 1

            fq_dis_ratio_delay = 1 / 2
            fq_dis_ratio_early = 1 / 2

            if i == 0:
                # default
                fq_dis_ratio_delay = 0
                if fq_dis_e_section_list[i][0] >= fq_dis_e_section_list[i][1] and fq_dis_d_section_list[area_id_next][
                    0] < fq_dis_d_section_list[area_id_next][1]:
                    fq_dis_ratio_early = 0
            elif i == gt_num:
                fq_dis_ratio_early = 0
                if fq_dis_d_section_list[i][0] >= fq_dis_d_section_list[i][1] and fq_dis_e_section_list[area_id_front][
                    0] < fq_dis_e_section_list[area_id_front][1]:
                    fq_dis_ratio_delay = 0
            else:
                if i == 1:
                    if fq_dis_d_section_list[i][0] < fq_dis_d_section_list[i][1] and \
                            fq_dis_e_section_list[area_id_front][0] >= fq_dis_e_section_list[area_id_front][1]:
                        fq_dis_ratio_delay = 1
                if i == gt_num - 1:
                    if fq_dis_e_section_list[i][0] < fq_dis_e_section_list[i][1] and \
                            fq_dis_d_section_list[area_id_next][0] >= fq_dis_d_section_list[area_id_next][1]:
                        fq_dis_ratio_early = 1

            adjust_score_fq_dis_e_d_list[i] = [fq_dis_ratio_delay, fq_dis_ratio_early]

        # adjust score
        for i in range(gt_num):
            area_id_next = i + 1
            # adjust early
            if fq_dis_e_prediction_group_list[i] == [] \
                    and precision_prediction_group_list[i] == [] and fq_dis_d_prediction_group_list[area_id_next] == []:
                adjust_fq_dis_e_list[i] = 0

            # adjust delay
            if fq_dis_d_prediction_group_list[area_id_next] == [] \
                    and precision_prediction_group_list[i] == [] and fq_dis_e_prediction_group_list[i] == []:
                adjust_adjust_fq_dis_d_list[area_id_next] = 0

    #  weight add
    # cal recall,precision

    local_recall_sum = 0
    local_precision_sum = 0
    local_f1_sum = 0
    local_fq_near_sum = 0
    local_fq_dis_sum = 0

    gt_detected_num = 0

    # 7th circle
    for i in range(gt_num):
        area_id_next = i + 1

        tq_section_end = tq_section_list[i][1]

        # judge empty before and after
        fq_near_e_area = fq_near_e_section_list[i]
        if fq_near_e_area != []:
            fq_near_e_end = fq_near_e_section_list[i][1]

        fq_near_d_area = fq_near_d_section_list[area_id_next]
        if fq_near_d_area != []:
            fq_near_d_start = fq_near_d_section_list[area_id_next][0]

        fq_near_e_pred_group = fq_near_e_prediction_group_list[i]
        precision_tq_pred_group = tq_prediction_group_list[i]
        fq_near_d_pred_group = fq_near_d_prediction_group_list[area_id_next]

        coefficient_p = func_parameter_list_precision_tq[i]["coefficient"]
        beta_p = func_parameter_list_precision_tq[i]["beta"]
        w_gt_p = func_parameter_list_precision_tq[i]["w_tq"]

        coefficient_r = func_parameter_list_recall[i]["coefficient"]
        beta_r = func_parameter_list_recall[i]["beta"]
        w_gt_r = func_parameter_list_recall[i]["w_tq"]

        pred_group_integral_precision_tq = 0
        pred_group_integral_precision_early = 0
        pred_group_integral_precision_delay = 0

        pred_group_integral_recall_tq = 0

        if precision_tq_pred_group != []:
            gt_detected_num += 1
        for j, basic_interval in enumerate(precision_tq_pred_group):
            if basic_interval != []:
                cal_integral_basic_interval_gt_precision = cal_integral_tq_section_power_func(basic_interval,
                                                                                              coefficient_p,
                                                                                              beta_p,
                                                                                              w_gt_p,
                                                                                              area_end=tq_section_end)

                cal_integral_basic_interval_gt_recall = cal_integral_tq_section_power_func(basic_interval,
                                                                                           coefficient_r,
                                                                                           beta_r,
                                                                                           w_gt_r,
                                                                                           area_end=tq_section_end)

                pred_group_integral_precision_tq += cal_integral_basic_interval_gt_precision
                pred_group_integral_recall_tq += cal_integral_basic_interval_gt_recall

        tq_section_integral_recall = tq_recall_integral_list[i]

        tq_recall = pred_group_integral_recall_tq / tq_section_integral_recall

        precision_all = pred_group_integral_precision_tq
        precision_valid = pred_group_integral_precision_tq

        coefficient_early = func_parameter_list_precision_fq_near[i]["coefficient_early"]
        beta_early = func_parameter_list_precision_fq_near[i]["beta_early"]

        coefficient_delay = coefficient_early
        beta_delay = beta_early

        for interval_id, basic_interval in enumerate(fq_near_e_pred_group):
            cal_integral_gt_before = cal_integral_fq_near_e_section_power_func(basic_interval,
                                                                               coefficient_early,
                                                                               beta_early,
                                                                               parameter_near_single_side_range=near_single_side_range,
                                                                               area_end=fq_near_e_end)
            pred_group_integral_precision_early += cal_integral_gt_before

        for interval_id, basic_interval in enumerate(fq_near_d_pred_group):
            cal_integral_gt_after = cal_integral_fq_near_d_section_power_func(basic_interval,
                                                                              coefficient_delay,
                                                                              beta_delay,
                                                                              parameter_near_single_side_range=near_single_side_range,
                                                                              area_start=fq_near_d_start)
            pred_group_integral_precision_delay += cal_integral_gt_after

        precision_all += pred_group_integral_precision_early
        precision_all += pred_group_integral_precision_delay

        tq_precision = precision_valid / precision_all if precision_all != 0 else 0

        tq_f1 = compute_f1_score(tq_precision, tq_recall)

        adjust_score_fq_near = adjust_score_fq_near_e_list[i] * adjust_ratio_fq_near_e_list[i][1] + \
                               adjust_score_fq_near_d_list[area_id_next] * adjust_ratio_fq_near_e_list[area_id_next][0]

        if distant_method == "whole":
            if distant_direction == "both":
                adjust_score_fq_dis = adjust_score_contribution_fq_dis_list[i][1] + \
                                      adjust_score_contribution_fq_dis_list[area_id_next][0]
            else:  # delay
                adjust_score_fq_dis = adjust_score_fq_dis_list[area_id_next]
        else:
            adjust_score_fq_dis = adjust_fq_dis_e_list[i] * adjust_score_fq_dis_e_d_list[i][1] + \
                                  adjust_adjust_fq_dis_d_list[area_id_next] * \
                                  adjust_score_fq_dis_e_d_list[area_id_next][0]

        local_recall_sum += tq_recall
        local_precision_sum += tq_precision
        local_f1_sum += tq_f1
        local_fq_near_sum += adjust_score_fq_near
        local_fq_dis_sum += adjust_score_fq_dis

    gt_detected_rate = gt_detected_num / gt_num
    if use_detection_rate:
        mean_recall = local_recall_sum / gt_num * gt_detected_rate
        mean_precision = local_precision_sum / gt_num * gt_detected_rate
    else:
        mean_recall = local_recall_sum / gt_num
        mean_precision = local_precision_sum / gt_num

    mean_pr_f1 = compute_f1_score(mean_precision, mean_recall)

    mean_fq_near = local_fq_near_sum / gt_num
    mean_fq_dis = local_fq_dis_sum / gt_num

    w_tq = parameter_dict["w_tq"]
    w_fq_near = parameter_dict["w_fq_near"]

    dqe_f1 = cal_dqe_row(w_tq, w_fq_near, mean_fq_dis,
                         mean_pr_f1, mean_fq_near)

    if cal_components:
        dqe_f1_w_tq = cal_dqe_row(1, 0, mean_fq_dis, mean_pr_f1, mean_fq_near)
        dqe_f1_w_fq_near = cal_dqe_row(0, 1, mean_fq_dis, mean_pr_f1, mean_fq_near)
        dqe_f1_w_fq_dis = cal_dqe_row(0, 0, mean_fq_dis, mean_pr_f1, mean_fq_near)

    if cal_components:
        return dqe_f1, dqe_f1_w_tq, dqe_f1_w_fq_near, dqe_f1_w_fq_dis
    else:
        return dqe_f1


def DQE_F1(y_true, binary_predicted, parameter_dict=parameter_dict, near_single_side_range=125, cal_components=False):
    """
    Evaluate binary anomaly-detection predictions on time-series data and compute the final DQE-F1 score.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels for the time series.
    binary_predicted : np.ndarray
        Binary detections of an algorithm by applying a threshold.
    ts_len : int
        Total length of the time series.
    gt_num : int
        Number of ground-truth anomalous events.
    parameter_dict : dict
        Application-specific configuration parameters.
    near_single_side_range : int
        Half-width of the temporal window used to define near-FQ regions;
        detections within this window are treated as relevant near-misses.
    cal_components : bool
        If True, also return the three DQE-F1 sub-scores (TQ, near-FQ, distant-FQ);
        otherwise return only the aggregated DQE-F1.

    Returns
    -------
    dqe_f1 : float
        Final DQE-F1 score for the given threshold.
    dqe_f1_w_tq : float, optional
        TQ-weighted DQE-F1 sub-score (returned only when cal_components=True).
    dqe_f1_w_fq_near : float, optional
        Near-FQ DQE-F1 sub-score (returned only when cal_components=True).
    dqe_f1_w_fq_dis : float, optional
        Distant-FQ DQE-F1 sub-score (returned only when cal_components=True).
    """

    ts_len = len(y_true)
    gt_interval_ranges = convert_vector_to_events_dqe(y_true)
    gt_num = len(gt_interval_ranges)

    pred_interval_ranges = convert_vector_to_events_dqe(binary_predicted)
    if cal_components:
        dqe_f1, dqe_f1_w_tq, dqe_f1_w_fq_near, dqe_f1_w_fq_dis = DQE_F1_section(gt_interval_ranges,
                                                                                pred_interval_ranges,
                                                                                ts_len,
                                                                                gt_num,
                                                                                parameter_dict,
                                                                                near_single_side_range=near_single_side_range,
                                                                                cal_components=cal_components)
    else:
        dqe_f1 = DQE_F1_section(gt_interval_ranges,
                                pred_interval_ranges,
                                ts_len,
                                gt_num,
                                parameter_dict,
                                near_single_side_range=near_single_side_range,
                                cal_components=cal_components)

    if cal_components:
        return dqe_f1, dqe_f1_w_tq, dqe_f1_w_fq_near, dqe_f1_w_fq_dis
    else:
        return dqe_f1


def triangle_weights_add(x: np.ndarray, values):
    """
    For any given 1-D array x (length arbitrary, no need to be evenly spaced),
    return a weight vector w of the same length satisfying:

    1. Symmetric around 0.5.
    2. Linearly increases from 0 to the peak over [0, 0.5], and linearly
       decreases from the peak to 0 over [0.5, 1].
    3. The weights sum to 1.

    Parameters
    ----------
    x : np.ndarray
        Real-valued vector whose elements must lie in [0, 1]
        (no bounds checking is performed).
    values : np.ndarray
        Array of the same shape as x; the quantities to be weighted.

    Returns
    -------
    weighted_sum : scalar
        Dot product of the normalized weights and `values`.
    weights : np.ndarray
        Normalized weight vector, same shape as `x`.
    """
    # 1. Raw triangular weights: base width = 1, height = 2 ⇒ area = 1
    height = 2.0
    weights = np.where(x <= 0.5,
                       height * (x / 0.5),  # left side
                       height * ((1 - x) / 0.5))  # right side
    # 2. Normalize so that the weights sum to 1
    weights /= weights.sum()

    weight_sum = weights @ values

    return weight_sum, weights


def DQE(y_true, y_score, parameter_dict=parameter_dict, near_single_side_range=125, max_fq_distant_len=None,
        thresh_num=101, cal_components=False):
    """
    Evaluate continuous anomaly scores in a threshold-free manner.

    The function performs all prerequisite steps for DQE:
    1. Section partitioning
    2. Integral of the entire TQ section (used for Recall-TQ)
    3. Triangle Weighted sum over all thresholds to produce the final threshold-free DQE score.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels for the time series.
    y_score : np.ndarray
        Algorithm output scores normalized to [0, 1], indicating anomaly likelihood.
    parameter_dict : dict
        Application-specific configuration parameters.
    near_single_side_range : int
        Half-width of the temporal window used to define near-FQ regions;
        detections within this window are treated as relevant near-misses.
    max_fq_distant_len : int
        Maximum length among all distant-FQ sections.
    thresh_num : int
        Number of thresholds to evaluate. Larger values yield finer-grained results
        at the cost of higher runtime.
    cal_components : bool
        If True, also return the three DQE sub-scores (TQ, near-FQ, distant-FQ);
        otherwise return only the aggregated DQE.

    Returns
    -------
    dqe : float
        Final threshold-free DQE score.
    dqe_w_tq : float, optional
        TQ-weighted DQE sub-score (returned only when cal_components=True).
    dqe_w_fq_near : float, optional
        Near-FQ DQE sub-score (returned only when cal_components=True).
    dqe_w_fq_dis : float, optional
        Distant-FQ DQE sub-score (returned only when cal_components=True).
    """

    ts_len = len(y_true)

    thresholds = np.linspace(1, 0, thresh_num)

    # array -> interval_ranges
    gt_interval_ranges = convert_vector_to_events_dqe(y_true)

    gt_num = len(gt_interval_ranges)

    weight_fq_near_early = parameter_dict["w_fq_near_early"]
    weight_fq_near_delay = parameter_dict["w_fq_near_delay"]

    distant_method = parameter_dict["distant_method"]

    beta = parameter_dict["beta"]

    weight_sum_method = parameter_dict["weight_sum_method"]

    if max_fq_distant_len is None:
        # o(m)
        max_fq_distant_len = find_max_fq_distant_length_in_single_ts(gt_interval_ranges, ts_len, gt_num,
                                                                     parameter_dict=parameter_dict,
                                                                     near_single_side_range=near_single_side_range)

    dqe_list = []
    if cal_components:
        dqe_list_w_tq = []
        dqe_list_w_fq_near = []
        dqe_list_w_fq_distant = []

    for i, threshold in enumerate(thresholds):
        if threshold <= 0:
            continue
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
                    if distant_method == "whole":
                        # fq_distant_i
                        fq_dis_section_list[i] = [0, fq_near_e_section_i_start]

                        # next section
                        # fq_distant_i_next
                        fq_dis_section_list[i + 1] = [fq_near_d_section_i_next_end, ts_len]
                    else:
                        # fq_distant_early_i
                        fq_dis_e_section_list[i] = [0, fq_near_e_section_i_start]

                        # next section
                        # fq_distant_delay_i_next
                        fq_dis_d_section_list[i + 1] = [fq_near_d_section_i_next_end, ts_len]

                    # merge section
                    if weight_fq_near_early <= 0:  # merge fq_near_early to fq_distant
                        # fq_near_early_i = empty
                        fq_near_e_section_list[i] = [tq_section_i_start, tq_section_i_start]
                        if distant_method == "whole":
                            # fq_distant_i
                            fq_dis_section_list[i] = [0, tq_section_i_start]
                        else:
                            # fq_distant_early_i
                            fq_dis_e_section_list[i] = [0, tq_section_i_start]
                    if weight_fq_near_delay <= 0:  # merge fq_near_delay to fq_distant
                        # fq_near_delay_i_next = empty
                        fq_near_d_section_list[i + 1] = [tq_section_i_end, tq_section_i_end]

                        if distant_method == "whole":
                            # next section
                            fq_dis_section_list[i + 1] = [tq_section_i_end, ts_len]
                        else:
                            # next section
                            # fq_distant_delay_i_next
                            fq_dis_d_section_list[i + 1] = [tq_section_i_end, ts_len]
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
                    if distant_method == "whole":
                        # fq_distant_i
                        fq_dis_section_list[i] = [0, fq_near_e_section_i_start]
                    else:
                        # fq_distant_early_i
                        fq_dis_e_section_list[i] = [0, fq_near_e_section_i_start]

                    # merge section
                    if weight_fq_near_early <= 0:
                        fq_near_e_section_list[i] = [tq_section_i_start, tq_section_i_start]
                        if distant_method == "whole":
                            fq_dis_section_list[i] = [0, tq_section_i_start]
                        else:
                            fq_dis_e_section_list[i] = [0, tq_section_i_start]
                    if weight_fq_near_delay <= 0:
                        fq_near_d_section_list[i + 1] = [tq_section_i_end, tq_section_i_end]
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
                if distant_method == "whole":
                    fq_dis_section_list[i] = [fq_near_d_i_end, fq_near_e_section_i_start]

                    # next section
                    fq_dis_section_list[i + 1] = [fq_near_d_section_i_next_end, ts_len]
                else:
                    fq_dis_section_i_mid = (fq_near_d_i_end + fq_near_e_section_i_start) / 2
                    # fq_dis_d_section_list[i] = [fq_near_d_i_end, fq_dis_section_i_mid]
                    # fq_dis_e_section_list[i] = [fq_dis_section_i_mid, fq_near_e_section_i_start]
                    fq_dis_d_section_list[i] = [fq_dis_section_i_mid, fq_near_e_section_i_start]
                    fq_dis_e_section_list[i] = [fq_near_d_i_end, fq_dis_section_i_mid]

                    # next section
                    fq_dis_d_section_list[i + 1] = [fq_near_d_section_i_next_end, ts_len]

                # merge section
                if weight_fq_near_early <= 0:
                    fq_near_e_section_list[i] = [tq_section_i_start, tq_section_i_start]
                    if distant_method == "whole":
                        fq_dis_section_list[i] = [fq_near_d_i_end, tq_section_i_start]
                    else:
                        fq_dis_section_i_mid = (fq_near_d_i_end + tq_section_i_start) / 2
                        # fq_dis_d_section_list[i] = [fq_near_d_i_end, fq_dis_section_i_mid]
                        # fq_dis_e_section_list[i] = [fq_dis_section_i_mid, tq_section_i_start]
                        fq_dis_d_section_list[i] = [fq_dis_section_i_mid, tq_section_i_start]
                        fq_dis_e_section_list[i] = [fq_near_d_i_end, fq_dis_section_i_mid]
                if weight_fq_near_delay <= 0:
                    fq_near_d_section_list[i + 1] = [tq_section_i_end, tq_section_i_end]
                    if distant_method == "whole":
                        fq_dis_section_list[i] = [tq_section_i_last_end, fq_near_e_section_i_start]
                        # next section
                        fq_dis_section_list[i + 1] = [tq_section_i_end, ts_len]
                    else:
                        fq_dis_section_i_mid = (tq_section_i_last_end + fq_near_e_section_i_start) / 2
                        # fq_dis_d_section_list[i] = [tq_section_i_last_end, fq_dis_section_i_mid]
                        # fq_dis_e_section_list[i] = [fq_dis_section_i_mid, fq_near_e_section_i_start]
                        fq_dis_d_section_list[i] = [fq_dis_section_i_mid, fq_near_e_section_i_start]
                        fq_dis_e_section_list[i] = [tq_section_i_last_end, fq_dis_section_i_mid]

                        # next section
                        fq_dis_d_section_list[i + 1] = [tq_section_i_end, ts_len]
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
                if distant_method == "whole":
                    fq_dis_section_list[i] = [fq_near_d_i_end, fq_near_e_section_i_start]
                else:
                    fq_dis_section_i_mid = (fq_near_d_i_end + fq_near_e_section_i_start) / 2
                    # fq_dis_d_section_list[i] = [fq_near_d_i_end, fq_dis_section_i_mid]
                    # fq_dis_e_section_list[i] = [fq_dis_section_i_mid, fq_near_e_section_i_start]
                    fq_dis_d_section_list[i] = [fq_dis_section_i_mid, fq_near_e_section_i_start]
                    fq_dis_e_section_list[i] = [fq_near_d_i_end, fq_dis_section_i_mid]

                # merge section
                if weight_fq_near_early <= 0:
                    fq_near_e_section_list[i] = [tq_section_i_start, tq_section_i_start]
                    if distant_method == "whole":
                        fq_dis_section_list[i] = [fq_near_d_i_end, tq_section_i_start]
                    else:
                        fq_dis_section_i_mid = (fq_near_d_i_end + tq_section_i_start) / 2
                        # fq_dis_d_section_list[i] = [fq_near_d_i_end, fq_dis_section_i_mid]
                        # fq_dis_e_section_list[i] = [fq_dis_section_i_mid, tq_section_i_start]
                        fq_dis_d_section_list[i] = [fq_dis_section_i_mid, tq_section_i_start]
                        fq_dis_e_section_list[i] = [fq_near_d_i_end, fq_dis_section_i_mid]
                if weight_fq_near_delay <= 0:
                    fq_near_d_section_list[i + 1] = [tq_section_i_end, tq_section_i_end]
                    if distant_method == "whole":
                        fq_dis_section_list[i] = [tq_section_i_last_end, fq_near_e_section_i_start]
                    else:
                        fq_dis_section_i_mid = (tq_section_i_last_end + fq_near_e_section_i_start) / 2
                        # fq_dis_d_section_list[i] = [fq_dis_section_list[i][0], fq_dis_section_i_mid]
                        # fq_dis_e_section_list[i] = [fq_dis_section_i_mid, fq_dis_section_list[i][1]]
                        fq_dis_d_section_list[i] = [fq_dis_section_i_mid, fq_dis_section_list[i][1]]
                        fq_dis_e_section_list[i] = [fq_dis_section_list[i][0], fq_dis_section_i_mid]

            # add split line
            if weight_fq_near_early > 0:
                split_line_set.add(fq_near_e_section_i_start)

            split_line_set.add(tq_section_i_start)
            split_line_set.add(tq_section_i_end)

            if weight_fq_near_delay > 0:
                split_line_set.add(fq_near_d_section_i_next_end)

            if distant_method == "split":
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

        tq_recall_integral_list = np.arange(gt_num, dtype=np.float64)

        func_parameter_list_precision_tq = [{} for _ in range(gt_num)]
        func_parameter_list_recall = [{} for _ in range(gt_num)]
        func_parameter_list_precision_fq_near = [{} for _ in range(gt_num)]

        # cal score function,tq_near score
        for i, area in enumerate(tq_section_list):
            tq_section_len = area[1] - area[0]

            w_tq_recall = 1
            a_recall = cal_power_function_parameter(beta, w_tq_recall, 0, tq_section_len)
            w_tq_precision = 1 / 2

            c_precision = cal_power_function_parameter(beta, w_tq_precision, 0, tq_section_len)

            entire_section_integral_tq_recall = cal_integral_tq_section_power_func(area, a_recall, beta, \
                                                                                   parameter_w_tq=w_tq_recall, \
                                                                                   area_end=area[1])

            func_parameter_list_precision_tq[i]["coefficient"] = c_precision
            func_parameter_list_precision_tq[i]["beta"] = beta
            func_parameter_list_precision_tq[i]["w_tq"] = w_tq_precision

            func_parameter_list_recall[i]["coefficient"] = a_recall
            func_parameter_list_recall[i]["beta"] = beta
            func_parameter_list_recall[i]["w_tq"] = w_tq_recall

            beta_early = beta
            coefficient_early = (1 - w_tq_precision) / pow(near_single_side_range, beta)

            func_parameter_list_precision_fq_near[i]["coefficient_early"] = coefficient_early
            func_parameter_list_precision_fq_near[i]["beta_early"] = beta_early

            tq_recall_integral_list[i] = entire_section_integral_tq_recall

        integral_tq_section_res = {
            "tq_recall_integral_list": tq_recall_integral_list,
            "func_parameter_list_precision_tq": func_parameter_list_precision_tq,
            "func_parameter_list_recall": func_parameter_list_recall,
            "func_parameter_list_precision_fq_near": func_parameter_list_precision_fq_near,
        }

        if cal_components:
            dqe_f1, dqe_f1_w_tq, dqe_f1_w_fq_near, dqe_f1_w_fq_dis = DQE_F1_section(gt_interval_ranges,
                                                                                    pred_interval_ranges,
                                                                                    ts_len,
                                                                                    gt_num,
                                                                                    parameter_dict,
                                                                                    near_single_side_range=near_single_side_range,
                                                                                    max_fq_distant_len=max_fq_distant_len,
                                                                                    cal_components=True,
                                                                                    partition_res=partition_res,
                                                                                    integral_tq_section_res=integral_tq_section_res)
        else:
            dqe_f1 = DQE_F1_section(gt_interval_ranges,
                                    pred_interval_ranges,
                                    ts_len,
                                    gt_num,
                                    parameter_dict,
                                    near_single_side_range=near_single_side_range,
                                    max_fq_distant_len=max_fq_distant_len,
                                    partition_res=partition_res,
                                    integral_tq_section_res=integral_tq_section_res)
        dqe_list.append(dqe_f1)

        if cal_components:
            dqe_list_w_tq.append(dqe_f1_w_tq)
            dqe_list_w_fq_near.append(dqe_f1_w_fq_near)
            dqe_list_w_fq_distant.append(dqe_f1_w_fq_dis)

    # weight sum
    if weight_sum_method == "equal":
        dqe = np.mean(dqe_list)
        if cal_components:
            dqe_w_tq = np.mean(dqe_list_w_tq)
            dqe_w_fq_near = np.mean(dqe_list_w_fq_near)
            dqe_w_fq_distant = np.mean(dqe_list_w_fq_distant)
    else:
        dqe, triangle_weights = triangle_weights_add(thresholds[:-1], np.array(dqe_list))
        if cal_components:
            dqe_w_tq, _ = triangle_weights_add(thresholds[:-1], np.array(dqe_list_w_tq))
            dqe_w_fq_near, _ = triangle_weights_add(thresholds[:-1], np.array(dqe_list_w_fq_near))
            dqe_w_fq_distant, _ = triangle_weights_add(thresholds[:-1],
                                                       np.array(dqe_list_w_fq_distant))

    if cal_components:
        return dqe, dqe_w_tq, dqe_w_fq_near, dqe_w_fq_distant
    else:
        return dqe


def cal_dqe_row(parameter_w_tq, parameter_w_fq_near, row_mean_distant_fp_value, row_mean_f1_value,
                row_mean_near_fp_value):
    # Calculating weighted sum DQE-F1.
    local_dqe_value = parameter_w_tq * row_mean_f1_value + \
                      parameter_w_fq_near * row_mean_near_fp_value + \
                      (1 - parameter_w_tq - parameter_w_fq_near) * row_mean_distant_fp_value
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


def find_max_fq_distant_length_in_single_ts(interval_ranges, ts_len, gt_num, parameter_dict=parameter_dict,
                                            near_single_side_range=125):
    """
    Determine the maximum length of any distant-FQ section in a single time series.

    Parameters
    ----------
    interval_ranges : list[list[float]]
        Ground-truth anomaly intervals.
    ts_len : int
        Total length of the time series.
    gt_num : int
        Number of ground-truth anomalous events.
    parameter_dict : dict
        Application-specific configuration parameters.
    near_single_side_range : int
        Half-width of the temporal window used to define near-FQ regions;
        detections within this window are treated as relevant near-misses.

    Returns
    -------
    max_fq_dis_len : int
        Maximum length among all distant-FQ sections for this time series.
    """

    distant_method = parameter_dict["distant_method"]
    w_fq_near_early = parameter_dict["w_fq_near_early"]
    w_fq_near_delay = parameter_dict["w_fq_near_delay"]

    max_fq_dis_len = 0

    for i, label_interval_range in enumerate(interval_ranges):
        gt_i_start, gt_i_end = label_interval_range
        if i == 0:  # same to distant strategy 1 and 2
            if w_fq_near_early <= 0:  # early is zero
                fq_dis_max_len_temp = max(gt_i_start - 0, 0)
            elif w_fq_near_delay <= 0:  # delay is zero
                fq_dis_max_len_temp = max(gt_i_start - 0 - near_single_side_range, 0)
            else:
                fq_dis_max_len_temp = max(gt_i_start - 0 - near_single_side_range, 0)
            if gt_num == 1:  # same to distant strategy 1 and 2
                if w_fq_near_early <= 0:
                    fq_dis_max_len_temp = max(fq_dis_max_len_temp,
                                              ts_len - gt_i_end - near_single_side_range)
                elif w_fq_near_delay <= 0:
                    fq_dis_max_len_temp = max(fq_dis_max_len_temp, ts_len - gt_i_end)
                else:
                    fq_dis_max_len_temp = max(fq_dis_max_len_temp,
                                              ts_len - gt_i_end - near_single_side_range)  # update ia_distant_length
        elif i == gt_num - 1:
            gt_i_last_start, gt_i_last_end = interval_ranges[i - 1]
            if distant_method == "whole":
                if w_fq_near_early <= 0:
                    fq_dis_max_len_temp = max(gt_i_start - gt_i_last_end - near_single_side_range * 1, 0)
                elif w_fq_near_delay <= 0:
                    fq_dis_max_len_temp = max(gt_i_start - gt_i_last_end - near_single_side_range * 1, 0)
                else:
                    fq_dis_max_len_temp = max(gt_i_start - gt_i_last_end - near_single_side_range * 2, 0)
            else:
                if w_fq_near_early <= 0:
                    fq_dis_max_len_temp = max((gt_i_start - gt_i_last_end - near_single_side_range * 1) / 2, 0)
                elif w_fq_near_delay <= 0:
                    fq_dis_max_len_temp = max((gt_i_start - gt_i_last_end - near_single_side_range * 1) / 2, 0)
                else:
                    fq_dis_max_len_temp = max((gt_i_start - gt_i_last_end - near_single_side_range * 2) / 2, 0)

            # last fq_distant section, same to distant strategy 1 and 2
            if w_fq_near_early <= 0:
                fq_dis_max_len_temp = max(fq_dis_max_len_temp, ts_len - gt_i_end - near_single_side_range)
            elif w_fq_near_delay <= 0:
                fq_dis_max_len_temp = max(fq_dis_max_len_temp, ts_len - gt_i_end)
            else:
                fq_dis_max_len_temp = max(fq_dis_max_len_temp,
                                          ts_len - gt_i_end - near_single_side_range)  # update ia_distant_length
        else:
            gt_i_last_start, gt_i_last_end = interval_ranges[i - 1]
            if distant_method == "whole":
                if w_fq_near_early <= 0:
                    fq_dis_max_len_temp = max(gt_i_start - gt_i_last_end - near_single_side_range * 1, 0)
                elif w_fq_near_delay <= 0:
                    fq_dis_max_len_temp = max(gt_i_start - gt_i_last_end - near_single_side_range * 1, 0)
                else:
                    fq_dis_max_len_temp = max(gt_i_start - gt_i_last_end - near_single_side_range * 2, 0)
            else:
                if w_fq_near_early <= 0:
                    fq_dis_max_len_temp = max((gt_i_start - gt_i_last_end - near_single_side_range * 1) / 2, 0)
                elif w_fq_near_delay <= 0:
                    fq_dis_max_len_temp = max((gt_i_start - gt_i_last_end - near_single_side_range * 1) / 2, 0)
                else:
                    fq_dis_max_len_temp = max((gt_i_start - gt_i_last_end - near_single_side_range * 2) / 2, 0)

        if fq_dis_max_len_temp > max_fq_dis_len:
            max_fq_dis_len = fq_dis_max_len_temp

    return max_fq_dis_len


def find_max_fq_distant_length_in_multi_ts(ts_data_list, parameter_dict=parameter_dict, near_single_side_range=None):
    """
    Determine the maximum length of any distant-FQ section across all time-series
    in one or multiple datasets.

    Parameters
    ----------
    ts_data_list : list[np.ndarray]
        List of ground-truth label arrays (one per time-series).
    parameter_dict : dict
        Application-specific configuration parameters.
    near_single_side_range : int
        Half-width of the temporal window used to define near-FQ regions;
        detections within this window are treated as relevant near-misses.

    Returns
    -------
    max_fq_dis_len : int
        Maximum length among all distant-FQ sections found in the input data.
    """

    if not ts_data_list:
        raise ValueError("list can bot be empty")
    if isinstance(ts_data_list, list):
        ts_data_list = np.array(ts_data_list)

    max_fq_dis_len = 0

    for single_ts_data in ts_data_list:
        if near_single_side_range is None:
            # find period of a time series
            near_single_side_range = find_length_rank(single_ts_data.reshape(-1, 1), rank=1)
        ts_len = len(single_ts_data)
        gt_interval_ranges = convert_vector_to_events_dqe(single_ts_data)
        gt_num = len(gt_interval_ranges)
        max_fq_dis_len = find_max_fq_distant_length_in_single_ts(gt_interval_ranges,
                                                                 ts_len,
                                                                 gt_num,
                                                                 parameter_dict,
                                                                 near_single_side_range,
                                                                 )
    max_fq_dis_len = max(max_fq_dis_len, 0)
    return max_fq_dis_len