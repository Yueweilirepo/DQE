import time

from .basic_metrics import basic_metricor, generate_curve
from metrics.pate.PATE_metric import PATE


def get_metrics(score, labels, slidingWindow=100, pred=None, version='opt', thre=250, print_msg=False, test_time=True,
                exp_name=None, ratio=1.0, near_miss_range_size=None, metric_list=None, average_aggregation=False,
                dynamic_thresh_range=None, cal_components=False, per_anomaly_res=False, thresh_num=100):
    metrics = {}
    metrics_consume_time = {}

    th_100_exp_list = [
        'Standard-F1',
        'AUC-ROC',
        'AUC-PR',

        "PA-K",

        'VUS-ROC',
        'VUS-PR',
        'PATE',

        'R-based-F1',
        'eTaPR_F1',
        'Affiliation-F',
        "DQE",
    ]

    if metric_list != None:
        exp_list = metric_list
    else:
        if exp_name == "AUC-ROC/AUC-PR issue case":
            exp_list = [
                'AUC-ROC',
                'AUC-PR'
            ]
        else:
            exp_list = th_100_exp_list

    time_consume_dict = {}

    '''
    Threshold Independent
    '''
    grader = basic_metricor()

    if "AUC-ROC" in exp_list:
        if test_time:
            time_start = time.time()
        if exp_name == "AUC-ROC/AUC-PR issue case":
            AUC_ROC = grader.metric_ROC(labels, score, plot_flag=True)
        else:
            AUC_ROC = grader.metric_ROC(labels, score)
        if test_time:
            time_end = time.time()
            time_consume = time_end - time_start
            time_consume_dict["AUC-ROC"] = time_consume
            if print_msg:
                print("AUC-ROC time_end - time_start", time_consume)
            metrics_consume_time["AUC-ROC"] = time_consume
        metrics['AUC-ROC'] = AUC_ROC

    if "AUC-PR" in exp_list:
        if test_time:
            time_start = time.time()
        if exp_name == "AUC-ROC/AUC-PR issue case":
            AUC_PR = grader.metric_PR(labels, score, plot_flag=True)
        else:
            AUC_PR = grader.metric_PR(labels, score)
        if test_time:
            time_end = time.time()
            time_consume = time_end - time_start
            time_consume_dict["AUC-PR"] = time_consume
            if print_msg:
                print("AUC-PR time_end - time_start", time_consume)
            metrics_consume_time["AUC-PR"] = time_consume
        metrics['AUC-PR'] = AUC_PR

    if "VUS-PR" in exp_list or "VUS-ROC" in exp_list:
        if near_miss_range_size != None:
            single_side_range_size = near_miss_range_size
        else:
            single_side_range_size = round(slidingWindow * ratio)

        if test_time:
            time_start = time.time()
        _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(labels.astype(int), score, single_side_range_size, version,
                                                           thre)
        if test_time:
            time_end = time.time()
            time_consume = time_end - time_start
            time_consume_dict["VUS-ROC VUS-PR"] = time_consume
            if print_msg:
                print("VUS-ROC VUS-PR time_end - time_start", time_consume)
            metrics_consume_time["VUS-ROC VUS-PR"] = time_consume

        metrics['VUS-PR'] = VUS_PR
        metrics['VUS-ROC'] = VUS_ROC

    if "PATE" in exp_list:
        if near_miss_range_size != None:
            e_buffer = d_buffer = near_miss_range_size
        else:
            slidingWindow = round(slidingWindow * ratio)
            e_buffer = d_buffer = slidingWindow

        if test_time:
            time_start = time.time()

        pate = PATE(labels, score, e_buffer, d_buffer, Big_Data=True, n_jobs=1, include_zero=False,
                    num_desired_thresholds=thre)
        if test_time:
            time_end = time.time()
            time_consume = time_end - time_start
            time_consume_dict["PATE"] = time_consume
            if print_msg:
                print("PATE time_end - time_start", time_consume)
            metrics_consume_time["PATE"] = time_consume

        metrics['PATE'] = pate

    if "DQE" in exp_list:
        if near_miss_range_size != None:
            near_single_side_range = near_miss_range_size
        else:
            slidingWindow = round(slidingWindow * ratio)
            near_single_side_range = slidingWindow

        if test_time:
            time_start = time.time()

        # nm range sensitivity,
        # default nm range, mean res,
        # time efficiency,
        # cal_components = False
        # per_anomaly_res = False

        # case analysis components
        # cal_components = True
        # per_anomaly_res = True

        th_lower = 0.0
        th_upper = 1.0

        if dynamic_thresh_range != None:
            # dynamic thresh range
            cal_components = False
            per_anomaly_res = False
            th_lower = dynamic_thresh_range[0]
            th_upper = dynamic_thresh_range[1]

        dqe_res_ts = grader.metric_DQE(labels, score, preds=pred, near_single_side_range=near_single_side_range,
                                       th_lower=th_lower, th_upper=th_upper, cal_components=cal_components,
                                       per_anomaly_res=per_anomaly_res, thresh_num=thresh_num)
        if test_time:
            time_end = time.time()
            time_consume = time_end - time_start
            time_consume_dict["dqe"] = time_consume
            if print_msg:
                print("dqe time_end - time_start", time_consume)
            metrics_consume_time["dqe"] = time_consume

        metrics['dqe'] = dqe_res_ts['dqe']
        if cal_components:
            metrics['dqe_cap'] = dqe_res_ts['dqe_cap']
            metrics['dqe_near_miss'] = dqe_res_ts['dqe_near_miss']
            metrics['dqe_false_alarm'] = dqe_res_ts['dqe_false_alarm']
        if per_anomaly_res:
            metrics['dqe_per_anomaly_res'] = dqe_res_ts['per_anomaly_res']

    '''
    Threshold Dependent
    if pred is None --> use the oracle threshold
    '''


    if "Standard-F1" in exp_list:
        if test_time:
            time_start = time.time()
        PointF1 = grader.metric_PointF1(labels, score, preds=pred, averaged_aggregation=average_aggregation)
        if test_time:
            time_end = time.time()
            time_consume = time_end - time_start
            time_consume_dict["Standard-F1"] = time_consume
            if print_msg:
                print("Standard-F1 time_end - time_start", time_consume)
            metrics_consume_time["Standard-F1"] = time_consume

        metrics['Standard-F1'] = PointF1

    if "PA-F1" in exp_list:
        if test_time:
            time_start = time.time()
        PointF1PA = grader.metric_PointF1PA(labels, score, preds=pred)
        if test_time:
            time_end = time.time()
            time_consume = time_end - time_start
            time_consume_dict["PA-F1"] = time_consume
            if print_msg:
                print("PA-F1 time_end - time_start", time_consume)
            metrics_consume_time["PA-F1"] = time_consume

        metrics['PA-F1'] = PointF1PA

    if "PA-K" in exp_list:
        if test_time:
            time_start = time.time()
        PointF1PA_K = grader.metric_PointF1PA_K(labels, score, preds=pred, averaged_aggregation=average_aggregation)
        if test_time:
            time_end = time.time()
            time_consume = time_end - time_start
            time_consume_dict["PA-K"] = time_consume
            if print_msg:
                print("PA-K time_end - time_start", time_consume)
            metrics_consume_time["PA-K"] = time_consume

        metrics['PA-K'] = PointF1PA_K

    if "R-based-F1" in exp_list:
        if test_time:
            time_start = time.time()
        RF1 = grader.metric_RF1(labels, score, preds=pred, averaged_aggregation=average_aggregation)
        if test_time:
            time_end = time.time()
            time_consume = time_end - time_start
            time_consume_dict["R-based-F1"] = time_consume
            if print_msg:
                print("R-based-F1 time_end - time_start", time_consume)
            metrics_consume_time["R-based-F1"] = time_consume

        metrics['R-based-F1'] = RF1
    if "Affiliation-F" in exp_list:
        if test_time:
            time_start = time.time()
        Affiliation_F = grader.metric_Affiliation(labels, score, preds=pred, averaged_aggregation=average_aggregation)
        if test_time:
            time_end = time.time()
            time_consume = time_end - time_start
            time_consume_dict["Affiliation-F"] = time_consume
            if print_msg:
                print("Affiliation-F time_end - time_start", time_consume)
            metrics_consume_time["Affiliation-F"] = time_consume

        metrics['Affiliation-F'] = Affiliation_F
    if "eTaPR_F1" in exp_list:
        if test_time:
            time_start = time.time()
        eTaPR_F1 = grader.metric_eTaPR_F1(labels, score, preds=pred, averaged_aggregation=average_aggregation)
        if test_time:
            time_end = time.time()
            time_consume = time_end - time_start
            time_consume_dict["eTaPR_F1"] = time_consume
            if print_msg:
                print("eTaPR_F1 time_end - time_start", time_consume)
            metrics_consume_time["eTaPR_F1"] = time_consume

        metrics['eTaPR_F1'] = eTaPR_F1

    if "PATE_F1" in exp_list:
        if test_time:
            time_start = time.time()
        e_buffer = d_buffer = slidingWindow
        pate_f1 = grader.metric_PATE_F1(labels, score, preds=pred, e_buffer=e_buffer, d_buffer=d_buffer)
        if test_time:
            time_end = time.time()
            time_consume = time_end - time_start
            time_consume_dict["PATE_F1"] = time_consume
            if print_msg:
                print("PATE_F1 time_end - time_start", time_consume)
            metrics_consume_time["PATE_F1"] = time_consume

        metrics['PATE_F1'] = pate_f1

    return metrics, metrics_consume_time

