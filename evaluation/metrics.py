from .basic_metrics import basic_metricor, generate_curve
from metrics.pate.PATE_metric import PATE



def get_metrics(score, labels, slidingWindow=100, pred=None, version='opt', thre=250,print_msg=True,test_time=True, exp_name=None):
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

    if exp_name == "AUC-ROC/AUC-PR issue case":
        exp_list = [
            'AUC-ROC',
            'AUC-PR'
        ]
    else:
        exp_list = th_100_exp_list


    '''
    Threshold Independent
    '''
    grader = basic_metricor()

    if "AUC-ROC" in exp_list:
        if exp_name == "AUC-ROC/AUC-PR issue case":
            AUC_ROC = grader.metric_ROC(labels, score, plot_flag = True)
        else:
            AUC_ROC = grader.metric_ROC(labels, score)
        metrics['AUC-ROC'] = AUC_ROC


    if "AUC-PR" in exp_list:
        if exp_name == "AUC-ROC/AUC-PR issue case":
            AUC_PR = grader.metric_PR(labels, score, plot_flag = True)
        else:
            AUC_PR = grader.metric_PR(labels, score)
        metrics['AUC-PR'] = AUC_PR

    if "VUS-PR"  in exp_list or "VUS-ROC" in exp_list:
        _, _, _, _, _, _,VUS_ROC, VUS_PR = generate_curve(labels.astype(int), score, slidingWindow, version, thre)
        metrics['VUS-PR'] = VUS_PR
        metrics['VUS-ROC'] = VUS_ROC

    if "PATE" in exp_list:
        e_buffer = d_buffer = slidingWindow//2
        pate = PATE(labels, score, e_buffer, d_buffer, Big_Data=True, n_jobs=1, include_zero=False,num_desired_thresholds=thre)
        metrics['PATE'] = pate

    if "DQE" in exp_list:
        dqe_res_ts = grader.metric_DQE(labels, score, preds=pred, near_single_side_range=slidingWindow/2)
        metrics['dqe'] = dqe_res_ts['dqe']
        metrics['dqe_cap'] = dqe_res_ts['dqe_cap']
        metrics['dqe_near_miss'] = dqe_res_ts['dqe_near_miss']
        metrics['dqe_false_alarm'] = dqe_res_ts['dqe_false_alarm']

    '''
    Threshold Dependent
    if pred is None --> use the oracle threshold
    '''

    if "Standard-F1" in exp_list:
        PointF1 = grader.metric_PointF1(labels, score, preds=pred)
        metrics['Standard-F1'] = PointF1

    if "PA-F1" in exp_list:
        PointF1PA = grader.metric_PointF1PA(labels, score, preds=pred)
        metrics['PA-F1'] = PointF1PA

    if "PA-K" in exp_list:
        PointF1PA_K = grader.metric_PointF1PA_K(labels, score, preds=pred)
        metrics['PA-K'] = PointF1PA_K

    if "R-based-F1" in exp_list:
        RF1 = grader.metric_RF1(labels, score, preds=pred)
        metrics['R-based-F1'] = RF1
    if "Affiliation-F" in exp_list:
        Affiliation_F = grader.metric_Affiliation(labels, score, preds=pred)
        metrics['Affiliation-F'] = Affiliation_F
    if "eTaPR_F1" in exp_list:
        eTaPR_F1 = grader.metric_eTaPR_F1(labels, score, preds=pred)
        metrics['eTaPR_F1'] = eTaPR_F1

    return metrics,metrics_consume_time

