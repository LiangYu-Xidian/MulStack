
import math
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score



#用于计算测试结果的指标
def performance(origin_labels, predict_labels, deci_value):

    if len(origin_labels) != len(predict_labels):
        raise ValueError("The number of the original labels must equal to that of the predicted labels.")

    PR = average_precision_score(origin_labels, deci_value, average='weighted', pos_label=1, sample_weight=None)

    TP = 0.0
    TN = 0.0
    FP = 0.0
    FN = 0.0

    for i in range(len(origin_labels)):
        if origin_labels[i] == 1.0 and predict_labels[i] == 1.0:
            TP += 1.0
        elif origin_labels[i] == 1.0 and predict_labels[i] == 0:
            FN += 1.0
        elif origin_labels[i] == 0 and predict_labels[i] == 1.0:
            FP += 1.0
        elif origin_labels[i] == 0 and predict_labels[i] == 0:
            TN += 1.0


    try:
        SN = TP / (TP + FN)
    except ZeroDivisionError:
        SN = 0.0
    try:
        SP = TN / (FP + TN)
    except ZeroDivisionError:
        SP = 0.0
    try:
        ACC = (TP + TN) / (TP + TN + FP + FN)
    except ZeroDivisionError:
        ACC = 0.0
    try:
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    except ZeroDivisionError:
        MCC = 0.0

    try:
        # AUC = plot_roc(deci_value, origin_labels, output, title, roc, roc_data_file)

        AUC = roc_auc_score(origin_labels, deci_value)
    except ZeroDivisionError:
        AUC = 0.0
    # del_file(roc_data_file)
    # if roc:
    #    plot_roc_curve(origin_labels, deci_value, dest_file_path, AUC)
    BAcc = (SN + SP) / 2


    #return ACC, MCC, AUC, SN, SP, BAcc
    return AUC,ACC,MCC,PR