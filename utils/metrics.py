import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score, average_precision_score


# The code is written with reference to the examples in the sklearn.metrics library.


# =====> Multi-label Macro-level <=====
def multi_label_compute_mAP(y_true, y_scores):
    """
        Mean Average Precision (mAP)
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    ap_list = []
    for i in range(y_true.shape[1]):
        ap = average_precision_score(y_true[:, i], y_scores[:, i])
        ap_list.append(ap)

    mAP = np.mean(ap_list)
    return mAP


def multi_label_compute_F1(y_true, y_pred, threshold=0.5):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    pred = np.where(y_pred >= threshold, 1, 0)
    f1s = f1_score(y_true=y_true, y_pred=pred, average=None)
    mf1 = np.mean(f1s)
    return mf1, f1s


def multi_label_compute_AUC(y_true, y_pred):
    return roc_auc_score(y_true=y_true, y_score=y_pred, average='macro')


def multi_label_compute_Acc(y_true, y_pred, threshold=0.5):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.where(y_pred >= threshold, 1, 0)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    return acc


def multi_label_compute_recall(y_true, y_pred, threshold=0.5):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.where(y_pred >= threshold, 1, 0)
    recall = recall_score(y_true=y_true, y_pred=y_pred, average=None)
    return np.mean(recall)


def multi_label_compute_precision(y_true, y_pred, threshold=0.5):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.where(y_pred >= threshold, 1, 0)
    precision = precision_score(y_true=y_true, y_pred=y_pred, average=None)
    return np.mean(precision)


# ======> Multi Class <======

def multi_class_compute_f1(y_prob, labels):
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(labels, axis=1)

    f1 = f1_score(y_true, y_pred, average='macro')
    return f1


def multi_class_compute_acc(y_prob, labels):
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(labels, axis=1)
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    return accuracy


def multi_class_compute_auc(y_prob, labels):
    y_true = np.argmax(labels, axis=1)
    auc = roc_auc_score(y_true=y_true, y_score=y_prob, multi_class='ovr')
    return auc


def multi_class_compute_recall(y_prob, labels):
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(labels, axis=1)
    recall = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
    return recall


def multi_class_compute_precision(y_prob, labels):
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(labels, axis=1)
    precision = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
    return precision


def metric(pred, true, is_multi_label=True):
    if is_multi_label:
        auc = multi_label_compute_AUC(y_true=true, y_pred=pred)
        f1, _ = multi_label_compute_F1(y_true=true, y_pred=pred)
        acc = multi_label_compute_Acc(y_true=true, y_pred=pred)
        recall = multi_label_compute_recall(y_true=true, y_pred=pred)
        precision = multi_label_compute_precision(y_true=true, y_pred=pred)
        mAP = multi_label_compute_mAP(y_true=true, y_scores=pred)
        return auc, f1, acc, recall, precision, mAP
    else:
        auc = multi_class_compute_auc(y_prob=pred, labels=true)
        f1 = multi_class_compute_f1(y_prob=pred, labels=true)
        acc = multi_class_compute_acc(y_prob=pred, labels=true)
        recall = multi_class_compute_recall(y_prob=pred, labels=true)
        precision = multi_class_compute_precision(y_prob=pred, labels=true)
        return auc, f1, acc, recall, precision
