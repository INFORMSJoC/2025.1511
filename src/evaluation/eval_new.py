# coding=utf-8



import numpy as np

import sklearn.metrics

import torch

import pandas as pd

from scipy.stats import weibull_min

from sklearn.metrics import f1_score,recall_score,precision_score

import os as _os

# ---------------------------------------------------------------------------

# Data root. All paths in this file are relative to it.

# Default: <repository>/data  (override with HCEMOR_DATA_ROOT env variable)

# Regenerate the dense embeddings first, see scripts/make_embeddings.md

# ---------------------------------------------------------------------------

DATA_ROOT = _os.environ.get(

    "HCEMOR_DATA_ROOT",

    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "data"),

)







def rank(data):

    index = []

    sorted_indices = np.argsort(data,kind='stable')

    for i in range(sorted_indices.shape[0]):

        index.append(np.where(sorted_indices==i)[0].item())

    index = np.array(index)

    return (index)



def alpha(data,ncls):

    temp = data[:ncls]

    alph = (10-rank(temp))/10



    return alph



def priproce(data):

    p = []

    for i in data:

        temp = []

        for item in i:

            if item > 1:

                temp.append(1)

            elif 0 <= item <= 1:

                temp.append(item)

            else:

                temp.append(0)

        p.append(temp)

    return p



def multi_label_auc(y_true, y_pred):

    """计算多标签AUC的函数

    :param y_true:真实标签，形状为[N, num_classes]

    :param y_pred:预测标签，形状为[N, num_classes]

    """

    # 初始化多标签AUC值

    total_auc = 0.

    # 计算每个标签的AUC值，并对所有标签的AUC值求平均

    for i in range(y_true.shape[1]):

        try:

            auc = sklearn.metrics.roc_auc_score(y_true[:, i], y_pred[:, i])

        except ValueError:

            auc = 0.5  # 如果标签中只有一个类别，则返回0.5

        total_auc += auc

    multi_auc = total_auc / y_true.shape[1]

    return multi_auc

def multi_label_f1(y_true, y_pred):

    """计算多标签AUC的函数

    :param y_true:真实标签，形状为[N, num_classes]

    :param y_pred:预测标签，形状为[N, num_classes]

    """

    # 初始化多标签AUC值

    total_f1 = 0.

    # 计算每个标签的AUC值，并对所有标签的AUC值求平均

    for i in range(y_true.shape[1]):

        try:

            f1 = sklearn.metrics.f1_score(y_true[:, i], y_pred[:, i])

        except ValueError:

            f1 = 0.5  # 如果标签中只有一个类别，则返回0.5

        total_f1 += f1

    multi_f1 = total_f1 / y_true.shape[1]

    return multi_f1

def multi_label_acc(y_true, y_pred):

    """计算多标签AUC的函数

    :param y_true:真实标签，形状为[N, num_classes]

    :param y_pred:预测标签，形状为[N, num_classes]

    """

    # 初始化多标签AUC值

    total_acc = 0.

    # 计算每个标签的AUC值，并对所有标签的AUC值求平均

    for i in range(y_true.shape[1]):

        try:

            acc = sklearn.metrics.accuracy_score(y_true[:, i], y_pred[:, i])

        except ValueError:

            acc = 0.5  # 如果标签中只有一个类别，则返回0.5

        total_acc += acc

    multi_acc = total_acc / y_true.shape[1]

    return multi_acc

def pridict(data):

    result = torch.sigmoid(data)

    result = torch.round(result)

    result = np.array(result,dtype=int)

    return result

closemetric = ['pre_mic','pre_mac','rec_mic','rec_mac','f1_mic','f1_mac','auc','hamming','acc','rank','cover','avepre']

openmetric = ['accuracy','prescision','recall','f1','auc']

def evaluate_close(y_true, y_pred):#y_pred=[[p1,p2,p3,``],[],]

    y = pridict(y_pred)

    y_pred = y_pred.numpy()

    acc=multi_label_acc(y_true, y)

    auc=multi_label_auc(y_true, y_pred)

    ham=sklearn.metrics.hamming_loss(y_true, y)

    f1=multi_label_f1(y_true, y)

    return [acc,ham,f1,auc], y_true, y

def evaluate_close2(y_true, y_pred):#y_pred=[[p1,p2,p3,``],[],]

    y = np.round(y_pred).astype(np.int_)

    pre_mic=precision_score(y_true, y,average='micro')

    pre_mac = precision_score(y_true, y, average='macro')

    rec_mic=recall_score(y_true, y,average='micro')

    rec_mac = recall_score(y_true, y, average='macro')

    f1_mic = f1_score(y_true, y,average='micro')

    f1_mac = f1_score(y_true, y,average='macro')

    auc = multi_label_auc(y_true, y_pred)

    acc = multi_label_acc(y_true, y)

    ham = sklearn.metrics.hamming_loss(y_true, y)

    rank = sklearn.metrics.label_ranking_loss(y_true,y)

    coverage = sklearn.metrics.coverage_error(y_true,y)

    avepre = sklearn.metrics.average_precision_score(y_true,y_pred)

    return [pre_mic,pre_mac,rec_mic,rec_mac,f1_mic,f1_mac,auc,ham,acc,rank,coverage,avepre], y_true, y

def evaluate_open1(y_true, y_pred):#y_pred=[[p1,p2,p3,``],[],] 以sklearn默认阈值为分类标准

    # y_pred = torch.sigmoid(y_pred)

    # y_pred = y_pred.numpy()

    reject_pred = []  # [1,1,0,0,] ood?

    # y_ground=[]

    prid = []

    for p in y_pred:  # loop every test prediction

        max_class = np.argmax(p)  # predicted class

        max_value = np.max(p)  # predicted probability

        reject_pred.append(1-max_value)

        prid.append(np.round(1-max_value))

    # for i in y_true:

    #     if 1 in i:

    #         y_ground.append(1)

    #     else:

    #         y_ground.append(0)

    auc=sklearn.metrics.roc_auc_score(y_true, reject_pred)

    recall=sklearn.metrics.recall_score(y_true, prid)

    precision=sklearn.metrics.precision_score(y_true, prid)

    acc=sklearn.metrics.accuracy_score(y_true, prid)

    f1=sklearn.metrics.f1_score(y_true, prid)

    return [acc,precision,recall,f1,auc], y_true, reject_pred







def evaluate_open2(y_true, y_pred, thres=0.5):#y_pred=[[p1,p2,p3,``],[],] 使用阈值判断是否拒绝，这种没有auc值

    reject_pred = []  # [1,1,0,0,] ood?

    y_ground=[]

    for p in y_pred:  # loop every test prediction

        max_class = np.argmax(p)  # predicted class

        max_value = np.max(p)  # predicted probability

        if max_value[max_class] > thres:

            reject_pred.append(0)  # predicted probability is greater than threshold, accept

        else:

            reject_pred.append(1)  # otherwise, reject

    for i in y_true:

        if 1 in i:

            y_ground.append(1)

        else:

            y_ground.append(0)

    recall=sklearn.metrics.recall_score(y_ground, reject_pred)

    precision=sklearn.metrics.precision_score(y_ground, reject_pred)

    acc=sklearn.metrics.hamming_loss(y_ground, reject_pred)

    f1=sklearn.metrics.f1_score(y_ground, reject_pred)

    return [acc,precision,recall,f1], y_ground, reject_pred











# ---------------------------------------------------------------------------
# Metric implementations used for the reported numbers. Table-level aggregation
# is done by scripts/make_tables.py from the raw per-fold outputs in results/.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("metric library; import the functions above")
