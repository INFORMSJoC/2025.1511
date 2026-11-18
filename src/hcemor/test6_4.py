import argparse
import os

import numpy as np
import torch
import pandas as pd
from scipy.stats import weibull_min
from sklearn.metrics import f1_score,recall_score,precision_score,roc_auc_score,accuracy_score,hamming_loss,label_ranking_loss,coverage_error,average_precision_score
from DCD_NET_v6_4 import DCDNet
from paths import (DATA_ROOT, MODEL_ROOT, emb_path, label_path,
                   support_path)
from torch.utils.data import DataLoader
target_gpu_index = int(os.environ.get("HCEMOR_GPU", "-1"))  # -1 forces CPU
# 设置要使用的 GPU 设备


if target_gpu_index >= 0 and torch.cuda.is_available():
    torch.cuda.set_device(target_gpu_index)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
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
            f1 = f1_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            f1 = 0.5  # 如果标签中只有一个类别，则返回0.5
        total_f1 += f1
    multi_f1 = total_f1 / y_true.shape[1]
    return multi_f1
def multi_label_acc(y_true, y_pred):

    # 初始化多标签AUC值
    total_acc = 0.
    # 计算每个标签的AUC值，并对所有标签的AUC值求平均
    for i in range(y_true.shape[1]):
        try:
            acc = accuracy_score(y_true[:, i], y_pred[:, i])
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
    ham=hamming_loss(y_true, y)
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
    ham = hamming_loss(y_true, y)
    rank = label_ranking_loss(y_true,y)
    coverage = coverage_error(y_true,y)
    avepre = average_precision_score(y_true,y_pred)
    return [pre_mic,pre_mac,rec_mic,rec_mac,f1_mic,f1_mac,auc,ham,acc,rank,coverage,avepre], y_true, y
def evaluate_open1(y_true, y_pred):#y_pred=[[p1,p2,p3,``],[],]
    y = np.round(y_pred).astype(np.int_)

    auc=roc_auc_score(y_true, y_pred)
    recall=recall_score(y_true, y)
    precision=precision_score(y_true, y)
    acc=accuracy_score(y_true, y)
    f1=f1_score(y_true, y)
    return [acc,precision,recall,f1,auc], y_true, y



def evaluate_open2(y_true, y_pred, thres=0.5):#y_pred=[[p1,p2,p3,``],[],]
    y_pred = y_pred.cpu().numpy()
    reject_pred = []  # [1,1,0,0,] ood?
    for p in y_pred:  # loop every test prediction
        if p > thres:
            reject_pred.append(1)  # predicted probability is greater than threshold, accept
        else:
            reject_pred.append(0)  # otherwise, reject
    auc = roc_auc_score(y_true, y_pred)
    recall = recall_score(y_true, reject_pred)
    precision = precision_score(y_true, reject_pred)
    acc = accuracy_score(y_true, reject_pred)
    f1 = f1_score(y_true, reject_pred)
    return [acc,precision,recall,f1,auc], y_true, reject_pred
def get_args():
    parser = argparse.ArgumentParser(description='Train DCD')
    parser.add_argument('--lr', default=0.0001, type=float, help="learning rate")
    parser.add_argument('--epochs', default=50, type=int, help="Number of training epochs")
    parser.add_argument('--batch_size', default=16, type=int, help="Batch size")
    parser.add_argument('--momentum', default=0.9, type=float, help="momentum")
    parser.add_argument('--weight_decay', default=0.0005, type=float, help="weight decay")
    parser.add_argument('--lossweight', default=0.7, type=float, help="weight beteween loss")
    parser.add_argument('--alpha', default=5, type=float, help="weight for attention of two level")
    parser.add_argument('--tempreture', default=1.0, type=float, help="degree of filting known class")
    parser.set_defaults(argument=True)
    return parser.parse_args()
fold = os.environ.get('HCEMOR_FOLD', '4')
EVAL_DATASET = os.environ.get('HCEMOR_DATASET', 'dataset2')
testquery_close = torch.from_numpy(np.load(emb_path(fold, 'close_test', dataset=EVAL_DATASET),
                              allow_pickle=True).astype(np.float32)).to(device)
train_X0 = torch.from_numpy(
        np.load(emb_path(fold, 'train', dataset=EVAL_DATASET), allow_pickle=True).astype(np.float32)).to(device)
closelabel = np.load(label_path(fold, 'close_test', dataset=EVAL_DATASET),
                         allow_pickle=True).astype(float)
testquery_open = torch.from_numpy(
        np.load(emb_path(fold, 'open_test', dataset=EVAL_DATASET)).astype(np.float32)).to(device)


openlabel = np.load(support_path(fold, 'open_test_results'),
                        allow_pickle=True).astype(float)

c2c = torch.from_numpy(np.load(support_path(fold, 'c2c')))
c2m = torch.from_numpy(np.load(support_path(fold, 'c2m')))
m2m = torch.from_numpy(np.load(support_path(fold, 'm2m')))
semantic_main = torch.from_numpy(np.load(support_path(fold, 'parent_class_unique'), allow_pickle=True).astype(np.float32))
semantic_sub = torch.from_numpy(np.load(support_path(fold, 'child_class'), allow_pickle=True).astype(np.float32))
X2 = np.load(support_path(None, 'top16'))

closetest = pd.DataFrame()
opentest = pd.DataFrame()
semantic_of_nan = torch.from_numpy(np.load(support_path(None, 'kmeans')))
args = get_args()
lr = args.lr
epochs = args.epochs
batch_size = args.batch_size
beta = args.lossweight
alpha = args.alpha
tempreture = args.tempreture
w = 1.0
name = 'HiCEMOR'
level1 = 0
_CKPT = os.environ.get('HCEMOR_CKPT', os.path.join(MODEL_ROOT, 'fold5', 'train_V6_410.pt'))
param = torch.load(_CKPT)#

inputs3 = torch.Tensor(X2)
bacdata = torch.from_numpy(np.load(support_path(None, 'top16'))).to(device)
model = DCDNet(semantic_main=semantic_main, semantic_sub=semantic_sub, sem_of_nan = inputs3,
                            beta=tempreture, c2m=c2m,granularity=alpha).to(device)
model.load_state_dict(param)
model.eval()

test_loader1 = DataLoader(train_X0, batch_size=batch_size, shuffle=False,drop_last=True)
test_loader2 = DataLoader(testquery_open, batch_size=batch_size, shuffle=False,drop_last=True)
test_loader3 = DataLoader(testquery_close, batch_size=batch_size, shuffle=False,drop_last=True)
res0 = []
res1 = []
openres0 = []
openres1 = []
predict_close = []
pri = []
with torch.no_grad():
    for batch in test_loader1:
        inputs = batch
        _, tempres = model(inputs,inputs)
        for item in tempres[0]:
            res0.append(item)
        for item in tempres[4]:
            res1.append(item)
    for batch in test_loader2:
        inputs = batch
        _, tempopenres = model(inputs, inputs)
        for item in tempopenres[0]:
            openres0.append(item)
        for item in tempopenres[4]:
            openres1.append(item)

    for batch in test_loader3:
        inputs = batch
        temppredict_close, _ = model(inputs, inputs)
        for item in temppredict_close:
            predict_close.append(item)


res0 = torch.stack(res0)
openres0 = torch.stack(openres0)
res1 = torch.stack(res1)
openres1 = torch.stack(openres1)
predict_close = torch.stack(predict_close)


fitdata = []
fitdata_ = []
for i in range(res0.shape[0]):
    fitdata.append(res0[i].cpu().numpy())
for i in range(res1.shape[0]):
    fitdata_.append(res1[i].cpu().numpy())

weibull_cls = []
DIS = np.array(fitdata)
print(DIS)
shap, loc, scale = weibull_min.fit(DIS)
weibull_cls.append(shap)
weibull_cls.append(loc)
weibull_cls.append(scale)

weibull_cls_ = []
DIS_ = np.array(fitdata_)

shap_, loc_, scale_ = weibull_min.fit(DIS_)
weibull_cls_.append(shap_)
weibull_cls_.append(loc_)
weibull_cls_.append(scale_)

predict_close = torch.sigmoid(predict_close).view(-1,semantic_sub.shape[0])
predict_close = predict_close.cpu().numpy()

resultclose, _, _ = evaluate_close2(closelabel[:predict_close.shape[0]], predict_close)

predict_open = []
for i in range(openres0.shape[0]):
    prob1 = weibull_min.cdf(openres0[i].cpu().numpy(), weibull_cls[0],
                           weibull_cls[1],
                           weibull_cls[2])
    prob2 = weibull_min.cdf(openres1[i].cpu().numpy(), weibull_cls_[0],
                           weibull_cls_[1],
                           weibull_cls_[2])
    prob = w * prob1 + (1 - w) * prob2
    predict_open.append(prob)

print(openres0)



resultopen, _, _ = evaluate_open1(openlabel[:len(predict_open)], predict_open)
for i in range(len(resultclose)):
    closetest.loc[level1, closemetric[i]] = resultclose[i]
    closetest.loc[level1, 'model'] = name
for j in range(len(resultopen)):
    opentest.loc[level1, openmetric[j]] = resultopen[j]
    opentest.loc[level1, 'model'] = name

print(resultclose)
print(resultopen)
_OUT = os.path.join(MODEL_ROOT, 'test_out'); os.makedirs(_OUT, exist_ok=True)
closetest.to_csv(os.path.join(_OUT, 'close.csv'))
opentest.to_csv(os.path.join(_OUT, 'open.csv'))
