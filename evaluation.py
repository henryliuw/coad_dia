import lifelines
import torch
from sklearn.metrics import roc_auc_score, recall_score, f1_score, precision_score

def accuracy(result, target): 
    if result.is_cuda:
        result = result.cpu()
        target = target.cpu()
    return torch.eq(result > 0.5, target.type(torch.BoolTensor)).sum().numpy() / len(target)

def auc(result, target):
    if result.is_cuda:
        result = result.cpu()
        target = target.cpu()
    return roc_auc_score(target.numpy(), result.detach().numpy())
 
def c_index(result, df):
    if result.is_cuda:
        result = result.cpu()
    #try:
    #    cph = CoxPHFitter().fit(df, duration_col="time", event_col="y", formula="predict")
    #except:
    #    return -1
    #return cph.concordance_index_
    return lifelines.utils.concordance_index(df['time'], result.detach().numpy(), ~df['y'])

def recall(result, target):
    if result.is_cuda:
        result = result.cpu()
        target = target.cpu()
    return recall_score(target.numpy(), result.detach().numpy() > 0.5)

def f1(result, target):
    if result.is_cuda:
        result = result.cpu()
        target = target.cpu()
    return f1_score(target.numpy(), result.detach().numpy() > 0.5)

def precision(result, target):
    if result.is_cuda:
        result = result.cpu()
        target = target.cpu()
    return recall_score(target.numpy(), result.detach().numpy() > 0.5)