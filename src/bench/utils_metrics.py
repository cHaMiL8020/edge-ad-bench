import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

def macro_f1(y_true, y_pred): return f1_score(y_true, y_pred, average="macro")

def auroc(y_true, y_score):
    try: return roc_auc_score(y_true, y_score)
    except Exception: return float("nan")

def precision(y_true, y_pred): return precision_score(y_true, y_pred, zero_division=0)
def recall(y_true, y_pred):    return recall_score(y_true, y_pred, zero_division=0)

def youden_threshold(y_true, y_score):
    thresholds = np.linspace(0,1,101)
    best_t, best_j = 0.5, -1
    for t in thresholds:
        yhat = (y_score >= t).astype(int)
        tp = np.sum((y_true==1) & (yhat==1))
        fn = np.sum((y_true==1) & (yhat==0))
        tn = np.sum((y_true==0) & (yhat==0))
        fp = np.sum((y_true==0) & (yhat==1))
        sens = tp / (tp+fn+1e-9)
        spec = tn / (tn+fp+1e-9)
        j = sens + spec - 1
        if j > best_j: best_j, best_t = j, t
    return float(best_t)
