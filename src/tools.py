import pandas as pd
import numpy as np
from sklearn import metrics
import time


def load_dataset(dataset):
    path = 'datasets/'

    if dataset == 'cover':
        return load_cover(path)
    elif dataset == 'dilbert':
        return load_dilbert(path)
    elif dataset == 'jannis':
        return load_jannis(path)
    else:
        print('dataset not in the list!')
        exit(1)

    return


def load_cover(path):
    # for cover (Forest CoverType) the first 10 features are continuous and other features are categorical. Since the
    # features are binary valued, and we apply one-hot encoding with drop="if_binary" option, so no need to encode them.

    df = pd.read_csv(path+'covtype.data', header=None)

    x = df.iloc[:, :54].values
    x = x.astype(float)
    y = df.iloc[:, 54].values
    y -= 1

    return x, y, 10


def load_dilbert(path):
    df_x = pd.read_csv(path + 'dilbert_train.data', header=None, delimiter=' ')
    df_y = pd.read_csv(path + 'dilbert_train.solution', header=None, delimiter=' ')

    x = df_x.values
    y = df_y.values

    # remove 'nan', which is caused by the extra space at end of each row in the dataset
    x = x[:, :-1]
    y = y[:, :-1]
    y = np.argmax(y, axis=1)

    return x, y, x.shape[1]


def load_jannis(path):
    df_x = pd.read_csv(path + 'jannis_train.data', header=None, delimiter=' ')
    df_y = pd.read_csv(path + 'jannis_train.solution', header=None, delimiter=' ')

    x = df_x.values
    y = df_y.values

    # remove 'nan', which is caused by the extra space at end of each row in the dataset
    x = x[:, :-1]
    y = y[:, :-1]
    y = np.argmax(y, axis=1)

    return x, y, x.shape[1]


def evaluate_clf_general(f_predict, x, y):
    pred = f_predict(x)

    if len(pred.shape) != 1:
        pred = np.argmax(pred, axis=1)

    p = metrics.precision_score(y_true=y, y_pred=pred, average=None, zero_division=0)
    r = metrics.recall_score(y_true=y, y_pred=pred, average=None, zero_division=0)
    f1 = metrics.f1_score(y_true=y, y_pred=pred, average=None, zero_division=0)

    for i in range(p.size):
        print(f"class {i}    p: {p[i]:.4f} - r: {r[i]:.4f} - f1: {f1[i]:.4f}")

    return


def evaluate_clf(f_predict, x, y, class_noise):
    start_time = time.time()
    pred = f_predict(x)
    run_time = time.time() - start_time

    if len(pred.shape) != 1:
        pred = np.argmax(pred, axis=1)

    # metrics for in-dist data
    in_labels = np.unique(y)
    in_labels = np.delete(in_labels, class_noise)
    in_f1_macro = metrics.f1_score(y_true=y, y_pred=pred, labels=in_labels, average='macro', zero_division=0)
    in_f1_weighted = metrics.f1_score(y_true=y, y_pred=pred, labels=in_labels, average='weighted', zero_division=0)

    # metrics for the ood class
    ood_p = metrics.precision_score(y_true=y, y_pred=pred, labels=[class_noise], average=None, zero_division=0)[0]
    ood_r = metrics.recall_score(y_true=y, y_pred=pred, labels=[class_noise], average=None, zero_division=0)[0]
    ood_f1 = metrics.f1_score(y_true=y, y_pred=pred, labels=[class_noise], average=None, zero_division=0)[0]

    # printing results
    print(f"prediction time:  {run_time}  seconds")
    print("in-dist data")
    print(f"  f1-macro: {in_f1_macro:.4f} - f1-weighted: {in_f1_weighted:.4f}")
    print("ood")
    print(f"  p: {ood_p:.4f} - r: {ood_r:.4f} - f1: {ood_f1:.4f}")

    return


def print_size_percentage(name_str, n_samples, n_total):
    percentage = (n_samples / n_total) * 100
    print(name_str, ":   number ", n_samples, " percentage ", percentage)
    return percentage


def print_parameters(args):
    print("Parameters:")
    for k, v in vars(args).items():
        print(k, " " * (20 - len(k)), v)
    print("-------------------------------------------------------------------------------")

    return


def expr_clf(method_name, predictor, x, y, class_ood):
    print(method_name + "\n")
    evaluate_clf_general(predictor, x, y)
    evaluate_clf(predictor, x, y, class_ood)
    print("---------------------------------------------")

    return
