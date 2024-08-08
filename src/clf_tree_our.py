import numpy as np
import time
import argparse
import warnings
from sklearn import metrics

import tools
import models

warnings.filterwarnings('ignore')

# Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--i', type=int, default=0)
parser.add_argument('--cv', dest='cv', action='store_true')
parser.set_defaults(cv=False)
parser.add_argument('--cont_std', type=float, default=0.01)
parser.add_argument('--cat_p', type=float, default=0.1)
parser.add_argument('--n', type=float, default=2)
args = parser.parse_args()

tools.print_parameters(args)

# loading the data
if args.cv is True:
    fold = args.i % 5
    print("fold: ", fold)
    data = np.load("saves_data/" + args.dataset + "_" + str(fold) + ".npz")
else:
    data = np.load("saves_data/" + args.dataset + ".npz")

x_train = data['x_train']
x_test = data['x_test']
y_train = data['y_train']
y_test = data['y_test']
dim_cont = data['dim_cont']
ood_test1 = data['ood_test1']
ood_test2 = data['ood_test2']
ood_test3 = data['ood_test3']
ood_test4 = data['ood_test4']

dim = x_train.shape[1]
n_classes = np.unique(y_train).size
class_noise = n_classes

N = int(args.n * x_train.shape[0])

print("dim: ", dim)
print("dim_cont: ", dim_cont)
print("n_classes: ", n_classes)
print("N: ", N)

max_depth = 15
print("max_depth: ", max_depth)

# training the ood oracle
if_start_time = time.time()
ood_oracle = tools.create_ood_oracle(x_train)
if_train_time = time.time() - if_start_time

pred = ood_oracle.predict(x_train)
tools.print_size_percentage("percentage of training data detected as OOD", pred[pred == -1].size, pred.size)


print("\n-------------------------------------------------------------------------------")
print("---  Training the model  ---\n")

ood_start_time = time.time()

x_train_robust, y_train_robust = tools.prepare_data_robust(x=x_train, y=y_train,
                                                           detector=ood_oracle.predict,
                                                           index_start_cat=dim_cont,
                                                           sigma=args.cont_std,
                                                           cat_p=args.cat_p,
                                                           N=N, class_noise=class_noise)

ood_time = time.time() - ood_start_time

model_s_start_time = time.time()

model_s = models.make_model_tree(x=x_train_robust, y=y_train_robust, n_classes=n_classes, max_depth=max_depth,
                                 robust=True, apply_weight=True)

model_s_train_time = time.time() - model_s_start_time


# printing number of in-dist samples detected as OOD
print("")
pred = model_s.predict(x_test)
tools.print_size_percentage("Number of in-dist points detected as OOD: ", pred[pred == class_noise].size, pred.size)
print("")

# Evaluations

# dict for saving the results
result = dict()

x_eval1 = np.concatenate((x_test, ood_test1), axis=0)
x_eval2 = np.concatenate((x_test, ood_test2), axis=0)
x_eval3 = np.concatenate((x_test, ood_test3), axis=0)
x_eval4 = np.concatenate((x_test, ood_test4), axis=0)

x_evals = {'1': x_eval1, '2': x_eval2, '3': x_eval3, '4': x_eval4}

print("\n-------------------------------------------------------------------------------")
print("---  Classification evaluations  ---\n")

y_eval_clf = np.concatenate((y_test, np.full(y_test.size, class_noise)), axis=0)

for ood_type, x_eval in x_evals.items():
    print("OOD ", ood_type)
    r = tools.expr_clf(method_name="robust", predictor=model_s.predict, x=x_eval, y=y_eval_clf, ood_type=ood_type,
                       class_noise=class_noise)
    result.update(r)

    print("")


print("\n-------------------------------------------------------------------------------")
print("---  OOD detection evaluations  ---\n")

# positive class (value 1): ood - negative class (value -1): in-dist
y_eval_ood = np.concatenate((np.full(x_test.shape[0], -1), np.full(x_test.shape[0], 1)), axis=0)

for ood_type, x_eval in x_evals.items():
    print("OOD ", ood_type)

    # Model Robust
    output = model_s.predict_proba(x_eval)
    output = output[:, class_noise]
    r = tools.expr_ood(method_name='robust', y=y_eval_ood, ood_score=output, ood_type=ood_type)
    result.update(r)

    print("")

print("\n")

print("\n-------------------------------------------------------------------------------")
print("---  Train time  ---\n")

print("train_time ood creation: ", ood_time)
print("train_time robust: ", model_s_train_time)
print("train_time if: ", if_train_time)

result['train_time ood'] = ood_time
result['train_time robust'] = model_s_train_time
result['train_time if'] = if_train_time


print("\n-------------------------------------------------------------------------------")
print("---  Maximum actual class probabilities  ---\n")

test_sets = {'in_dist': x_test, 'ood1': ood_test1, 'ood2': ood_test2, 'ood3': ood_test3, 'ood4': ood_test4}

print("Robust:")
for name, test_set in test_sets.items():
    output = model_s.predict_proba(test_set)[:, :-1]
    output = np.max(output, axis=1)
    avg = np.average(output)
    std = np.std(output)

    print("{}    {:.2f} $\pm$ {:.2f}".format(name, avg, std))
    result['act_class robust ' + name] = avg


print("\n-------------------------------------------------------------------------------")
print("---  OOD probability  ---\n")

for name, test_set in test_sets.items():
    output = model_s.predict_proba(test_set)[:, -1]
    avg = np.average(output)
    std = np.std(output)

    print("{}    {:.2f} $\pm$ {:.2f}".format(name, avg, std))
    result['ood_prob '+name] = avg


print("\n-------------------------------------------------------------------------------")
print("---  in-dist classification evaluations  ---\n")
pred = model_s.predict(x_test)

if len(pred.shape) != 1:
    pred = np.argmax(pred, axis=1)

in_clf_f1_macro = metrics.f1_score(y_true=y_test, y_pred=pred, labels=range(n_classes), average='macro', zero_division=0)
in_clf_f1_weighted = metrics.f1_score(y_true=y_test, y_pred=pred, labels=range(n_classes), average='weighted', zero_division=0)

print("robust in_clf_f1_macro :", in_clf_f1_macro)
print("robust in_clf_f1_weighted :", in_clf_f1_weighted)

result["robust in_clf_f1_macro"] = in_clf_f1_macro
result["robust in_clf_f1_weighted"] = in_clf_f1_weighted

print("\n")

print("\n\n")

print("result dict clf_tree_our")
print(result)
