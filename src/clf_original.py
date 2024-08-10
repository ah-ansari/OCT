"""
This script implements and evaluates the Original model along with several related baselines for OOD-aware
classification tasks, including Pipeline, Energy, MSP, and ReAct. These baselines act as post-processing methods that
build upon the Original model. The implementation of Energy, MSP, and ReAct is sourced from their respective GitHub
repositories.

The script supports both Evaluation Setting I (ood_class) and Evaluation Setting II (all_in_dist).
Refer to the argparse parameter descriptions for detailed configuration options.
"""

import numpy as np
import torch
import torch.nn.functional as F
import time
import argparse
import warnings

import oct
import tools
import models

warnings.filterwarnings('ignore')

# Parameters
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str,
                    help="Dataset name.")
parser.add_argument('--setting', type=str,
                    help="Experiment setting: 'all_in_dist' or 'ood_class_d', where 'd' is the OOD class label.")
parser.add_argument('--fold', type=int, default=-1,
                    help="Fold number. Use 0-4 for cross-validation, and -1 for datasets without cross-validation.")
parser.add_argument('--esp', type=int, default=20,
                    help="Early stopping patience (epochs).")
parser.add_argument('--lr', type=float, default=0.001,
                    help="Learning rate.")
parser.add_argument('--batch_size', type=int, default=64,
                    help="Batch size.")
parser.add_argument('--save', action='store_true',
                    help="Save the trained model if set.")

args = parser.parse_args()

tools.print_parameters(args)

# Loading the data
if args.fold > -1:
    data = np.load(f"saves_data/{args.dataset}_{args.setting}_{args.fold}.npz")
else:
    data = np.load(f"saves_data/{args.dataset}_{args.setting}.npz")

x_train = data['x_train']
y_train = data['y_train']

dim_cont = data['dim_cont']
dim = x_train.shape[1]
n_classes = np.unique(y_train).size

# The class label designated for OOD samples is the k-th class, where classes 0 to k-1 represent
# in-distribution categories.
class_ood = n_classes

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("dim:", dim)
print("dim_cont:", dim_cont)
print("n_classes:", n_classes)
print("device:", device)

# Training the OOD oracle
if_start_time = time.time()
ood_oracle = oct.create_ood_oracle(x_train)
if_time = time.time() - if_start_time

pred = ood_oracle.predict(x_train)
tools.print_size_percentage("number of training data detected as OOD", pred[pred == -1].size, pred.size)

print("\n-------------------------------------------------------------------------------")
print("---  Training the model  ---\n")
model_start_time = time.time()

model_original, predictor_original = models.make_model_dnn(x=x_train, y=y_train, batch_size=args.batch_size, lr=args.lr,
                                                           apply_weight=True, esp=args.esp, device=device)

model_time = time.time() - model_start_time

# Evaluations


def pipeline(x):
    pred = ood_oracle.predict(x)
    pred[pred == 1] = predictor_original(x[pred == 1])
    pred[pred == -1] = class_ood

    return pred


def get_softmax_output(model, x):
    with torch.no_grad():
        tensor_x = torch.tensor(x).float().to(device)
        output = F.softmax(model(tensor_x)).data.cpu().numpy()

    return output


# Maximum Softmax Probability (MSP)

# identifying the threshold for OOD detection: the threshold is set in a way such that 98% of the in-dist train data are
# detected as in.
softmax_output = get_softmax_output(model_original, x_train)
ood_score = -1 * np.max(softmax_output, axis=1)
ood_score = np.sort(ood_score)
thr_idx = int(ood_score.shape[0]*0.98)
thr_msp = ood_score[thr_idx]


def predictor_msp(x):
    softmax_output = get_softmax_output(model_original, x)
    pred = np.argmax(softmax_output, axis=1)
    ood_score = -1 * np.max(softmax_output, axis=1)

    ood_score = (ood_score > thr_msp)
    pred[ood_score] = class_ood

    return pred


# Energy

# the function to get the energy score, code taken from Energy's paper GitHub repository
# (the higher energy, positive, means OOD and the lower energy, negative, means in-dist)
def get_energy_score(x):
    with torch.no_grad():
        tensor_x = torch.tensor(x).float().to(device)
        output = model_original(tensor_x)
        ood_score = -1 * torch.logsumexp(output, dim=1).cpu().numpy()

    return ood_score


# identifying the threshold for OOD detection: the threshold is set in a way such that 98% of the in-dist train data are
# detected as in.
score = get_energy_score(x_train)
score = np.sort(score)
thr_idx = int(score.shape[0]*0.98)
thr_energy = score[thr_idx]


def predictor_energy(x):
    with torch.no_grad():
        tensor_x = torch.tensor(x).float().to(device)
        output = model_original(tensor_x)
        ood_score = -1 * torch.logsumexp(output, dim=1).cpu().numpy()

        _, pred = torch.max(F.softmax(output, 1), 1)
        pred = pred.data.cpu().numpy()

    ood_score = (ood_score > thr_energy)
    pred[ood_score] = class_ood

    return pred


# ReAct

# adding the functions that are required for react, code taken from ReAct's paper GitHub repository

def forward_thr(self, x, thr):
    x = F.relu(self.lin1(x))
    x = F.relu(self.lin2(x))
    x = F.relu(self.lin3(x))
    x = x.clip(max=thr)
    x = self.lin4(x)
    return x


def penultimate(self, x):
    x = F.relu(self.lin1(x))
    x = F.relu(self.lin2(x))
    x = F.relu(self.lin3(x))
    return x


models.DNNModel.forward_thr = forward_thr
models.DNNModel.penultimate = penultimate

# First, identify the cutoff threshold. The threshold is set to 90 percentile of activation for in-distribution samples.
activation = model_original.penultimate(torch.tensor(x_train).float().to(device))
react_cutoff_thr = np.percentile(activation.flatten().detach().numpy(), 90)
print("react thr: ", react_cutoff_thr)


# Second, adjust the energy threshold for React (in the same way as energy). Note that ReAct uses energy for
# OOD detection
def get_energy_score_react(x):
    with torch.no_grad():
        tensor_x = torch.tensor(x).float().to(device)
        output = model_original.forward_thr(tensor_x, react_cutoff_thr)
        ood_score = -1 * torch.logsumexp(output, dim=1).cpu().numpy()

    return ood_score


score = get_energy_score_react(x_train)
score = np.sort(score)
thr_idx = int(score.shape[0]*0.98)
thr_energy_react = score[thr_idx]


def predictor_react(x):
    with torch.no_grad():
        tensor_x = torch.tensor(x).float().to(device)
        output = model_original.forward_thr(tensor_x, react_cutoff_thr)
        ood_score = -1 * torch.logsumexp(output, dim=1).cpu().numpy()

        _, pred = torch.max(F.softmax(output, 1), 1)
        pred = pred.data.cpu().numpy()

    ood_score = (ood_score > thr_energy_react)
    pred[ood_score] = class_ood

    return pred


# Prepare the test sets
x_test = data['x_test']
y_test = data['y_test']

if args.setting == 'all_in_dist':
    ood_test1 = data['ood_test1']
    ood_test2 = data['ood_test2']
    ood_test3 = data['ood_test3']
    ood_test4 = data['ood_test4']

    x_eval1 = np.concatenate((x_test, ood_test1), axis=0)
    x_eval2 = np.concatenate((x_test, ood_test2), axis=0)
    x_eval3 = np.concatenate((x_test, ood_test3), axis=0)
    x_eval4 = np.concatenate((x_test, ood_test4), axis=0)

    x_evals = {'1': x_eval1, '2': x_eval2, '3': x_eval3, '4': x_eval4}
else:
    ood_test = data['ood_test']
    x_eval = np.concatenate((x_test, ood_test), axis=0)
    x_evals = {'1': x_eval}

# OOD test set is of equal size to the in_dist test set
y_eval_clf = np.concatenate((y_test, np.full(y_test.size, class_ood)), axis=0)

print("\n-------------------------------------------------------------------------------")
print("---  OOD-Aware Classification Evaluations  ---\n")

# structure of the dictionary: (method_name: predictor)
clf_methods = {'pipeline': pipeline, 'original': predictor_original, 'energy': predictor_energy, 'msp': predictor_msp,
               'react': predictor_react}

for ood_type, x_eval in x_evals.items():
    print("OOD ", ood_type)
    for method_name, predictor in clf_methods.items():
        tools.expr_clf(method_name=method_name, predictor=predictor, x=x_eval, y=y_eval_clf, class_ood=class_ood)

    print("")


print("\n-------------------------------------------------------------------------------")
print("---  Train time  ---\n")

pipeline_time = if_time + model_time

print("time original: ", model_time)
print("time if: ", if_time)
print("time pipeline: ", pipeline_time)

# saving the model
if args.save is True:
    save_path = "saves_model/" + args.dataset + "_" + str(args.i)
    torch.save(model_original.state_dict(), save_path + "_original")
