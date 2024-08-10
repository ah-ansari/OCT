"""
This script trains the OCT model and evaluates its performance on OOD-aware classification task.
It supports both Evaluation Setting I (ood_class) and Evaluation Setting II (all_in_dist).
The script allows for various experimental configurations, including the choice of evaluation setting,
applying cross-validation, and customization of OOD sample synthesis parameters. These configurations
are controlled through command-line arguments provided via argparse; refer to the parameter descriptions for details.
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
parser.add_argument('--sigma', type=float, default=0.01,
                    help="Sigma for Gaussian noise for continuous features.")
parser.add_argument('--p', type=float, default=0.1,
                    help="Perturbation probability for categorical features.")
parser.add_argument('--n', type=float, default=2,
                    help="Number of OOD samples, used in the form of a multiplier: number of OOD samples = n * size x.")
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

n_ood = int(args.n * x_train.shape[0])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("dim:", dim)
print("dim_cont:", dim_cont)
print("n_classes:", n_classes)
print("n_ood:", n_ood)
print("device:", device)

# Training the OOD oracle
if_start_time = time.time()
ood_oracle = oct.create_ood_oracle(x_train)
if_time = time.time() - if_start_time

pred = ood_oracle.predict(x_train)
tools.print_size_percentage("number of training data detected as OOD", pred[pred == -1].size, pred.size)

print("\n-------------------------------------------------------------------------------")
print("---  Training the model  ---\n")

print("Creating training OOD samples")
ood_start_time = time.time()

x_train_robust, y_train_robust = oct.create_training_data(x=x_train, y=y_train,
                                                          ood_oracle=ood_oracle.predict,
                                                          index_start_cat=dim_cont,
                                                          sigma=args.sigma, p=args.p, n=n_ood,
                                                          class_ood=class_ood)

ood_time = time.time() - ood_start_time

print("Training the model")
model_start_time = time.time()
model_oct, predictor_oct = models.make_model_dnn(x=x_train_robust, y=y_train_robust, batch_size=args.batch_size,
                                                 lr=args.lr, extra_ood_class=True, apply_weight=True, esp=args.esp,
                                                 device=device)
model_time = time.time() - model_start_time


# Evaluations
def get_softmax_output(model, x):
    with torch.no_grad():
        tensor_x = torch.tensor(x).float().to(device)
        output = F.softmax(model(tensor_x)).data.cpu().numpy()
    return output


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

for ood_type, x_eval in x_evals.items():
    print("OOD", ood_type)
    tools.expr_clf(method_name="OCT", predictor=predictor_oct, x=x_eval, y=y_eval_clf, class_ood=class_ood)
    print("")

print("\n-------------------------------------------------------------------------------")
print("---  Train time  ---\n")

print("time if:", if_time)
print("time ood creation:", ood_time)
print("time model training:", model_time)

# Saving the model
if args.save:
    save_path = f"saves_model/{args.dataset}_{args.i}"
    torch.save(model_oct.state_dict(), save_path + "_oct")
