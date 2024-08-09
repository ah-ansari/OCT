import numpy as np
import torch
import torch.nn.functional as F
import time
import argparse
import warnings

import tools
import models

warnings.filterwarnings('ignore')

# Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--i', type=int, default=0)
parser.add_argument('--cv', dest='cv', action='store_true')
parser.set_defaults(cv=False)
parser.add_argument('--esp', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--save', dest='save', action='store_true')
parser.set_defaults(save=False)
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("dim: ", dim)
print("dim_cont: ", dim_cont)
print("n_classes: ", n_classes)
print("device: ", device)

# training the ood oracle
if_start_time = time.time()
ood_oracle = tools.create_ood_oracle(x_train)
if_train_time = time.time() - if_start_time

pred = ood_oracle.predict(x_train)
tools.print_size_percentage("percentage of training data detected as OOD", pred[pred == -1].size, pred.size)


print("\n-------------------------------------------------------------------------------")
print("---  Training the model  ---\n")
model_o_start_time = time.time()

model_o, predictor_o = models.make_model_dnn(x=x_train, y=y_train, batch_size=args.batch_size, lr=args.lr,
                                             apply_weight=True, esp=args.esp, device=device)

model_o_train_time = time.time() - model_o_start_time

# Evaluations


def pipeline(x):
    pred = ood_oracle.predict(x)
    pred[pred == 1] = predictor_o(x[pred == 1])
    pred[pred == -1] = class_noise

    return pred


def get_softmax_output(model, x):
    with torch.no_grad():
        tensor_x = torch.tensor(x).float().to(device)
        output = F.softmax(model(tensor_x)).data.cpu().numpy()

    return output


# Maximum Softmax Probability (MSP)

# identifying the threshold for OOD detection: the threshold is set in a way such that 98% of the in-dist train data are
# detected as in.
softmax_output = get_softmax_output(model_o, x_train)
ood_score = -1 * np.max(softmax_output, axis=1)
ood_score = np.sort(ood_score)
thr_idx = int(ood_score.shape[0]*0.98)
thr_msp = ood_score[thr_idx]


def predictor_msp(x):
    softmax_output = get_softmax_output(model_o, x)
    pred = np.argmax(softmax_output, axis=1)
    ood_score = -1 * np.max(softmax_output, axis=1)

    ood_score = (ood_score > thr_msp)
    pred[ood_score] = class_noise

    return pred


# Energy

# the function to get the energy score, code taken from Energy's paper GitHub repository
# (the higher energy, positive, means OOD and the lower energy, negative, means in-dist)
def get_energy_score(x):
    with torch.no_grad():
        tensor_x = torch.tensor(x).float().to(device)
        output = model_o(tensor_x)
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
        output = model_o(tensor_x)
        ood_score = -1 * torch.logsumexp(output, dim=1).cpu().numpy()

        _, pred = torch.max(F.softmax(output, 1), 1)
        pred = pred.data.cpu().numpy()

    ood_score = (ood_score > thr_energy)
    pred[ood_score] = class_noise

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
activation = model_o.penultimate(torch.tensor(x_train).float().to(device))
react_cutoff_thr = np.percentile(activation.flatten().detach().numpy(), 90)
print("react thr: ", react_cutoff_thr)


# Second, adjust the energy threshold for React (in the same way as energy). Note that ReAct uses energy for
# OOD detection
def get_energy_score_react(x):
    with torch.no_grad():
        tensor_x = torch.tensor(x).float().to(device)
        output = model_o.forward_thr(tensor_x, react_cutoff_thr)
        ood_score = -1 * torch.logsumexp(output, dim=1).cpu().numpy()

    return ood_score


score = get_energy_score_react(x_train)
score = np.sort(score)
thr_idx = int(score.shape[0]*0.98)
thr_energy_react = score[thr_idx]


def predictor_react(x):
    with torch.no_grad():
        tensor_x = torch.tensor(x).float().to(device)
        output = model_o.forward_thr(tensor_x, react_cutoff_thr)
        ood_score = -1 * torch.logsumexp(output, dim=1).cpu().numpy()

        _, pred = torch.max(F.softmax(output, 1), 1)
        pred = pred.data.cpu().numpy()

    ood_score = (ood_score > thr_energy_react)
    pred[ood_score] = class_noise

    return pred


x_eval1 = np.concatenate((x_test, ood_test1), axis=0)
x_eval2 = np.concatenate((x_test, ood_test2), axis=0)
x_eval3 = np.concatenate((x_test, ood_test3), axis=0)
x_eval4 = np.concatenate((x_test, ood_test4), axis=0)

x_evals = {'1': x_eval1, '2': x_eval2, '3': x_eval3, '4': x_eval4}

print("\n-------------------------------------------------------------------------------")
print("---  OOD-Aware Classification Evaluations  ---\n")

y_eval_clf = np.concatenate((y_test, np.full(y_test.size, class_noise)), axis=0)

# structure of the dictionary: (method_name: predictor)
clf_methods = {'pipeline': pipeline, 'original': predictor_o, 'energy': predictor_energy, 'msp': predictor_msp,
               'react': predictor_react}

for ood_type, x_eval in x_evals.items():
    print("OOD ", ood_type)
    for method_name, predictor in clf_methods.items():
        tools.expr_clf(method_name=method_name, predictor=predictor, x=x_eval, y=y_eval_clf, class_ood=class_noise)

    print("")


print("\n-------------------------------------------------------------------------------")
print("---  Train time  ---\n")

pipeline_train_time = if_train_time + model_o_train_time

print("train_time original: ", model_o_train_time)
print("train_time if: ", if_train_time)
print("train_time pipeline: ", pipeline_train_time)

# saving the model
if args.save is True:
    save_path = "saves_model/" + args.dataset + "_" + str(args.i)
    torch.save(model_o.state_dict(), save_path + "_o")
