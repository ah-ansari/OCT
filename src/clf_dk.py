"""
This script implements the Don't Know (DK) model and evaluates its performance on OOD-aware classification task.
Since no official implementation of the DK model was available, this script provides our own implementation.

For details on configuring experiments, refer to the argparse parameter descriptions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import argparse
import warnings

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

# m denotes the number of grids, identified by considering the largest value that can fit in 32GB of memory
m_dataset = {'adult': 6, 'compas': 31, 'cover': 1, 'gmsc': 11, 'heloc': 3, 'jannis': 1, 'dilbert': 1}
for d in m_dataset:
    if args.dataset.startswith(d):
        m = m_dataset[d]

print("m: ", m)

# d_m is the length of a side of a grid
d_m = 1 / m
grid = np.zeros(np.power(m, dim), dtype=bool)


def grid_ind(ind):
    a = 0
    for i in range(len(ind)-1, 0, -1):
        a += ind[i] * dim
    a += ind[0]

    return a


for i in range(x_train.shape[0]):
    ind = []
    for j in range(dim):
        a = int(x_train[i, j] / d_m)
        if a == m:
            a = m-1
        ind.append(a)

    grid[grid_ind(ind)] = 1


# create the OOD points
x_ood = []

while len(x_ood) != n_ood:
    ind = np.random.randint(low=0, high=m, size=dim)
    # ind = tuple(ind)

    if grid[grid_ind(ind)] == 0:
        ood_sample = np.zeros(dim)
        for j in range(dim):
            low = ind[j] * d_m
            high = low + d_m
            ood_sample[j] = np.random.uniform(low=low, high=high)

        x_ood.append(ood_sample)

x_ood = np.array(x_ood)


print("OOD points created")


def make_model_dont_know(x, y, batch_size=64, lr=0.001, robust=False, apply_weight=True, esp=20,
                         device=torch.device('cpu'), verbose=False):
    n_inputs = x.shape[1]
    n_classes = np.unique(y).size

    # split the validation data
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)

    tensor_x_train = torch.Tensor(x_train).float()
    tensor_y_train = torch.Tensor(y_train).long()

    tensor_x_val = torch.Tensor(x_val).float()
    tensor_y_val = torch.Tensor(y_val).long()

    data_train = TensorDataset(tensor_x_train, tensor_y_train)
    data_val = TensorDataset(tensor_x_val, tensor_y_val)

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=0,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=False, num_workers=0,
                                             pin_memory=True)

    # class weights
    if robust is True:
        print("robust")
        y_w = y[y != np.max(y)]
        c_w = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_w), y=y_w)
        c_w /= np.sum(c_w)
        c_w = np.append(c_w, 1)
        c_w = torch.Tensor(c_w).to(device)
    else:
        c_w = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
        c_w /= np.sum(c_w)
        c_w = torch.Tensor(c_w).to(device)

    model = models.DNNModel(n_inputs, n_classes, layers=[32, 16, 8])
    model.apply(models.init_weights)
    model.to(device)

    if apply_weight is True:
        print("class weights:  ", c_w)
        criterion = nn.CrossEntropyLoss(weight=c_w)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    n_epochs = 10000

    # early stopping: check if the loss is not decreasing considering a threshold, then stop after patience steps
    es_min_loss = np.inf
    es_patience = esp
    es_threshold = 0.0001
    es_i = 0

    for epoch in range(n_epochs):
        train_loss = 0.0
        val_loss = 0.0

        # train
        model.train()
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # apply sensitivity

            # calculating derivative according using the following link
            # "https://stackoverflow.com/questions/51666410/how-to-use-pytorch-to-calculate-the-gradients-of-outputs-w-r-t-the-inputs-in-a"
            x_ood = data[target == class_ood].detach().clone()
            x_ood.requires_grad = True
            y_ood = model(x_ood)[:, -1]
            y_ood = torch.mean(y_ood)
            d = torch.autograd.grad(y_ood, x_ood)[0]
            sensitivity = torch.mean(d)

            loss = loss + 0.01 * sensitivity

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)


        # validation
        model.eval()
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)

            # apply sensitivity

            # calculating derivative according using the following link
            # "https://stackoverflow.com/questions/51666410/how-to-use-pytorch-to-calculate-the-gradients-of-outputs-w-r-t-the-inputs-in-a"
            x_ood = data[target == class_ood].detach().clone()
            x_ood.requires_grad = True
            y_ood = model(x_ood)[:, -1]
            y_ood = torch.mean(y_ood)
            d = torch.autograd.grad(y_ood, x_ood)[0]
            sensitivity = torch.mean(d)

            loss = loss + 0.01 * sensitivity

            val_loss += loss.item() * data.size(0)

        train_loss = train_loss / len(train_loader.sampler)
        val_loss = val_loss / len(val_loader.sampler)

        if verbose is True:
            print(f"Epoch: {epoch} - train loss: {train_loss:.5f} - val loss: {val_loss:.5f} - "
                  f"min loss: {es_min_loss:.5f} - es_i: {es_i}")

        # Early Stopping: check if the validation loss has improved
        if (val_loss < es_min_loss) and (abs(val_loss - es_min_loss) > es_threshold):
            es_i = 0
            es_min_loss = val_loss
        else:
            es_i += 1
            if es_i == es_patience:
                print(f"Early Stopping detected    iteration: {epoch}")
                break

    predictor = models.create_predictor(model, device)
    print("\n* train")
    tools.evaluate_clf_general(predictor, x_train, y_train)
    print("\n* validation")
    tools.evaluate_clf_general(predictor, x_val, y_val)

    return model, predictor


x_dk = np.concatenate((x_train, x_ood), axis=0)
y_dk = np.concatenate((y_train, np.full(x_ood.shape[0], class_ood)), axis=0)


model, predictor = make_model_dont_know(x=x_dk, y=y_dk, robust=True, batch_size=args.batch_size, lr=args.lr,
                                        apply_weight=True, esp=args.esp, device=device)


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
    print("OOD ", ood_type)
    tools.expr_clf(method_name='dk', predictor=predictor, x=x_eval, y=y_eval_clf, class_ood=class_ood)

    print("")


# saving the model
if args.save is True:
    save_path = "saves_model/" + args.dataset + "_" + str(args.i)
    torch.save(model.state_dict(), save_path + "_dk")
