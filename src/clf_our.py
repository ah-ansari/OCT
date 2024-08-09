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
parser.add_argument('--dataset', type=str)
parser.add_argument('--i', type=int, default=0)
parser.add_argument('--cv', dest='cv', action='store_true')
parser.set_defaults(cv=False)
parser.add_argument('--sigma', type=float, default=0.01)
parser.add_argument('--cat_p', type=float, default=0.1)
parser.add_argument('--n', type=float, default=2)
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

N = int(args.n * x_train.shape[0])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("dim: ", dim)
print("dim_cont: ", dim_cont)
print("n_classes: ", n_classes)
print("N: ", N)
print("device: ", device)

# training the ood oracle
if_start_time = time.time()
ood_oracle = oct.create_ood_oracle(x_train)
if_train_time = time.time() - if_start_time

pred = ood_oracle.predict(x_train)
tools.print_size_percentage("percentage of training data detected as OOD", pred[pred == -1].size, pred.size)


print("\n-------------------------------------------------------------------------------")
print("---  Training the model  ---\n")

ood_start_time = time.time()

x_train_robust, y_train_robust = oct.create_training_data(x=x_train, y=y_train,
                                                          ood_oracle=ood_oracle.predict,
                                                          index_start_cat=dim_cont,
                                                          sigma=args.sigma,
                                                          p=args.cat_p,
                                                          n=N, class_ood=class_noise)

ood_time = time.time() - ood_start_time


model_s_start_time = time.time()

model_s, predictor_s = models.make_model_dnn(x=x_train_robust, y=y_train_robust, robust=True,
                                             batch_size=args.batch_size, lr=args.lr, apply_weight=True, esp=args.esp,
                                             device=device)

model_s_train_time = time.time() - model_s_start_time


# Evaluations

def get_softmax_output(model, x):
    with torch.no_grad():
        tensor_x = torch.tensor(x).float().to(device)
        output = F.softmax(model(tensor_x)).data.cpu().numpy()

    return output


x_eval1 = np.concatenate((x_test, ood_test1), axis=0)
x_eval2 = np.concatenate((x_test, ood_test2), axis=0)
x_eval3 = np.concatenate((x_test, ood_test3), axis=0)
x_eval4 = np.concatenate((x_test, ood_test4), axis=0)

x_evals = {'1': x_eval1, '2': x_eval2, '3': x_eval3, '4': x_eval4}

print("\n-------------------------------------------------------------------------------")
print("---  OOD-Aware Classification Evaluations  ---\n")

y_eval_clf = np.concatenate((y_test, np.full(y_test.size, class_noise)), axis=0)

for ood_type, x_eval in x_evals.items():
    print("OOD ", ood_type)
    tools.expr_clf(method_name="robust", predictor=predictor_s, x=x_eval, y=y_eval_clf, ood_type=ood_type,
                   class_noise=class_noise)

    print("")


print("\n-------------------------------------------------------------------------------")
print("---  Train time  ---\n")

print("train_time if: ", if_train_time)
print("train_time ood creation: ", ood_time)
print("train_time robust: ", model_s_train_time)

# saving the model
if args.save is True:
    save_path = "saves_model/" + args.dataset + "_" + str(args.i)
    torch.save(model_s.state_dict(), save_path + "_s")
