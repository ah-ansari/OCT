import numpy as np
import torch
import argparse
import warnings

from carla.recourse_methods import Revise

import tools
import cf

warnings.filterwarnings('ignore')

# Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--i', type=int, default=0)
parser.add_argument('--n_queries', type=int, default=500)
args = parser.parse_args()

tools.print_parameters(args)

# load the dataset
dataset = cf.load_dataset(args.dataset)
train_order = cf.get_train_order(dataset)

x_train = dataset.df_train[train_order].values
y_train = dataset.df_train[dataset.target].values

n_classes = np.unique(y_train).size
class_noise = n_classes

model_o, model_s, model_dk = cf.load_models_dnn(dim=len(train_order), i=args.i, data_name=args.dataset)
carla_model_o = cf.CarlaModelDNN(dataset, model_o, train_order)
carla_model_s = cf.CarlaModelDNN(dataset, model_s, train_order)
carla_model_dk = cf.CarlaModelDNN(dataset, model_dk, train_order)

expr = cf.CFExpr(dataset=dataset, model_o=model_o, model_s=model_s, model_dk=model_dk, n_queries=args.n_queries)

print("\n")

# dict for saving the results
result = dict()

if args.dataset == "compas":
    layers = [len(train_order) - len(dataset.immutables), 8, 10, 5]
else:
    layers = [len(train_order) - len(dataset.immutables), 16, 32, 10]

print("layers ", layers)
print("\n")


params_o = {
    "data_name": args.dataset,
    "lambda": 0.5,
    "optimizer": "adam",
    "lr": 0.1,
    "max_iter": 1000,
    "target_class": [0, 1],
    "binary_cat_features": True,
    "vae_params": {
        "layers": layers,
        "train": True,
        "lambda_reg": 1e-6,
        "epochs": 5,
        "lr": 1e-3,
        "batch_size": 32,
    },
}

params_s = {
    "data_name": args.dataset,
    "lambda": 0.5,
    "optimizer": "adam",
    "lr": 0.1,
    "max_iter": 1000,
    "target_class": [0, 1, 0],
    "binary_cat_features": True,
    "vae_params": {
        "layers": layers,
        "train": True,
        "lambda_reg": 1e-6,
        "epochs": 5,
        "lr": 1e-3,
        "batch_size": 32,
    },
}


print("*   model_o\n")
revise = Revise(carla_model_o, dataset, params_o)
r = expr.carla_cf_expr(revise, model_name="original")
for k, v in r.items():
    result['model_o ' + k] = v

print("\n---------------------------------------------\n")

print("*   model_dk\n")
revise = Revise(carla_model_dk, dataset, params_s)
r = expr.carla_cf_expr(revise, model_name="dk")
for k, v in r.items():
    result['model_dk ' + k] = v

print("\n---------------------------------------------\n")

print("*   model_s\n")
revise = Revise(carla_model_s, dataset, params_s)
r = expr.carla_cf_expr(revise, model_name="robust")
for k, v in r.items():
    result['model_s ' + k] = v


print("\n\n")
print("result dict revise")
print(result)