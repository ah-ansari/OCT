"""
This script runs counterfactual experiments using the Revise algorithm.
Before running this script, ensure that the OCT model and the baseline models (Original and DK)
have been trained and saved in the 'saves_model' folder.

To use this script, provide the dataset name and the number of queries as command line arguments.
"""

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
parser.add_argument('--dataset', type=str,
                    help="Dataset name.")
parser.add_argument('--n_queries', type=int, default=500,
                    help="Number of query samples.")
args = parser.parse_args()

tools.print_parameters(args)

# load the dataset
dataset = cf.load_dataset(args.dataset)
train_order = cf.get_train_order(dataset)

x_train = dataset.df_train[train_order].values
y_train = dataset.df_train[dataset.target].values

n_classes = np.unique(y_train).size
class_noise = n_classes

model_original, model_oct, model_dk = cf.load_models_dnn(dim=len(train_order), data_name=args.dataset)
carla_model_original = cf.CarlaModelDNN(dataset, model_original, train_order)
carla_model_oct = cf.CarlaModelDNN(dataset, model_oct, train_order)
carla_model_dk = cf.CarlaModelDNN(dataset, model_dk, train_order)

expr = cf.CFExpr(dataset=dataset, model_original=model_original, model_oct=model_oct, model_dk=model_dk,
                 n_queries=args.n_queries)

print("\n")

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


print("*   model_original\n")
revise = Revise(carla_model_original, dataset, params_o)
expr.carla_cf_expr(revise, model_name="original")

print("\n---------------------------------------------\n")

print("*   model_dk\n")
revise = Revise(carla_model_dk, dataset, params_s)
expr.carla_cf_expr(revise, model_name="dk")

print("\n---------------------------------------------\n")

print("*   model_oct\n")
revise = Revise(carla_model_oct, dataset, params_s)
expr.carla_cf_expr(revise, model_name="oct")
