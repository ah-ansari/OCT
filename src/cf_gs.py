import numpy as np
import torch
import argparse
import warnings

from carla.recourse_methods import GrowingSpheres

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

model_o, model_s, model_dk = cf.load_models_dnn(dim=len(train_order), i=args.i, data_name=args.dataset)
carla_model_o = cf.CarlaModelDNN(dataset, model_o, train_order)
carla_model_s = cf.CarlaModelDNN(dataset, model_s, train_order)
carla_model_dk = cf.CarlaModelDNN(dataset, model_dk, train_order)

expr = cf.CFExpr(dataset=dataset, model_o=model_o, model_s=model_s, model_dk=model_dk, n_queries=args.n_queries)

print("\n")

# dict for saving the results
result = dict()

print("*   model_o\n")
gs = GrowingSpheres(carla_model_o)
r = expr.carla_cf_expr(gs, model_name="original")
for k, v in r.items():
    result['model_o ' + k] = v


print("\n---------------------------------------------\n")

print("*   model_dk\n")
gs = GrowingSpheres(carla_model_dk)
r = expr.carla_cf_expr(gs, model_name="dk")
for k, v in r.items():
    result['model_dk ' + k] = v


print("\n---------------------------------------------\n")

print("*   model_s\n")
gs = GrowingSpheres(carla_model_s)
r = expr.carla_cf_expr(gs, model_name="robust")
for k, v in r.items():
    result['model_s ' + k] = v


print("\n\n")
print("result dict gs")
print(result)
