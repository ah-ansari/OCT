import numpy as np
import torch
import argparse
import warnings

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
expr = cf.CFExpr(dataset=dataset, model_o=model_o, model_s=model_s, model_dk=model_dk, n_queries=args.n_queries)

cf_lr = 0.05
print("cf_lr: ", cf_lr)

print("\n")

# dict for saving the results
result = dict()

print("*   model_o\n")
r = expr.dice_cf_expr(model_name="original", cf_lr=cf_lr)
for k, v in r.items():
    result['model_o ' + k] = v

print("\n---------------------------------------------\n")

print("*   model_dk\n")
r = expr.dice_cf_expr(model_name="dk", cf_lr=cf_lr)
for k, v in r.items():
    result['model_dk ' + k] = v

print("\n---------------------------------------------\n")

print("*   model_s\n")
r = expr.dice_cf_expr(model_name="robust", cf_lr=cf_lr)
for k, v in r.items():
    result['model_s ' + k] = v


print("\n\n")
print("result dict gd")
print(result)
