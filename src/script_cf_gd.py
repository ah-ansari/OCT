"""
This script runs counterfactual experiments using the Gradient Descent (GD) algorithm.
Before running this script, ensure that the OCT model and the baseline models (Original and DK)
have been trained and saved in the 'saves_model' folder.

To use this script, provide the dataset name and the number of queries as command line arguments.
"""

import numpy as np
import torch
import argparse
import warnings

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

model_original, model_oct, model_dk = cf.load_models_dnn(dim=len(train_order), data_name=args.dataset)
expr = cf.CFExpr(dataset=dataset, model_original=model_original, model_oct=model_oct, model_dk=model_dk,
                 n_queries=args.n_queries)

cf_lr = 0.05
print("cf_lr: ", cf_lr)
print("\n")

print("*   model_original\n")
expr.dice_cf_expr(model_name="original", cf_lr=cf_lr)

print("\n---------------------------------------------\n")

print("*   model_dk\n")
expr.dice_cf_expr(model_name="dk", cf_lr=cf_lr)

print("\n---------------------------------------------\n")

print("*   model_oct\n")
expr.dice_cf_expr(model_name="oct", cf_lr=cf_lr)
