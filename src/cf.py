"""
This file provides auxiliary functions for conducting counterfactual experiments, serving as a wrapper around
the CARLA and DiCE libraries. It handles dataset loading, model integration, and evaluation processes for
counterfactual experiment.

Key Components:
* Data Handling:
   - Functions to load and process datasets from the CARLA library, preparing them for model training and evaluation.

* Counterfactual Experimentation:
   - The `CFExpr` class encapsulates the logic for running counterfactual experiments with classification models. It
     includes methods for performing experiments using CARLA and DiCE, as well as calculating cost metrics such as
     APS (Average Percentile Shift) for continues features, categorical cost, and sparsity.

* CARLA Model Integration:
   - The `CarlaModelDNN` class implements the CARLA `MLModel` interface, enabling integration of our DNN models
     with the CARLA library to generate counterfactuals.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import percentileofscore
import time

from carla.data.catalog import OnlineCatalog
from carla import MLModel
import dice_ml

import oct
import tools
import models


def get_train_order(dataset: OnlineCatalog):
    # define the order of features for training the models.
    # The order: first, continuous features, then categorical ones.
    cont = dataset.continuous
    cat = list(dataset.encoder.get_feature_names_out(dataset.categorical))
    # concatenate the two feature sets
    train_order = cont + cat

    return train_order


def load_from_dataset(dataset: OnlineCatalog):
    train_order = get_train_order(dataset)

    x_train = dataset.df_train[train_order].values
    y_train = dataset.df_train[dataset.target].values

    x_test = dataset.df_test[train_order].values
    y_test = dataset.df_test[dataset.target].values

    return x_train, y_train, x_test, y_test, train_order


def predict(model, x):
    with torch.no_grad():
        tensor_x = torch.tensor(x).float()
        output = model(tensor_x)
        output = output.data.cpu().numpy()

    pred = np.argmax(output, axis=1)

    return pred


def load_models_dnn(dim, data_name):
    n_classes = 2
    save_path = "saves_model/" + data_name

    model_original = models.DNNModel(dim, n_classes, layers=[32, 16, 8])
    model_original.load_state_dict(torch.load(save_path+"_original"))

    model_oct = models.DNNModel(dim, n_classes + 1, layers=[32, 16, 8])
    model_oct.load_state_dict(torch.load(save_path+"_oct"))

    model_dk = models.DNNModel(dim, n_classes + 1, layers=[32, 16, 8])
    model_dk.load_state_dict(torch.load(save_path + "_dk"))

    return model_original, model_oct, model_dk


def load_dataset(data_name):
    # load the dataset from carla
    if data_name == 'gmsc':
        dataset = OnlineCatalog('give_me_some_credit')
    else:
        dataset = OnlineCatalog(data_name)

    return dataset


class CFExpr:
    def __init__(self, dataset, model_original, model_oct, model_dk, n_queries):
        self.dataset = dataset
        self.model_original = model_original
        self.model_oct = model_oct
        self.model_dk = model_dk
        self.n_queries = n_queries

        self.x_train, self.y_train, self.x_test, self.y_test, self.train_order = load_from_dataset(self.dataset)
        self.ood_oracle = oct.create_ood_oracle(self.x_train)

        self.x_aps = self.dataset.df[self.dataset.continuous].values
        self.dim = len(self.train_order)
        self.dim_cont = len(self.dataset.continuous)
        self.dim_cat = self.dim - self.dim_cont

        self.query_idx = self.select_query_points()

    def select_query_points(self):
        query_class = 0

        pred_original = predict(self.model_original, self.x_test)
        pred_oct = predict(self.model_oct, self.x_test)
        pred_dk = predict(self.model_dk, self.x_test)

        query_idx = ((pred_original == query_class) & (pred_oct == query_class) & (pred_dk == query_class)
                     & (self.y_test == query_class))

        return query_idx

    def calculate_aps_cont(self, x, q):
        # percentileofscore returns percentage, the value is between 0 and 100
        aps = 0
        for i in range(self.dim_cont):
            percentile_x = percentileofscore(a=self.x_aps[:, i], score=x[i], kind='weak')
            percentile_q = percentileofscore(a=self.x_aps[:, i], score=q[i], kind='weak')
            aps += abs(percentile_x - percentile_q)

        aps = aps / self.dim_cont

        return aps

    def calculate_cost_cat(self, x, q):
        # Remind that the dataset is such that the continuous features are first then categorical features.
        # So, categorical features start from dim_cont.
        cost_cat = np.sum(x[self.dim_cont:] != q[self.dim_cont:])
        cost_cat = (cost_cat / self.dim_cat) * 100

        return cost_cat

    def calculate_sparsity(self, x, q):
        # sparsity is defined for all the features both cont and cat
        diff = np.abs(x - q)
        # the following threshold is applied to eliminate the effect of residual errors
        sparsity = np.sum(diff < 0.000001)
        sparsity = (sparsity / self.dim) * 100

        return sparsity

    def process_results(self, x, q, run_time, model_name):
        print("total run_time: ", run_time)
        print("")

        run_time /= self.n_queries
        print("run_time (per each query sample): ", run_time)
        print("")

        if x.shape[0] == 0:
            print("The method has failed to return any counterfactuals.")
            return

        if model_name == "oct":
            pred = predict(self.model_oct, x)
        elif model_name == "dk":
            pred = predict(self.model_dk, x)
        elif model_name == "original":
            pred = predict(self.model_original, x)
        else:
            print("model unknown")
            exit(1)

        tools.print_size_percentage("pred 1 (should always be 100%)", pred[pred == 1].size, pred.size)
        print("")

        n_success = x.shape[0]
        tools.print_size_percentage("success", n_success, self.n_queries)
        print("")

        # According to the IsolationForest's doc, predict returns -1 for outliers and 1 for inliers
        pred = self.ood_oracle.predict(x)
        n_ood = pred[pred == -1].size
        n_in = pred[pred == 1].size

        tools.print_size_percentage("ood", n_ood, self.n_queries)
        print("")

        tools.print_size_percentage("valid (in-dist)", n_in, self.n_queries)
        print("")

        # only consider the valid (in-distribution) CF examples in reporting the cost metrics
        x = x[pred == 1]
        q = q[pred == 1]

        if x.shape[0] == 0:
            print("The method has failed to return any in-distribution Counterfactuals.")
            return

        # cost metrics
        aps_cont = 0
        cost_cat = 0
        sparsity = 0

        for i in range(x.shape[0]):
            aps_cont += self.calculate_aps_cont(x[i], q[i])
            cost_cat += self.calculate_cost_cat(x[i], q[i])
            sparsity += self.calculate_sparsity(x[i], q[i])

        aps_cont /= x.shape[0]
        cost_cat /= x.shape[0]
        sparsity /= x.shape[0]

        print("aps_cont ", aps_cont)
        print("cost_cat ", cost_cat)
        print("sparsity ", sparsity)

        return

    def carla_cf_expr(self, cf_method, model_name):
        # prepare query points
        queries = self.dataset.df_test.iloc[self.query_idx]
        if self.n_queries < queries.shape[0]:
            queries = queries.iloc[:self.n_queries]
        else:
            self.n_queries = queries.shape[0]

        print("n_queries: ", self.n_queries)

        # CF experiment
        start_time = time.time()
        counterfactuals = cf_method.get_counterfactuals(queries)
        run_time = time.time() - start_time

        x = counterfactuals[self.train_order].values
        q = queries[self.train_order].values

        # remove CFs with nan (in carla CFs that has failed are filled with nan)
        x_nan = np.isnan(x[:, 0])
        x = x[~ x_nan]
        q = q[~ x_nan]

        self.process_results(x, q, run_time, model_name)

        return

    def dice_cf_expr(self, model_name, cf_lr):
        print("# " + model_name)
        if model_name == "oct":
            model = self.model_oct
            target_th = 1 / 3
        elif model_name == "dk":
            model = self.model_dk
            target_th = 1 / 3
        elif model_name == "original":
            model = self.model_original
            target_th = 1 / 2
        else:
            print("model_name unknown")
            exit(1)

        # prepare the dataset for the format of dice
        df_columns = self.dataset.continuous + self.dataset.categorical
        df_columns.append(self.dataset.target)

        df = self.dataset.inverse_transform(self.dataset.df)
        df = df[df_columns]

        features_to_vary = [c for c in df_columns if c not in self.dataset.immutables and c != self.dataset.target]

        print("features_to_vary")
        print(features_to_vary)

        # prepare query points
        queries = self.dataset.df_test.iloc[self.query_idx]

        if self.n_queries > queries.shape[0]:
            self.n_queries = queries.shape[0]

        print("n_queries ", self.n_queries)

        # change the query format to dice (dice accepts queries in their original format)
        queries = self.dataset.inverse_transform(queries)
        queries = queries[df_columns]

        # CF expr
        # prepare arrays for saving the results
        r_x = []
        r_q = []
        run_time = 0

        # Step 1: dice_ml.Data
        d = dice_ml.Data(dataframe=df, continuous_features=self.dataset.continuous, outcome_name=self.dataset.target)

        # Step 2: dice_ml.Model
        backend = 'PYT'
        m = dice_ml.Model(model=model, backend=backend)

        # Step 3: initiate DiCE
        if model_name == "oct" or model_name == "dk":
            print("dice multi model selected")
            exp = dice_ml.Dice(d, m, method='multi')
        else:
            exp = dice_ml.Dice(d, m)

        for query_index in range(self.n_queries):
            query = queries.iloc[query_index].to_dict()

            start_time = time.time()

            # generate counterfactuals
            dice_exp = exp.generate_counterfactuals(query,
                                                    total_CFs=1,
                                                    desired_class=1,
                                                    diversity_weight=0,
                                                    learning_rate=cf_lr,
                                                    stopping_threshold=target_th,
                                                    features_to_vary=features_to_vary,
                                                    posthoc_sparsity_param=None)

            run_time += time.time() - start_time

            # x = dice_exp.cf_examples_list[0].final_cfs_df_sparse
            x = dice_exp.cf_examples_list[0].final_cfs_df

            # check if the method has found a CF example
            if x.shape[0] == 0:
                # the method has failed to return valid CF example, skip and does not add anything to r_x
                continue

            # transform the returned CF example to the [0, 1] range
            x = d.get_ohe_min_max_normalized_data(x[d.feature_names]).values
            x = x[0]

            # transform the query
            q = queries.iloc[query_index: (query_index + 1)]
            q = d.get_ohe_min_max_normalized_data(q[d.feature_names]).values
            q = q[0]

            r_x.append(x)
            r_q.append(q)

        # processing the results
        r_x = np.array(r_x)
        r_q = np.array(r_q)

        self.process_results(r_x, r_q, run_time, model_name)

        return


# the DNN MLModel interface for CARLA methods
class CarlaModelDNN(MLModel):
    def __init__(self, data, model, train_order):
        super().__init__(data)
        # The constructor can be used to load or build an
        # arbitrary black-box-model
        self._mymodel = model
        self.train_order = train_order

    # List of the feature order the ml model was trained on
    @property
    def feature_input_order(self):
        return self.train_order

    # The ML framework the model was trained on
    @property
    def backend(self):
        return "pytorch"

    # The black-box model object
    @property
    def raw_model(self):
        return self._mymodel

    # The predict function outputs
    # the continuous prediction of the model
    # Not used in the methods, and not implemented
    def predict(self, x):
        return None

    # The predict_proba method outputs
    # the prediction as class probabilities
    def predict_proba(self, x):
        if isinstance(x, pd.DataFrame):
            x = x[self.train_order].values
        if not torch.is_tensor(x):
            x = torch.from_numpy(np.array(x)).float()

        output = self._mymodel(x)
        output = F.softmax(output, dim=1)

        return output.data.cpu().numpy()

    def predict_proba_diff(self, x):
        output = self._mymodel(x)
        output = F.softmax(output, dim=1)

        return output
