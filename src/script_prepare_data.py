"""
This script prepares data for the experiments conducted in the paper.

### Datasets:
- Cover, Dilbert, Jannis: These datasets must be downloaded and placed in the 'datasets/' folder. They do not
  include public train-test splits, so five-fold cross-validation is applied. These datasets are loaded using
  functions from the 'tools.py' file.
- Adult, Compas, GMSC, Heloc: These datasets are automatically loaded from the CARLA library, which provides
  predefined public train-test splits used in our experiments.

Note: All datasets follow a format where continuous features precede categorical features. Additionally,
all categorical features are binary. Our method is implemented based on these two assumptions.

### Preprocessing:
- Continuous features are scaled using a MinMax scaler.
- Categorical features remain unchanged since they are binary across all datasets.

### Experiment Settings:
The script prepares data according to two test settings:
1. Setting I (ood_class): One class is considered OOD, and the script adjusts the training and test data accordingly.
2. Setting II (all_in_dist): The entire dataset is treated as in-distribution, and OOD test samples are synthesized.

For both settings, the script generates:
- In-distribution training data: Used as input for the OCT model and other baselines.
- In-distribution and OOD testing sets: Used for model evaluation. In Setting I, each class is treated as
  the OOD class once. In Setting II, four different OOD test sets are synthesized.

### Output:
- The prepared data is saved in the 'saves_data/' folder as `.npz` files.
- Files are named `[dataset_name]_[setting_type]_[fold_number].npz` for datasets with cross-validation, and
  `[dataset_name]_[setting_type].npz` for CARLA datasets.

### Key Functions:
- `prepare_data_cross_validation_all_in_dist`: Prepares data according to Setting II for datasets requiring
  five-fold cross-validation.
- `prepare_data_carla_all_in_dist`: Prepares data according to Setting II for CARLA datasets (no cross-validation
  required).
- `prepare_data_carla_ood_class`: Prepares data according to Setting I for CARLA datasets (no cross-validation
  required).
- `prepare_data_cross_validation_ood_class`: Prepares data according to Setting I for datasets requiring
  five-fold cross-validation, treating one class as OOD.

The script saves the generated datasets for subsequent use in model training and evaluation.
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn import preprocessing
import warnings

import oct
import tools
import cf

warnings.filterwarnings('ignore')

save_folder = 'saves_data/'


def prepare_data_cross_validation_all_in_dist(dataset_name):
    # loading the dataset
    # each dataset is assumed to be in the format of first continues features and then categorical features
    x, y, dim_cont = tools.load_dataset(dataset_name)
    x = x.astype(float)
    dim_cat = x.shape[1] - dim_cont

    print(dataset_name + "\n")

    print("number of samples in each class:")
    for c in np.unique(y):
        tools.print_size_percentage("class " + str(c), y[y == c].size, y.size)
    print("")
    print("x shape: ", x.shape)
    print("dim_cont: ", dim_cont)
    print("dim_cat: ", dim_cat)

    # one hot encoding of the categorical features is applied at the time the datasets are loaded, but continuous
    # features are not scaled, and scaling is applied here
    scaler = preprocessing.MinMaxScaler()
    x[:, :dim_cont] = scaler.fit_transform(x[:, :dim_cont])

    kf = KFold(n_splits=5, random_state=0, shuffle=True)

    for i, (train_index, test_index) in enumerate(kf.split(x)):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # training the ood oracle
        ood_oracle = oct.create_ood_oracle(x_train)

        # creating ood test: sigma=0.01, p=0.1
        ood_test1 = oct.create_ood(x=x_test, ood_oracle=ood_oracle.predict, index_start_cat=dim_cont,
                                   sigma=0.01, p=0.1, n=x_test.shape[0])

        # creating ood test: sigma=0.1, p=0.2
        ood_test2 = oct.create_ood(x=x_test, ood_oracle=ood_oracle.predict, index_start_cat=dim_cont,
                                   sigma=0.1, p=0.2, n=x_test.shape[0])

        # creating ood test: sigma=0.5, p=0.4
        ood_test3 = oct.create_ood(x=x_test, ood_oracle=ood_oracle.predict, index_start_cat=dim_cont,
                                   sigma=0.5, p=0.4, n=x_test.shape[0])

        # creating ood test: ood uniform
        ood_test4 = oct.sample_ood_uniform(ood_oracle=ood_oracle.predict, n=x_test.shape[0], dim_cont=dim_cont,
                                           dim_cat=dim_cat)

        # saving the created sets
        file_name = save_folder + dataset_name + "_all_in_dist_" + str(i)
        np.savez(file_name, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, dim_cont=dim_cont,
                 ood_test1=ood_test1, ood_test2=ood_test2, ood_test3=ood_test3, ood_test4=ood_test4)

    print("\n---------------------------------------------------------------------\n")
    return


def prepare_data_carla_all_in_dist(dataset_name):
    print(dataset_name)

    dataset = cf.load_dataset(dataset_name)
    x_train, y_train, x_test, y_test, _ = cf.load_from_dataset(dataset)

    dim_cont = len(dataset.continuous)
    dim_cat = len(dataset.categorical)

    # creating the ood oracle
    ood_oracle = oct.create_ood_oracle(x_train)

    # creating ood test: sigma=0.01, p=0.1
    ood_test1 = oct.create_ood(x=x_test, ood_oracle=ood_oracle.predict, index_start_cat=dim_cont,
                               sigma=0.01, p=0.1, n=x_test.shape[0])

    # creating ood test: sigma=0.1, p=0.2
    ood_test2 = oct.create_ood(x=x_test, ood_oracle=ood_oracle.predict, index_start_cat=dim_cont,
                               sigma=0.1, p=0.2, n=x_test.shape[0])

    # creating ood test: sigma=0.5, p=0.4
    ood_test3 = oct.create_ood(x=x_test, ood_oracle=ood_oracle.predict, index_start_cat=dim_cont,
                               sigma=0.5, p=0.4, n=x_test.shape[0])

    # creating ood test: ood uniform
    ood_test4 = oct.sample_ood_uniform(ood_oracle=ood_oracle.predict, n=x_test.shape[0], dim_cont=dim_cont,
                                       dim_cat=dim_cat)

    # saving the created sets
    file_name = save_folder + dataset_name + "_all_in_dist"
    np.savez(file_name, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, dim_cont=dim_cont,
             ood_test1=ood_test1, ood_test2=ood_test2, ood_test3=ood_test3, ood_test4=ood_test4)

    print("\n---------------------------------------------------------------------\n")

    return


def fix_y_ood_class(y, ood_class):
    for i in range(len(y)):
        if y[i] == ood_class:
            y[i] = -1
        elif y[i] > ood_class:
            y[i] -= 1
    return y


def prepare_data_cross_validation_ood_class(dataset_name):
    # loading the dataset
    x, y, dim_cont = tools.load_dataset(dataset_name)
    x = x.astype(float)
    dim_cat = x.shape[1] - dim_cont

    print(dataset_name + "\n")

    print("number of samples in each class:")
    for c in np.unique(y):
        tools.print_size_percentage("class " + str(c), y[y == c].size, y.size)
    print("")
    print("x shape: ", x.shape)
    print("dim_cont: ", dim_cont)
    print("dim_cat: ", dim_cat)

    # one hot encoding of the categorical features is applied at the time the datasets are loaded, but continuous
    # features are not scaled, and scaling is applied here
    scaler = preprocessing.MinMaxScaler()
    x[:, :dim_cont] = scaler.fit_transform(x[:, :dim_cont])

    for ood_class in np.unique(y):
        print("* ood_class: ", ood_class)

        y_fix = fix_y_ood_class(y.copy(), ood_class)

        x_in = x[y_fix != -1]
        y_in = y_fix[y_fix != -1]
        x_ood_orig = x[y_fix == -1]

        print("x_in shape: ", x_in.shape)
        print("x_ood shape: ", x_ood_orig.shape)

        kf = KFold(n_splits=5, random_state=0, shuffle=True)

        print("\n")

        for i, (train_index, test_index) in enumerate(kf.split(x_in)):
            print("fold: ", i)

            x_train_in, x_test_in = x_in[train_index], x_in[test_index]
            y_train_in, y_test_in = y_in[train_index], y_in[test_index]

            # training the ood oracle
            ood_oracle = oct.create_ood_oracle(x_train_in)

            pred = ood_oracle.predict(x_ood_orig)
            x_ood = x_ood_orig[pred == -1]

            print("x_ood shape (after filtering): ", x_ood.shape)
            print("x_test shape (before applying min with x_test): ", x_test_in.shape)

            m = min(x_ood.shape[0], x_test_in.shape[0])
            x_test_in = x_test_in[:m]
            y_test_in = y_test_in[:m]
            x_ood = x_ood[:m]

            print("sizes after applying min: ")
            print("x_test: ", x_test_in.shape)
            print("x_ood: ", x_ood.shape)

            print("number of samples in each class of the test set:")
            for c in np.unique(y_test_in):
                tools.print_size_percentage("class " + str(c), y_test_in[y_test_in == c].size, y_test_in.size)

            # saving the created sets
            file_name = save_folder + dataset_name + "_ood_class" + str(ood_class) + "_" + str(i)
            np.savez(file_name, x_train=x_train_in, x_test=x_test_in, y_train=y_train_in, y_test=y_test_in,
                     dim_cont=dim_cont, ood_test=x_ood)

            print("\n----------------------------------\n")
        print("===================================================")
    print("\n---------------------------------------------------------------------\n")
    return


def prepare_data_carla_ood_class(dataset_name):
    print(dataset_name)

    dataset = cf.load_dataset(dataset_name)
    x_train, y_train, x_test, y_test, _ = cf.load_from_dataset(dataset)

    print("number of samples in each class:")
    for c in np.unique(y_train):
        tools.print_size_percentage("class " + str(c), y_train[y_train == c].size, y_train.size)

    dim_cont = len(dataset.continuous)
    dim_cat = len(dataset.categorical)

    print("dim_cont: ", dim_cont)
    print("dim_cat: ", dim_cat)

    print("\n")

    for ood_class in np.unique(y_train):
        print("* ood_class: ", ood_class)

        y_train_fix = fix_y_ood_class(y_train.copy(), ood_class)
        y_test_fix = fix_y_ood_class(y_test.copy(), ood_class)

        x_train_in = x_train[y_train_fix != -1]
        y_train_in = y_train_fix[y_train_fix != -1]

        x_test_in = x_test[y_test_fix != -1]
        y_test_in = y_test_fix[y_test_fix != -1]

        x_ood = np.concatenate((x_train[y_train_fix == -1], x_test[y_test_fix == -1]), axis=0)

        print("x_in shape: ", x_train_in.shape)
        print("x_ood shape: ", x_ood.shape)

        # training the ood oracle
        ood_oracle = oct.create_ood_oracle(x_train_in)

        pred = ood_oracle.predict(x_ood)
        x_ood = x_ood[pred == -1]

        print("x_ood shape (after filtering): ", x_ood.shape)
        print("x_test shape (before applying min with x_test): ", x_test_in.shape)

        m = min(x_ood.shape[0], x_test_in.shape[0])
        x_test_in = x_test_in[:m]
        y_test_in = y_test_in[:m]
        x_ood = x_ood[:m]

        print("sizes after applying min: ")
        print("x_test: ", x_test_in.shape)
        print("x_ood: ", x_ood.shape)

        print("number of samples in each class of the test set:")
        for c in np.unique(y_test_in):
            tools.print_size_percentage("class " + str(c), y_test_in[y_test_in == c].size, y_test_in.size)

        # saving the created sets
        file_name = save_folder + dataset_name + "_ood_class" + str(ood_class)
        np.savez(file_name, x_train=x_train_in, x_test=x_test_in, y_train=y_train_in, y_test=y_test_in,
                 dim_cont=dim_cont, ood_test=x_ood)
        print("===================================================")
    print("\n---------------------------------------------------------------------\n")
    return


for data in ['adult', 'compas', 'gmsc', 'heloc']:
    prepare_data_carla_all_in_dist(data)
    prepare_data_carla_ood_class(data)

for data in ['cover', 'jannis', 'dilbert']:
    prepare_data_cross_validation_all_in_dist(data)
    prepare_data_cross_validation_ood_class(data)
