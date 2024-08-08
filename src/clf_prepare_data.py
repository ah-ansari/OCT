import numpy as np
from sklearn.model_selection import KFold
from sklearn import preprocessing
import warnings

import oct
import tools
import cf

warnings.filterwarnings('ignore')

save_folder = 'saves_data/'


def prepare_data_cv(dataset_name):
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
        file_name = save_folder + dataset_name + "_" + str(i)
        np.savez(file_name, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, dim_cont=dim_cont,
                 ood_test1=ood_test1, ood_test2=ood_test2, ood_test3=ood_test3, ood_test4=ood_test4)

    print("\n---------------------------------------------------------------------\n")
    return


def prepare_data_carla(dataset_name):
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
    file_name = save_folder + dataset_name
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
                 dim_cont=dim_cont, ood_test1=x_ood, ood_test2=x_ood, ood_test3=x_ood, ood_test4=x_ood)
        print("===================================================")
    print("\n---------------------------------------------------------------------\n")
    return


def prepare_data_cv_ood_class(dataset_name):
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
                     dim_cont=dim_cont, ood_test1=x_ood, ood_test2=x_ood, ood_test3=x_ood, ood_test4=x_ood)

            print("\n----------------------------------\n")
        print("===================================================")
    print("\n---------------------------------------------------------------------\n")
    return


for data in ['adult', 'compas', 'gmsc', 'heloc']:
    prepare_data_carla(data)
    prepare_data_carla_ood_class(data)

for data in ['cover', 'jannis', 'dilbert', 'fabert']:
    prepare_data_cv(data)
    prepare_data_cv_ood_class(data)
