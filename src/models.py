"""
This file implements functions and classes for classification algorithms, including
Deep Neural Networks (DNNs) and decision tree-based models using XGBoost.

The two key functions are:
* `make_model_dnn`: Trains a 3-layer DNN with specified layer sizes on a given dataset,
  featuring support for early stopping and optional class weighting to handle imbalanced datasets.

* `make_model_tree`: Trains a decision tree-based model using XGBoost, with options
  to apply class weights for handling imbalanced datasets.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import xgboost
from typing import List, Callable, Tuple, Optional

import tools


class DNNModel(nn.Module):
    def __init__(self, n_inputs: int, n_classes: int, layers: List[int]):
        super(DNNModel, self).__init__()

        if len(layers) != 3:
            print("Err!! number of NN layers should be 3.")
            exit(1)

        print("Neural Network model layers: ", layers)

        self.lin1 = nn.Linear(n_inputs, layers[0])
        self.lin2 = nn.Linear(layers[0], layers[1])
        self.lin3 = nn.Linear(layers[1], layers[2])
        self.lin4 = nn.Linear(layers[2], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin4(x)
        return x


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    return


def create_predictor(model: nn.Module, device: torch.device) -> Callable[[np.ndarray], np.ndarray]:
    def predict(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor_x = torch.tensor(x).float().to(device)
            output = model(tensor_x)
            output = output.data.cpu().numpy()

        pred = np.argmax(output, axis=1)
        return pred

    return predict


def make_model_dnn(x: np.ndarray, y: np.ndarray, batch_size: int = 64, lr: float = 0.001,
                   layers: Optional[List[int]] = None, extra_ood_class: bool = False, apply_weight: bool = True,
                   esp: int = 20, device: torch.device = torch.device('cpu'), verbose: bool = False) \
        -> Tuple[nn.Module, Callable[[np.ndarray], np.ndarray]]:
    if layers is None:
        layers = [32, 16, 8]
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

    # class weights: set the weights according to the distribution of class labels, and also it is considered that
    # the sample has an extra OOD class or not
    if extra_ood_class is True:
        print("The model has OOD as extra class")
        y_w = y[y != np.max(y)]
        c_w = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_w), y=y_w)
        c_w /= np.sum(c_w)
        c_w = np.append(c_w, 1)
        c_w = torch.Tensor(c_w).to(device)
    else:
        c_w = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
        c_w /= np.sum(c_w)
        c_w = torch.Tensor(c_w).to(device)

    model = DNNModel(n_inputs, n_classes, layers)
    model.apply(init_weights)
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

    predictor = create_predictor(model, device)
    print("\n* train")
    tools.evaluate_clf_general(predictor, x_train, y_train)
    print("\n* validation")
    tools.evaluate_clf_general(predictor, x_val, y_val)

    return model, predictor


def make_model_tree(x: np.ndarray, y: np.ndarray, n_classes: int, max_depth: int, extra_ood_class: bool = False,
                    apply_weight: bool = False) -> xgboost.XGBClassifier:
    # split the validation data
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)

    # class weights
    if extra_ood_class is True:
        print("robust")
        y_w = y[y != np.max(y)]
        c_w = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_w), y=y_w)
        c_w /= np.sum(c_w)
        c_w = np.append(c_w, 1)
    else:
        c_w = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
        c_w /= np.sum(c_w)

    # define the sample weights
    s_w = np.empty(y_train.shape[0], dtype='float')
    for i, c in enumerate(y_train):
        s_w[i] = c_w[int(c)]

    if extra_ood_class is True:
        n_classes += 1

    model = xgboost.XGBClassifier(objective='multi:softprob', num_class=n_classes, max_depth=max_depth)

    if apply_weight:
        print("model weighted")
        print("weights", c_w)
        model.fit(x_train, y_train, sample_weight=s_w)
    else:
        model.fit(x_train, y_train)

    print("\n* train")
    tools.evaluate_clf_general(model.predict, x_train, y_train)
    print("\n* validation")
    tools.evaluate_clf_general(model.predict, x_val, y_val)

    return model
