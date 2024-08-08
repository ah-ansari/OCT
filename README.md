# OCT
The official implementation of the paper "Out-of-Distribution Aware Classification for Tabular Data" CIKM 2024.


# Out-Of-Distribution Aware Classification for Tabular Data

We've provided the source code for the implementations discussed in the paper, and the code for all baseline models. The experiments were performed exclusively on the CPU, without utilizing any GPU.


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Datasets
The Adult, Compas, GMSC, and Heloc datasets will be automatically downloaded from the CARLA library on the first run, requiring no additional actions. For the Cover, Dilbert, and Jannis datasets, manual downloads are necessary from the following sources: [Cover Dataset](https://archive.ics.uci.edu/ml/datasets/covertype) and [Jannis and Dilbert Datasets](https://automl.chalearn.org/data). Once obtained, place the downloaded files in the designated `datasets/` folder.

## OOD-Aware Classification Experiment

To conduct the OOD-aware classification experiment, start by executing the following command to preprocess data, apply five-fold cross validation (for Cover and Jannis, which do not have public train-test), and create OOD testing points. The output will be saved in the `saves_data/` folder.

```
python src/clf_prepare_data.py
```


To train and evaluate **our proposed model (OCT)**, run:

```
python src/clf_oct.py --dataset <dataset_name> --sigma 0.01 --p 0.1 --n 2
```
The model requires parameters sigma, p, and n. If not specified, default values will be employed.


To train and evaluate the baselines, run:
```
python src/clf_xxx.py --dataset <dataset_name>
```
`xxx` indicate the name of the baseline (e.g., `clf_exposure.py` is the implementation of Exposure baseline). Please note that the code for the Original, Pipeline, Energy, and ReAct methods resides within the `clf_original.py` file.


## Counterfactual Explanations Experiment

For the counterfactual experiment, first run the code for our model (OCT) and Original model with `--save` option to save the classification models in the `saves_model/` folder:

```
python src/clf_oct.py --dataset <dataset_name> --sigma 0.001 --p 0.1 --n 2 --save
python src/clf_original.py --dataset <dataset_name> --save
```

Then, run the counterfactual method experiment file:
```
python src/cf_xxx.py --dataset <dataset_name>
```
`xxx` denote the name of the counterfactual algorithm: gd, gs, cchvae, revise (e.g., `cf_gd.py` is the file to run the experiment for gd counterfactual algorithm).



## Libraries

For the counterfactual experiment, we have used the [CARLA](https://github.com/carla-recourse/CARLA) and [DiCE-ML](https://github.com/interpretml/DiCE) counterfactual libraries. The source code of these libraries are already included in the `src` folder. We have applied slight modifications to the original code of these libraries. Details of these modifications are avaible in `src/carla/modified_files.txt` for the CARLA library and `src/dice_ml/modified_files.txt` for the DiCE-ML library.
