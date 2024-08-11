# OCT

The official implementation of the paper "Out-of-Distribution Aware Classification for Tabular Data", presented in CIKM 2024.


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Datasets
The Adult, Compas, GMSC, and Heloc datasets will be automatically downloaded from the CARLA library on the first run, requiring no additional actions. For the Cover, Dilbert, and Jannis datasets, manual downloads are necessary from the following sources: [Cover Dataset](https://archive.ics.uci.edu/ml/datasets/covertype) and [Jannis and Dilbert Datasets](https://automl.chalearn.org/data). Once obtained, place the downloaded files in the designated `datasets/` folder.

## OOD-Aware Classification Experiment

To conduct the OOD-aware classification experiment, start by executing the following script to preprocess data, apply five-fold cross validation (for Cover, Dilbert, and Jannis datasets, which do not have public train-test), and create OOD test sets for both evaluation settings I and II. The output will be saved in the `saves_data/` folder.

```
python src/script_prepare_data.py
```

To train and evaluate our proposed OCT model, run:

```
python src/script_clf_oct.py --dataset <dataset_name> --setting <ood_classD|all_in_dist> [--fold <int>] [--sigma <float>] [--p <float>] [--n <int>] [--save]
```

**Parameters:**

- **`--dataset <dataset_name>`**: Specifies the dataset to be used.

- **`--setting <ood_classD|all_in_dist>`**: Defines the evaluation setting.
  - **`ood_classD`**: Used for Test Setting I, where one of the classes is treated as OOD. Replace `D` with the class number to specify which class is considered OOD (e.g., `ood_class0` indicates class 0 is considered OOD).
  - **`all_in_dist`**: Used for Test Setting II, where all classes are considered in-distribution, and synthesized OOD sets are used for evaluation.

- **`--fold <int>`** *(optional)*: Specifies the fold number for Cover, Dilbert, and Jannis datasets, for which cross-validation is applied. This argument should not be set for Adult, Compas, GMSC, and Heloc datasets, where cross-validation is not applied.

- **`[--sigma <float>]`** *(optional)*: Specifies the sigma value for applying Gaussian noise to continuous features. Default is `0.01`.

- **`[--p <float>]`** *(optional)*: Sets the perturbation probability for categorical features. Default is `0.1`.

- **`[--n <int>]`** *(optional)*: Determines the number of OOD samples, expressed as a multiplier (`n * size of in-distribution data`). Default is `2`.

- **`[--save]`** *(optional)*: Saves the trained model to the `saves_model/` folder if specified.

For more information on other parameters, please refer to the Python file.

**Examples:**
- To evaluate OCT on the Adult dataset using Test Setting I (ood_class) with class 0 considered as OOD:
  ```
  python src/script_clf_oct.py --dataset adult --setting ood_class0 --sigma 0.01 --p 0.1 --n 2
  ```
- To evaluate OCT on the Cover dataset using Test Setting II (all_in_dist) on fold 0:
  ```
  python src/script_clf_oct.py --dataset cover --setting all_in_dist --fold 0 --sigma 0.01 --p 0.1 --n 2
  ```


To train and evaluate the Original model, run:
```
python src/script_clf_original.py --dataset <dataset_name> --setting <ood_classD|all_in_dist> [--fold <int>]
```


## Counterfactual Explanations Experiment

For the counterfactual experiment, first run the code for our OCT model, as well as the Original and DK models, using the configuration `--setting all_in_dist` and `--save` to store the trained classification models in the `saves_model/` folder:

```
python src/script_clf_oct.py --dataset <dataset_name> -setting all_in_dist --save
python src/script_clf_original.py --dataset <dataset_name> -setting all_in_dist --save
python src/script_clf_dk.py --dataset <dataset_name> -setting all_in_dist --save
```

Then, run the specific counterfactual method script:
```
python src/script_cf_xxx.py --dataset <dataset_name>
```
`xxx` denote the name of the counterfactual algorithm: gd, gs, cchvae, revise. For example, to run the experiment with the gd counterfactual algorithm, use `script_cf_gd.py`.



## Libraries

For the counterfactual experiment, we utilized the [CARLA](https://github.com/carla-recourse/CARLA) and [DiCE-ML](https://github.com/interpretml/DiCE) libraries. The source code for these libraries is already included in the src folder, so no additional setup is required. We made minor modifications to the original code, primarily altering the stopping condition in the counterfactual search from `prediction=[0, 1]` to `prediction=[0,1,0]` to account for the additional OOD class in the OCT model. Detailed information about these modifications can be found in `src/carla/modified_files.txt` for the CARLA library and `src/dice_ml/modified_files.txt` for the DiCE-ML library.
