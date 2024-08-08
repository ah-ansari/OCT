import numpy as np
from sklearn.ensemble import IsolationForest


def sample_neighbor(x_sample, index_start_cat, sigma, p):
    """
    Sample a point in the neighborhood of x.

    This function samples a point in the neighborhood of the input point `x_sample`, by adding Gaussian noise to continuous features and perturbing categorical features with a certain probability.

    Parameters:
    x_sample (array-like): The sample for which a random point in its neighborhood is created.
    index_start_cat (int): The index where categorical features start in the data record. The dataset is assumed to have continuous features first, followed by categorical features.
    sigma (float): The standard deviation of the Gaussian noise added to continuous features.
    p (float): The probability of perturbing each categorical feature.

    Returns:
    array-like: A random sample in the neighborhood of `x_sample`, defined by `sigma` for continuous features and `p` for categorical features.
    """
    s = x_sample.copy()

    # sample continuous features
    for i in range(index_start_cat):
        # add Gaussian noise with mean 0 and the given standard deviation
        s[i] += np.random.normal(0, sigma)

    # sample categorical features
    for i in range(index_start_cat, x_sample.size):
        if np.random.uniform(0, 1) < p:
            # randomly select a value from [0, 1], as our datasets only have binary features for categorical variables.
            s[i] = np.random.choice(2)

    s = np.clip(s, a_min=0, a_max=1)

    return s


def create_ood(x, ood_oracle, index_start_cat, sigma, p, n):
    """
    Generate boundary OOD samples.

    This function generates `n` OOD samples that are close to the in-distribution points provided in `x`. The OOD samples are created by perturbing the continuous and categorical features of `x`.

    Parameters:
    x (array-like): The in-distribution data points close to which OOD samples are created.
    ood_oracle (callable): The OOD detection oracle function. It should return -1 for OOD samples and 1 for in-distribution samples.
    index_start_cat (int): The index where categorical features start in the data record. The dataset is assumed to have continuous features first, followed by categorical features.
    sigma (float): The standard deviation of the Gaussian noise added to continuous features.
    p (float): The probability of perturbing each categorical feature.
    n (int): The number of OOD samples to be created.

    Returns:
    array-like: A set of `n` OOD points.
    """
    ood_points = []

    # select n points from x randomly
    xx_index = np.random.choice(x.shape[0], n)
    xx = x.copy()[xx_index, :]

    # create one OOD sample for each point in xx
    alpha = 1
    n_total = 0
    while xx.shape[0] != 0:
        s = []
        for x_i in range(xx.shape[0]):
            s.append(sample_neighbor(xx[x_i], index_start_cat, alpha * sigma, alpha * p))
        s = np.array(s)

        pred = ood_oracle(s)
        s_ood = s[pred == -1]
        # reduce xx to only those for which OOD is not created
        xx = xx[pred == 1]

        alpha *= 2
        n_total += s.shape[0]

        ood_points.append(s_ood)

    ood_points = np.concatenate(ood_points, axis=0)

    return ood_points


def sample_ood_uniform(ood_oracle, n, dim_cont, dim_cat):
    """
    Generate uniform OOD samples.

    This function generates `n` OOD samples uniformly at random from the input feature space. It creates random samples and uses the OOD oracle to filter out in-distribution samples, ensuring the final set contains only OOD points.

    Parameters:
    ood_oracle (callable): The OOD detection oracle function. It should return -1 for OOD samples and 1 for in-distribution samples.
    n (int): The number of OOD samples to be created.
    dim_cont (int): The number of continuous features in the data record. The dataset is assumed to have continuous features first, followed by categorical features.
    dim_cat (int): The number of categorical features in the data record.

    Returns:
    array-like: A set of `n` OOD points.
    """
    ood_points = []
    n_ood = 0
    n_total = 0

    while n_ood < n:
        s_cont = np.random.uniform(0, 1.0000001, size=(n, dim_cont))
        s_cat = np.random.choice(2, size=(n, dim_cat))
        s = np.concatenate((s_cont, s_cat), axis=1)

        pred = ood_oracle(s)
        ood_points.append(s[pred == -1])
        n_ood += pred[pred == -1].size
        n_total += n

    ood_points = np.concatenate(ood_points, axis=0)
    if ood_points.shape[0] > n:
        selecting_index = np.random.choice(range(ood_points.shape[0]), size=n, replace=False)
        ood_points = ood_points[selecting_index, :]

    return ood_points


def create_training_data(x, y, ood_oracle, index_start_cat, sigma, p, n, class_ood):
    """
    Create the training data for the OOD-Aware model.

    This function creates the data used for training our OOD-aware model by augmenting the in-distribution data `x` (features) and `y` (labels) with boundary OOD samples.

    Parameters:
    x (array-like): In-distribution training data, features.
    y (array-like): In-distribution training data, labels.
    ood_oracle (callable): The OOD detection oracle function. It should return -1 for OOD samples and 1 for in-distribution samples.
    index_start_cat (int): The index where categorical features start in the data record. The dataset is assumed to have continuous features first, followed by categorical features.
    sigma (float): The standard deviation of the Gaussian noise added to continuous features.
    p (float): The probability of perturbing each categorical feature.
    n (int): The number of OOD samples to be created.
    class_ood (int): The class label to be assigned to OOD samples.

    Returns:
    tuple: A tuple containing the augmented training features and labels.
        - array-like: The augmented training features.
        - array-like: The augmented training labels.
    """
    ood_points = create_ood(x=x, ood_oracle=ood_oracle, index_start_cat=index_start_cat, sigma=sigma, p=p, n=n)

    x_oct = np.concatenate((x, ood_points), axis=0)
    y_oct = np.concatenate((y, np.full(ood_points.shape[0], class_ood)), axis=0)

    return x_oct, y_oct


def create_ood_oracle(x):
    """
    Create an OOD detection oracle using Isolation Forest.

    This function trains an Isolation Forest model on the provided in-distribution data `x`.

    Parameters:
    x (array-like): The in-distribution data used to train the OOD oracle.

    Returns:
    IsolationForest: The trained Isolation Forest model that acts as the OOD oracle.
    """
    ood_oracle = IsolationForest(random_state=0, contamination=0.02).fit(x)

    return ood_oracle
