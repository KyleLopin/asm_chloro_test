# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.covariance import MinCovDet
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, GroupShuffleSplit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.utils import extmath

# local files
import get_data
from global_classes import CVScores


def pca(x, n_components=2, robust=False):
    """
        Perform Principal Component Analysis (PCA) on the given dataset.

        Args:
            x (pd.DataFrame or np.ndarray): The input data for PCA, with observations as
            rows and features as columns.
            n_components (int, optional): The number of principal components to compute.
            Defaults to 2.
            robust (bool, optional): If True, use Robust PCA with Minimum Covariance Determinant.
            Defaults to False.

        Returns:
            np.ndarray: The PCA-transformed data with the specified number of components.

        References:
            - PCA explanation: https://nirpyresearch.com/pca-kernel-pca-explained/
            - Robust PCA: https://nirpyresearch.com/robust-pca/
    """
    # Preprocessing - Standard Scaler
    x_std = StandardScaler().fit_transform(x)

    # Calculate covariance matrix
    if robust:
        robust_cov = MinCovDet().fit(StandardScaler(with_std=False).fit_transform(x))
        cov_matrix = robust_cov.covariance_
    else:
        cov_matrix = np.cov(x_std.T)

    # Get eigenvalues and eigenvectors
    _, eig_vecs = np.linalg.eigh(cov_matrix)

    # flip eigenvectors' sign to enforce deterministic output
    eig_vecs, _ = extmath.svd_flip(eig_vecs, np.empty_like(eig_vecs).T)

    # Concatenate the eigenvectors corresponding to the highest n_components eigenvalues
    matrix_w = np.column_stack([eig_vecs[:, -i] for i in range(1, n_components + 1)])

    # Get the PCA reduced data
    x_pca = x_std.dot(matrix_w)

    return x_pca


def view_score_vs_n_components(x: pd.DataFrame, y: pd.Series,
                               groups: pd.Series = None,
                               regr: RegressorMixin = LinearRegression(),
                               cv=None, model_type: str = "PCA",
                               max_comps: int = 20):
    """
        Visualize the performance of a regression model as a function of the number
        of principal components used in PCA.

        This function applies PCA or Kernel PCA to the input data, fits a regression
        model to the transformed data, and evaluates its performance using cross-validation.
        The results are plotted to show the relationship between the number of principal
        components and the regression model's performance.

        Args:
            x (pd.DataFrame): The input data for PCA, with observations as rows and
                features as columns.
            y (pd.Series): The target variable.
            groups (pd.Series, optional): Group labels for the samples used while
                splitting the dataset into train/test set. If None, no grouping is applied.
                Defaults to None.
            regr (RegressorMixin, optional): The regression model to evaluate for PCA decomposition.
                For a 'PLS' model type, this is ignored.  Defaults to LinearRegression().
            cv (cross-validation generator, optional): A cross-validation generator.
                 Defaults to None which uses GroupShuffleSplit with test_size=0.20 and n_splits=300
            model_type (str, optional): The type of PCA to use. Can be 'PCA' for standard PCA,
                'Kernel' for Kernel PCA, or 'PLS' for Partial Least Squared. If 'PCA' or
                'KernelPCA' are used, the regression specified in 'regr' is used.
                For 'PLS', PLSRegression is used and the regr variable is ignored.
                Defaults to 'PCA'.
            max_comps (int, optional): The maximum number of components to test.
                Defaults to 20, but will shrink to be the maximum components available.

        Raises:
            ValueError: If 'pca_type' is not 'PCA' or 'Kernel'.

        Returns:
            None: The function plots the cross-validation scores for the regression model.

        """
    cv_scores = CVScores()
    if not cv:
        cv = GroupShuffleSplit(test_size=0.20, n_splits=300)

    max_components = min(max_comps, x.shape[1]+1)
    for i in range(1, max_components+1):
        if model_type == "PCA":
            x_pca = PCA(n_components=i).fit_transform(x)
        elif model_type == "KernelPCA":
            x_pca = KernelPCA(n_components=i, kernel='rbf').fit_transform(x)
        elif model_type == "PLS":
            x_pca = x
            regr = PLSRegression(n_components=i)
        else:
            raise ValueError(f"'pca_type' needs to be 'PCA' or 'KernelPCA', "
                             f"'{model_type}' is not valid")

        # x_pca = _pca(n_components=i).fit_transform(x)
        scores = cross_validate(regr, x_pca, y, groups=groups,
                                scoring='r2', cv=cv,
                                return_train_score=True)
        cv_scores.add_scores(i, scores)
    cv_scores.plot()
    plt.ylim([0.8, 1.0])
    plt.show()


def view_decomposed_components(x: pd.DataFrame,
                               y: pd.Series = None,
                               decompose_type: str = "PCA",
                               n_comps: int = 20,
                               scale: bool = False):
    """
    Visualize the decomposed components of a dataset using PCA or PLS.

    This function scales the data if specified, applies the selected decomposition method
    (PCA or PLS), and plots the first `n_comps` components. The components are displayed
    in pairs, with each subplot showing two components.

    Args:
        x (pd.DataFrame): The input data with observations as rows and features as columns.
        y (pd.Series, optional): The target variable required for PLS decomposition. Defaults to None.
        decompose_type (str, optional): The type of decomposition to use. Can be 'PCA' or 'PLS'.
            Defaults to "PCA".
        n_comps (int, optional): The number of components to compute and plot. Defaults to 20.
        scale (bool, optional): If True, the data is scaled before decomposition. Defaults to False.

    Raises:
        ValueError: If `decompose_type` is not 'PCA' or 'PLS', or if `decompose_type` is 'PLS' and `y` is None.

    Returns:
        None: The function displays a series of plots showing the decomposed components.
    """
    if scale:
        columns_hold = x.columns
        x = StandardScaler().fit_transform(x)
        x = pd.DataFrame(x, columns=columns_hold)
    n_comps = min(n_comps, x.shape[1])
    n_graphs = round(n_comps / 2)
    if decompose_type == "PCA":
        _pca = PCA(n_components=n_comps)
        _pca.fit(x)
        x_components = _pca.components_.T
    elif decompose_type == "PLS":
        # not 100% sure this is right
        if y is None:
            raise ValueError("If using the PLS decomposition, you need to supply a y")
        pls = PLSRegression(n_components=n_comps)
        pls.fit(x, y)
        x_components = pls.x_weights_
    else:
        raise ValueError(f"'decompose_type' needs to be 'PCA' or 'PLS': "
                         f"'{decompose_type}' is not valid")
    _, axs = plt.subplots(nrows=n_graphs, sharex=True)
    axs = list(axs)
    print(x_components.shape)
    for i in range(n_graphs):
        axs[i].plot(x.columns, x_components[:, i*2:i*2+2])


if __name__ == '__main__':
    _x, _y, _groups = get_data.get_x_y(sensor="as7262", int_time=100,
                                       led_current="12.5 mA", leaf="mango",
                                       measurement_type="reflectance",
                                       send_leaf_numbers=True)
    _y = _y['Avg Total Chlorophyll (µg/cm2)']
    # poly_features = PolynomialFeatures(2)
    # poly_x = poly_features.fit_transform(_x)
    # _x = pd.DataFrame(poly_x, columns=poly_features.get_feature_names_out(_x.columns))
    # view_score_vs_n_components(_x, _y, _groups, model_type="PLS")
    view_decomposed_components(_x, _y, n_comps=3,
                               decompose_type="PCA",
                               scale=True)
    plt.xticks(rotation=50)
    plt.tight_layout()
    plt.show()
