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


def view_pca_n_components(x: pd.DataFrame, y: pd.Series,
                          groups: pd.Series = None,
                          regr: RegressorMixin = LinearRegression(),
                          cv=None, pca_type: str = "PCA"):
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
            regr (RegressorMixin, optional): The regression model to evaluate.
                Defaults to LinearRegression().
            cv (cross-validation generator, optional): A cross-validation generator.
                 Defaults to None which uses GroupShuffleSplit with test_size=0.20 and n_splits=300
            pca_type (str, optional): The type of PCA to use. Can be 'PCA' for standard PCA or
                'Kernel' for Kernel PCA. Defaults to 'PCA'.

        Raises:
            ValueError: If 'pca_type' is not 'PCA' or 'Kernel'.

        Returns:
            None: The function plots the cross-validation scores for the regression model.

        """
    cv_scores = CVScores()
    if not cv:
        cv = GroupShuffleSplit(test_size=0.20, n_splits=300)

    if pca_type == "PCA":
        _pca = PCA
    elif pca_type == "Kernel":
        _pca = KernelPCA
    else:
        raise ValueError(f"'pca_type' needs to be 'PCA' or 'Kernel', '{pca_type}' is not valid")

    for i in range(1, x.shape[1]+1):
        x_pca = _pca(n_components=i).fit_transform(x)
        scores = cross_validate(regr, x_pca, y, groups=groups,
                                scoring='r2', cv=cv,
                                return_train_score=True)
        cv_scores.add_scores(i, scores)
    cv_scores.plot()
    # plt.ylim([0.8, 1.0])
    plt.show()


if __name__ == '__main__':
    _x, _y, _groups = get_data.get_x_y(sensor="as7262", int_time=100,
                                       led_current="12.5 mA", leaf="mango",
                                       measurement_type="reflectance",
                                       send_leaf_numbers=True)
    _y = _y['Avg Total Chlorophyll (µg/cm2)']
    _x = PolynomialFeatures(3).fit_transform(_x)
    view_pca_n_components(_x, _y, _groups)
