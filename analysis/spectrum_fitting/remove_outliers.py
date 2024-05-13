# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Module Name: outlier_removal
Description: This module provides functions for removing outliers from regression
models based on residuals.
Author: Kyle Lopin (Naresuan University) <kylel@nu.ac.th>
Copyright (c) 2023

Functions:
- remove_outliers_from_model: Removes outliers from a regression model based on residuals.
- mahalanobis_outlier_removal: Removes outliers from a DataFrame based on Mahalanobis distance.
- calculate_residues: Calculate residuals for each row in a DataFrame based on group-wise means.
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
# for testing data
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin  # for type-hinting
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, StandardScaler

# local files
import decomposition
import get_data
# noinspection PyUnresolvedReferences
import preprocessors  # used in __main__ for testing


def view_pca_versus_robust_pca(x: pd.DataFrame):
    pca_pipeline = Pipeline([('scalar', StandardScaler(with_std=False)),
                             ('pca', PCA())])
    robust_pipeline = Pipeline([('scalar', RobustScaler()),
                                ('pca', PCA())])
    pca_pipeline.fit(x)
    plt.plot(pca_pipeline["pca"].components_[0], 'b',
             label="PCA")
    robust_cov = MinCovDet().fit(StandardScaler(with_std=False).fit_transform(x))
    cov_matrix = robust_cov.covariance_

    values, vectors = np.linalg.eig(cov_matrix)

    sorted_values = np.argsort(values)[::-1]
    sorted_vectors = vectors[:, sorted_values]
    plt.plot(sorted_vectors[:, 0], 'r',
             label="custom Robust PCA")

    robust_pipeline.fit(x)
    plt.plot(robust_pipeline["pca"].components_[0], 'g',
             label="built in Robust PCA")


def view_pca_linear_regr_vs_robust(x: pd.DataFrame, y: pd.Series):
    # fit a linear regression model to pca decomposed data
    # pca_pipeline = Pipeline([('scalar', StandardScaler(with_std=False)),
    #                          ('pca', PCA()),
    #                          ("linear regression", LinearRegression())])
    regr = LinearRegression
    # regr = LassoCV
    # pca_pipeline.fit(x, y)
    x = PolynomialFeatures().fit_transform(x)
    x_ss = StandardScaler(with_std=False).fit_transform(x)
    # x_pca = PCA().fit_transform(x_ss)
    x_pca = decomposition.pca(x_ss, n_components=6)
    pca_model = regr().fit(x_pca, y)
    print(pca_model.score(x_pca, y))
    linear_regr = regr().fit(x, y)
    print(linear_regr.score(x, y))
    x_robust_pca = decomposition.robust_pca(x_ss, n_components=6)
    robust_regr = regr().fit(x_robust_pca, y)
    print(robust_regr.score(x_robust_pca, y))


def remove_outliers_from_model(regressor: RegressorMixin,
                               x: pd.DataFrame,
                               y: pd.Series,
                               groups: pd.Series,
                               cutoff: float = 3.0,
                               remove_recursive: bool = False,
                               verbose: bool = False
                               ) -> tuple[pd.DataFrame, pd.Series,
                                          pd.Series]:
    """
    Remove outliers from fitting data to a regression model based on the
    residuals in the individual groups.

    Args:
        regressor (RegressorMixin): The regression model to be used for fitting.
        x (pd.DataFrame): Input DataFrame containing the features.
        y (pd.Series): Series containing the target variable.
        groups (pd.Series): Series defining the groups for each data point.
            It should have the same length and index as 'x' and 'y'.
        cutoff (float, optional): The cutoff value to determine outliers.
            Data points with residuals greater than 'cutoff * std(residuals)'
            are considered outliers. Defaults to 3.0.
        remove_recursive (bool, optional): If True, remove outliers iteratively until
            no more outliers are found. Defaults to False.
        verbose (bool, optional): If True, print verbose output including statistics about outliers.
            Defaults to False.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.Series]: A tuple containing the cleaned DataFrame 'x',
            the corresponding cleaned Series 'y', and the cleaned Series 'groups'
            after removing outliers.
    """
    while True:  # if recursive is True keep running till no more outliers
        # fit the model and get the y predicted values
        regressor = regressor.fit(x, y)
        if verbose:  # save initial score to display later
            initial_score = regressor.score(x, y)
        y_fit = regressor.predict(x)
        # # reset the index so you can calculate the averages
        y_fit = pd.Series(y_fit, index=x.index)
        # pass y_fit to to remove residue function
        residues = calculate_residues(y_fit, groups=groups)

        # OLD WAY BELOW
        # y_fit_avg = y_fit.groupby(by=y_fit.index).mean()
        # # calculate the resides as averages, else the larger values will though off the results
        # residues = (y_fit_avg - y_fit) / y_fit_avg
        cutoff_level = cutoff*residues.std()
        # remove any residues larger than cutoff
        outlier_mask = residues > cutoff_level  # type: pd.Series[bool]
        outlier_mask.index = x.index
        outliers = x.loc[outlier_mask]
        # drop outliers from all 3 groups
        x.drop(index=outliers.index, inplace=True)
        y.drop(index=outliers.index, inplace=True)
        groups.drop(index=outliers.index, inplace=True)
        if verbose:  # print out some basic statistics
            print(f"residue std:  {residues.std():0.2f} with {outliers.shape[0]} outliers")
            print(f"scores went from {initial_score:0.2f} to {regressor.score(x, y):0.2f}")
        # return if not recursive or there are no outliers
        if not remove_recursive or outliers.shape[0] == 0:
            return x, y, groups


def mahalanobis_outlier_removal(x: pd.DataFrame,
                                use_robust: bool = True,
                                display_hist: bool = False,
                                cutoff_limit: float = 3.0,
                                ) -> np.ndarray[bool]:
    """
        Remove outliers from a DataFrame based on Mahalanobis distance.

        Parameters:
            x (pd.DataFrame): Input DataFrame containing numerical data.
            use_robust (bool): Whether to use a robust estimator (default True).
            display_hist (bool): Whether to display histograms (default False).
            If True the function will also show the data in a blocking way.
            cutoff_limit (float): Number of standard deviations to consider outliers (default 3.0).

        Returns:
            pd.DataFrame: DataFrame mask
    """
    if use_robust:  # fit a MCD robust estimator to data
        covariance = MinCovDet()
    else:  # fit a MLE estimator to data
        covariance = EmpiricalCovariance().fit(x)
    covariance.fit(x)
    # calculate the mahalanobis distances
    # and calculate the cubic root to simulate a normal distribution
    mahal_distances = covariance.mahalanobis(x - covariance.location_)**0.33

    # shift the distribution to calculate the distance from the standard deviation easier
    shifted_mahal = mahal_distances - mahal_distances.mean()
    cutoff = cutoff_limit*shifted_mahal.std()
    data_mask = np.where((-cutoff < shifted_mahal) & (shifted_mahal < cutoff), True, False)
    if display_hist:
        plt.hist(mahal_distances, bins=100)
        plt.figure(2)
        plt.hist(shifted_mahal, bins=100)
        plt.axvline(cutoff, ls='--')
        print(f"outliers removed = {x.shape[0]-x[data_mask].shape[0]}")
        plt.show()
    return data_mask


def calculate_residues(x: pd.DataFrame | pd.Series,
                       groups: pd.Series) -> pd.DataFrame:
    """
        Calculate residuals for each row in a DataFrame based on the difference of the individual
        value(s) and the average of all the other rows in that group.

        Args:
            x (pandas.DataFrame): Input DataFrame containing the data.
            groups (pandas.Series): Series defining the groups for each row in the DataFrame.

        Returns:
            pandas.DataFrame: DataFrame containing residuals for each row,
            calculated based on group-wise means.
            The index and columns of the returned DataFrame match those of the input DataFrame 'x'.
    """
    if isinstance(x, pd.DataFrame):
        new_df = pd.DataFrame(columns=x.columns)
    elif isinstance(x, pd.Series):
        new_df = pd.Series()
    else:
        raise TypeError(f"Needs to use a pandas Series or DataFrame, {type(x)} is not valid")
    for index in x.index:
        # get indexes with the same leaf number minus the current index
        other_leaf_indexes = groups.index[groups == groups[index]].drop(index)
        # calculate the difference between the current index and the mean of the
        # spectrum for the other leaves
        residue = x.loc[index] - x.loc[other_leaf_indexes].mean()
        new_df.loc[index] = residue
    return new_df


if __name__ == '__main__':
    _x, _y, _groups = get_data.get_x_y(sensor="as7262", int_time=150,
                                       led_current="12.5 mA", leaf="mango",
                                       measurement_type="reflectance",
                                       send_leaf_numbers=True)
    _y = _y['Avg Total Chlorophyll (µg/cm2)']
    # print(LinearRegression().fit(_x, _y).score(_x, _y))
    # _x, _y, _groups = remove_outliers_from_model(LinearRegression(),
    #                                              _x, _y, _groups,
    #                                              remove_recursive=True,
    #                                              verbose=True)
    # mahalanobis_outlier_removal(_x, display_hist=True)
    # residue_spectra = calculate_residues(_x, _groups)
    # data_mask = mahalanobis_outlier_removal(residue_spectra,
    #                                         display_hist=True)
    # _x = _x[data_mask]
    # _y = _y[data_mask]
    # _groups = _groups[data_mask]
    # print(LinearRegression().fit(_x, _y).score(_x, _y))
    # print(trimmed_x.shape)
    # plt.plot(residue_spectra.T)

    # view_pca_versus_robust_pca(_x)
    # Read data
    # url = 'https://raw.githubusercontent.com/nevernervous78/nirpyresearch/master/data/milk.csv'
    # data = pd.read_csv(url)
    #
    # # Assign spectra to the array X
    # X = data.values[:, 2:].astype('float32')
    # view_pca_versus_robust_pca(x=_x)
    view_pca_linear_regr_vs_robust(_x, _y)

    plt.show()
