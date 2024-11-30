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

# installed libraries
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin  # for type-hinting
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression

# local files
import get_data
import visualize_raw_data
ssl._create_default_https_context = ssl._create_unverified_context
# figure formatting constants
COLOR_BAR_AXIS = [.90, .1, 0.02, 0.8]

SENSORS = ["as7262", "as7263", "as7265x"]
ALL_LEAVES = ["banana", "jasmine", "mango", "rice", "sugarcane"]


def remove_outliers_from_model(regressor: RegressorMixin,
                               x: pd.DataFrame,
                               y: pd.Series,
                               groups: pd.Series,
                               cutoff: float = 3.0,
                               remove_recursive: bool = False,
                               verbose: bool = False,
                               show_hist: bool = False,
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
        show_hist (bool, optional): If True, plot the histogram of model residues

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
        plt.hist(residues)
        plt.show()

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
            print(f"residue std:  {residues.std():0.3f} with {outliers.shape[0]} outliers")
            print(f"scores went from {initial_score:0.3f} to {regressor.score(x, y):0.3f}")
        # return if not recursive or there are no outliers
        if not remove_recursive or outliers.shape[0] == 0:
            return x, y, groups


def mahalanobis_outlier_removal(x: pd.DataFrame,
                                use_robust: bool = True,
                                display_hist: bool = False,
                                cutoff_limit: float = 4.0,
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
        plt.hist(shifted_mahal, bins=100)
        plt.axvline(cutoff, ls='--')
        print(f"outliers removed = {x.shape[0]-x[data_mask].shape[0]}")
        plt.show()
    print(f"outliers removed = {x.shape[0] - x[data_mask].shape[0]}")
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

    # else:  i np.array just pass
    #     raise TypeError(f"Needs to use a pandas Series or DataFrame, {type(x)} is not valid")
    for index in x.index:
        # get indexes with the same leaf number minus the current index
        other_leaf_indexes = groups.index[groups == groups[index]].drop(index)
        # calculate the difference between the current index and the mean of the
        # spectrum for the other leaves
        residue = x.loc[index] - x.loc[other_leaf_indexes].mean()
        new_df.loc[index] = residue
    return new_df


def remove_outliers_from_residues(x, y, groups,
                                  regr_model: RegressorMixin=LinearRegression(),
                                  verbose: bool=False):
    if verbose:
        initial_score = regr_model.fit(x, y).score(x, y)
        initial_samples = x.shape[0]
    residues = calculate_residues(x, groups)
    data_mask = mahalanobis_outlier_removal(residues)
    x = x[data_mask]
    # y = y[data_mask]
    # groups = groups[data_mask]
    if verbose:
        print(f"outliers removed = {initial_samples - x.shape[0]}")
        print(f"scores went from {initial_score:0.2f} to {regr_model.fit(x, y).score(x, y):0.2f}")
    # return x, y, groups
    return data_mask


def vis_outlier():

    for leaf in ["mango", "sugarcane"]:
        for sensor in SENSORS:
            led = "White LED"
            if sensor == "as7265x":
                led = "b'White IR'"

            x, y, groups = get_data.get_x_y(sensor=sensor, int_time=50, led=led,
                                            led_current="12.5 mA", leaf=leaf,
                                            measurement_type="reflectance",
                                            send_leaf_numbers=True)

            # x = preprocessors.snv(x)
            # index = x.index
            # x = StandardScaler().fit_transform(x)
            # x = pd.DataFrame(x, index=index)
            # print(type(x))
            residue_spectra = calculate_residues(x, groups)
            data_mask = mahalanobis_outlier_removal(residue_spectra)
            # data_mask = mahalanobis_outlier_removal(x)

            print(pd.Series(data_mask).value_counts())
            groups = groups[data_mask]
            print('leaves left =', groups.nunique())
            print(sensor, leaf)
            plt.figure()
            for idx in range(x.shape[0]):
                color = 'green' if data_mask[idx] else 'red'
                alpha = .1 if data_mask[idx] else 1.0
                plt.plot(x.iloc[idx, :], color=color, alpha=alpha)
                # plt.plot(x[idx, :], color=color, alpha=alpha)

            plt.show()






def make_manuscript_figure(leaf: str = "mango"):
    """
    Create a figure comparing outliers detected by Mahalanobis distance and Mahalanobis distance on the residues
    across multiple sensors.

    Args:
        leaf (str): name of the string of the data to get

    """

    color_map, map_norm = visualize_raw_data.make_color_map(0, 100)

    ###### INNER FUNCTION
    def make_outlier_plot(x: pd.DataFrame, y: pd.Series,
                          use_residue: bool, ax: plt.Axes,
                          groups: pd.Series | None = None):
        columns = x.columns
        if use_residue:
            residues = calculate_residues(x, groups)
            mask = mahalanobis_outlier_removal(residues)
        else:
            mask = mahalanobis_outlier_removal(x)
        wavelengths = x.columns
        x_wavelengths = [int(wavelength.split()[0]) for wavelength in wavelengths]

        for idx in range(x.shape[0]):
            if mask[idx]:
                color, alpha, z = color_map(map_norm(y.iloc[idx])), 0.3, 1
            else:
                # Determine the final color using the colormap based on y
                color, alpha, z = 'red', 1.0, 2

            ax.plot(x_wavelengths, x.iloc[idx, :],
                            color=color, alpha=alpha, zorder=z)
    ####### END INNER FUNCTION

    # Create a figure
    fig = plt.figure(figsize=(6, 8))

    # Create GridSpec with 4 rows and 2 columns
    gs = GridSpec(5, 2, figure=fig, height_ratios=[1, 1, 0.2, 1, 1])

    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[3, :])  # Merge row 3, columns 0 and 1
    ax6 = fig.add_subplot(gs[4, :])  # Merge row 3, columns 0 and 1

    for sensor in SENSORS:
        led = "White LED"
        if sensor == "as7265x":
            led = "b'White IR'"
        pls = PLSRegression(n_components=6)

        x, y, groups = get_data.get_x_y(sensor=sensor, int_time=50, led=led,
                                        led_current="12.5 mA", leaf=leaf,
                                        measurement_type="reflectance",
                                        send_leaf_numbers=True)
        y = y["Avg Total Chlorophyll (µg/cm2)"]
        wavelengths = x.columns
        x_wavelengths = [int(wavelength.split()[0]) for wavelength in wavelengths]

        if sensor == "as7262":  # DRY yourself, its late
            # graph outliers based on Mahalanobis distances only
            make_outlier_plot(x, y, use_residue=False, ax=ax1)
            ax1.set_xticklabels([])
            make_outlier_plot(x, y, use_residue=True, ax=ax2, groups=groups)
            ax2.set_xticks(ticks=x_wavelengths, labels=wavelengths, rotation=45)
        elif sensor == 'as7263':
            make_outlier_plot(x, y, use_residue=False, ax=ax3)
            ax3.set_xticklabels([])
            make_outlier_plot(x, y, use_residue=True, ax=ax4, groups=groups)
            ax4.set_xticks(ticks=x_wavelengths, labels=wavelengths, rotation=45)
        elif sensor == "as7265x":
            make_outlier_plot(x, y, use_residue=False, ax=ax5)
            ax5.set_xticklabels([])
            make_outlier_plot(x, y, use_residue=True, ax=ax6, groups=groups)
            ax6.set_xticks(ticks=x_wavelengths, labels=wavelengths, rotation=60)

    # label to each axis with a-f
    for ax, letter in zip([ax1, ax2, ax3, ax4, ax5, ax6],
                          ['a', 'b', 'c', 'd', 'e', 'f']):
        pass

    ax5.plot([], [], color='red', label="Outlier")
    ax5.legend(loc="upper left")
    # add color bar at end
    color_map = mpl.cm.ScalarMappable(norm=map_norm, cmap=color_map)
    color_bar_axis = fig.add_axes(COLOR_BAR_AXIS)
    color_bar = fig.colorbar(color_map, cax=color_bar_axis, orientation="vertical",
                                fraction=0.08)
    # Adjust the label padding (distance from the color bar)
    color_bar.set_label(r'Total Chlorophyll ($\mu$g/cm$^2$)',
                        labelpad=-1)
    fig.subplots_adjust(left=0.1, wspace=0.22, right=0.85, hspace=0.2)

    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    make_manuscript_figure("mango")
    # vis_outlier()
    # for sensor in SENSORS:
    #     for leaf in ALL_LEAVES:
    #         led = "White LED"
    #         if sensor == "as7265x":
    #             led = "b'White IR'"
    #         _x, _y, _groups = get_data.get_x_y(
    #             sensor=sensor, int_time=50, led_current="12.5 mA", leaf=leaf,
    #             measurement_type="reflectance", send_leaf_numbers=True)
    #         # nn(_x)
    #         # _x = preprocessors.snv(_x)
    #         residue_spectra = calculate_residues(_x, _groups)
    #         mask = mahalanobis_outlier_removal(residue_spectra)
