# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin  # for type-hinting
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit

# local files
import get_data
# noinspection PyUnresolvedReferences
import preprocessors  # used in __main__ for testing


def remove_outliers_from_model(regressor: RegressorMixin, x, y, group):
    model = regressor.fit(x, y)
    y_predict = model.predict(x)

    residues = y-y_predict
    print(residues.mean(), residues.std())

    n, bins, _ = plt.hist(residues, bins=100)
    plt.figure(2)
    plt.scatter(y, y_predict)

    plt.show()


def recursively_remove_outliers(regressor: LinearRegression,
                                **kwargs):
    x, y, groups = get_data.get_x_y(send_leaf_numbers=True,
                                    **kwargs)
    y = y["Avg Total Chlorophyll (µg/cm2)"]
    print(groups)
    groups = pd.Series(groups)
    regressor = regressor.fit(x, y)
    print(regressor.score(x, y))
    x = x.set_index(groups)

    std, x, y = remove_outlier(regressor, x, y, groups)

    regressor = regressor.fit(x, y)
    y_fit = regressor.predict(x)
    print(regressor.score(x, y))
    y_fit = pd.Series(y_fit, index=x.index)
    y = y.groupby(by=y.index).mean()
    y_fit = y_fit.groupby(by=y_fit.index).mean()
    plt.scatter(y, y_fit)
    print(r2_score(y, y_fit))
    plt.show()


def remove_outlier(regressor: LinearRegression,
                   x, y, groups):
    # Run until break conditions of no residues larger than 2.58 of the residues std
    print("start")
    regressor = regressor.fit(x, y)
    y_fit = regressor.predict(x)
    y_fit = pd.Series(y_fit, index=groups)
    y_fit_avg = y_fit.groupby(by=y_fit.index).mean()
    residues = (y_fit_avg - y_fit) / y_fit_avg
    cutoff = 3*residues.std()
    while residues.max() > cutoff:
        print('aa', residues.max(), cutoff)
        regressor = regressor.fit(x, y)
        y_fit = regressor.predict(x)
        y_fit = pd.Series(y_fit, index=groups)
        y_fit_avg = y_fit.groupby(by=y_fit.index).mean()
        residues = (y_fit_avg-y_fit)/y_fit_avg
        # print(y)
        print("max: ", residues.max())
        residue_max_index = residues.argmax()
        # dropping index of biggest residue,
        # first reset the index so a whole leaf will not be dropped

        x = x.reset_index()
        print(f"shapes: {y.shape}, {x.shape}")
        # print(x[x["Leaf No."] == 46])
        # print(y.iloc[135:138])
        # print(y_fit[135:138], residues.iloc[135:138])
        # print(residue_max_index, 'max value: ', residues.max(),
        #       residues.iloc[residue_max_index], 'std:', residues.std())
        print("removing data leaf", x.iloc[residue_max_index]["Leaf No."], residue_max_index)
        x.drop(index=x.iloc[residue_max_index].name, inplace=True)
        y.drop(y.index[residue_max_index], inplace=True)
        groups.drop(groups.index[residue_max_index], inplace=True)
        # go back to set the leaf number as the index again
        x = x.set_index("Leaf No.")
    # plt.hist(residues, bins=100)
    # plt.show()
    y.index = groups
    return residues.std(), x, y


def mahalanobis_outlier_removal(x: pd.DataFrame,
                                use_robust: bool = True,
                                display_hist: bool = False,
                                cutoff_limit: float = 3.0,
                                ) -> pd.DataFrame:
    """
        Remove outliers from a DataFrame based on Mahalanobis distance.

        Parameters:
            x (pd.DataFrame): Input DataFrame containing numerical data.
            use_robust (bool): Whether to use a robust estimator (default True).
            display_hist (bool): Whether to display histograms (default False).
            If True the function will also show the data in a blocking way.
            cutoff_limit (float): Number of standard deviations to consider outliers (default 3.0).

        Returns:
            pd.DataFrame: DataFrame with outliers removed.
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
    data_mask = np.array([True if -cutoff < i < cutoff else False for i in shifted_mahal])
    if display_hist:
        plt.hist(mahal_distances, bins=100)
        plt.figure(2)
        plt.hist(shifted_mahal, bins=100)
        plt.show()
    return x[data_mask]


def calculate_spectrum_residue(x: pd.DataFrame,
                               groups:pd.Series) -> pd.DataFrame:
    new_df = pd.DataFrame(columns=x.columns)
    for index in x.index:
        other_leaf_indexes = groups.index[groups == groups[index]].drop(index)
        residue = x.loc[index] - x.loc[other_leaf_indexes].mean()
        new_df.loc[index] = residue
    return new_df


if __name__ == '__main__':
    # remove_outliers_from_model(SVR(),
    #                            sensor="as7262", int_time=150,
    #                            led_current="25 mA", leaf="mango",
    #                            measurement_type="raw")
    # pipe = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2),
    #                      LassoCV(max_iter=40000))
    # recursively_remove_outliers(pipe,
    #                             sensor="as7262", int_time=150,
    #                             led_current="12.5 mA", leaf="mango",
    #                             measurement_type="raw"
    #                             )
    _x, _, _groups = get_data.get_x_y(sensor="as7262", int_time=150,
                                     led_current="100 mA", leaf="mango",
                                     measurement_type="reflectance",
                                     send_leaf_numbers=True)
    # mahalanobis_outlier_removal(x, display_hist=True)
    residue_spectrums = calculate_spectrum_residue(_x, _groups)
    trimmed_x = mahalanobis_outlier_removal(residue_spectrums,
                                            display_hist=False)
    print(trimmed_x.shape)
    # plt.plot(residue_spectrums.T)
    # plt.show()
