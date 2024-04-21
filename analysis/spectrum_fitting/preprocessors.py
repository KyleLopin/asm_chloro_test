# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"


# installed libraries
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


def polynomial_expansion(x: pd.DataFrame) -> pd.DataFrame:
    new_x = x.copy(deep=True)
    print(new_x)
    new_columns = list(new_x.columns)
    for column in new_x.columns:
        new_x[f"({column})^2"] = x[column]**2
    return new_x


if __name__ == '__main__':
    _x = pd.DataFrame([[1, 2], [3, 4]], columns=["450 nm", "500 nm"])

    _x = polynomial_expansion(_x)
    print('===')
    print(_x)
