# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# installed libraries
import matplotlib as mpl
import matplotlib.pyplot as plt

# local files
import get_data


def fit_model(sensor: str = "as7262", leaf: str = "banana",
              leds: list[str] = [], currents: list[str] = ["12.5 mA"],
              integration_times: list[int] = [100],
              measurement_type: str = "reflectance",
              ax: plt.Axes = None, take_mean=False):
    data = get_data.get_data(sensor=sensor, leaf=leaf,
                             measurement_type=measurement_type,
                             mean=take_mean)
    if leds:
        data = get_data.get_data_slices(df=data, selected_column="led",
                                        values=leds)
    if integration_times:
        data = get_data.get_data_slices(df=data, selected_column="integration time",
                                        values=integration_times)
    if currents:
        data = get_data.get_data_slices(df=data, selected_column="led current",
                                        values=currents)
    print(data)


if __name__ == '__main__':
    fit_model(sensor="as7262")
