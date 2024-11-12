# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Make the figure to show how the led drive pin voltage is affected by the
led current sink
"""

__author__ = "Kyle Vitautas Lopin"

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



df = pd.read_excel("led_drv_pins.xlsx")
labels = ["White LED", "UV LED", "IR LED"]
colors = ['black', 'dodgerblue', 'firebrick']
# Define threshold for voltage change
THRESHOLD = 0.01

# Create lists to store aligned times and voltages
aligned_times = []
aligned_voltages = []

# Process each voltage series
for i in range(0, 6, 2):  # Iterate over every other column for Time and Voltage
    time_series = df.iloc[:, i]
    voltage_series = df.iloc[:, i + 1]

    # Find the index of the first significant change
    change_index = np.argmax(np.abs(voltage_series.diff()) > THRESHOLD)

    # Align the time series so the first significant change is at time = 0
    aligned_time = time_series - time_series.iloc[change_index]
    aligned_times.append(aligned_time)
    aligned_voltages.append(voltage_series)

# Plot each aligned series
for i in range(3):
    plt.plot(aligned_times[i], aligned_voltages[i], label=labels[i],
             color=colors[i])

# add line to show the cutoff line
plt.axhline(0.3, ls='--', color='red', label="Cutoff line")

plt.xlabel('Time (seconds)')
plt.ylabel('Voltage (V)')
# Place the legend on the right side of the plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('LED drive pin voltage')
plt.xlim([-0.1, 0.6])
plt.tight_layout()
plt.show()
plt.savefig("led_drv_pin.eps")
