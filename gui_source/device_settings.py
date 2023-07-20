# standard libraries
from enum import Enum
import logging
import tkinter as tk

# local files
import data_class
import psoc_spectrometers

__author__ = 'Kyle Vitautas Lopin'

# this might not be needed
WAVELENGTH_AS7262 = [450, 500, 550, 570, 600, 650]
WAVELENGTH_AS7263 = [610, 680, 730, 760, 810, 860]


class BankMode(Enum):
    BANK_MODE_0 = 0
    BANK_MODE_1 = 1
    BANK_MODE_2 = 2
    BANK_MODE_3 = 3


class GainSetting(Enum):
    GAIN_SETTING_1X = 0
    GAIN_SETTING_3_7X = 1
    GAIN_SETTING_16X = 2
    GAIN_SETTING_64X = 3


GAIN_SETTING_MAP = {"1": GainSetting.GAIN_SETTING_1X, "3.7": GainSetting.GAIN_SETTING_3_7X,
                    "16": GainSetting.GAIN_SETTING_16X, "64": GainSetting.GAIN_SETTING_64X}


class LEDPowerSetting(Enum):
    LED_POWER_12_5_mA = 0
    LED_POWER_25_mA = 1
    LED_POWER_50_mA = 2
    LED_POWER_100_mA = 3


LED_POWER_MAP = {"12.5 mA": LEDPowerSetting.LED_POWER_12_5_mA,
                 "25 mA": LEDPowerSetting.LED_POWER_25_mA,
                 "50 mA": LEDPowerSetting.LED_POWER_50_mA,
                 "100 mA": LEDPowerSetting.LED_POWER_100_mA}


class IndPowerSetting(Enum):
    IND_POWER_1_mA = 0
    IND_POWER_2_mA = 1
    IND_POWER_4_mA = 2
    IND_POWER_8_mA = 3


class DisplayTypes(Enum):
    counts = "Counts"
    power = "uW / cm2"
    concentration = "umol / cm2"


READ_RATE_MAP = {"200 ms": 0.2, "500 ms": 0.5, "1 sec": 1, "5 sec": 5,
                 "10 sec": 10, "30 sec": 30}

# TODO: set the __set__ method to keep numbers in range


class AS726X_Settings(object):
    def __init__(self, device: psoc_spectrometers.AS726X, type):
        self.device = device
        self.type = type
        logging.debug('type: {0}'.format(type))
        self.gain = GainSetting.GAIN_SETTING_1X.value
        self.measurement_mode = BankMode.BANK_MODE_3.value
        self.integration_time = 5.6 * 255
        self.LED_power_level = LEDPowerSetting.LED_POWER_12_5_mA
        self.LED_on = False
        self.ind_power_level = IndPowerSetting.IND_POWER_1_mA
        self.ind_on = False
        self.read_period = 1000  # milliseconds per read
        self.reading = tk.BooleanVar()
        self.reading.set(False)
        self.reading.trace("w", self.toggle_read)
        self.run_settings = None  # variable to save the run setting to save in the data file

        # if the device should continuously read and average the data to show
        self.average_reads = tk.BooleanVar()
        self.average_reads.set(True)

        self.gain_var = tk.StringVar()
        self.gain_var.set("1")
        self.gain_var.trace("w", self.gain_var_set)

        # leave the device only using bank mode 3 for now
        self.measurement_mode_var = tk.StringVar()

        self.integration_time_var = tk.StringVar()
        self.integration_time_var.set(self.integration_time)
        self.integration_time_var.trace("w", self.integration_time_set)

        self.LED_power_level_var = tk.StringVar()
        self.LED_power_level_var.set(str("12.5 mA"))
        self.LED_power_level_var.trace("w", self.LED_power_set)

        self.read_period_var = tk.StringVar()
        self.read_period_var.set("1 sec")
        self.read_period_var.trace("w", self.read_period_set)

        if type == "AS7262":
            self.wavelengths = WAVELENGTH_AS7262
        elif type == "AS7263":
            self.wavelengths = WAVELENGTH_AS7263
        logging.debug('type: {0}; wavelengthss: {1}'.format(type, self.wavelengths))

        self.graph = None

    def gain_var_set(self, *args):
        # self.gain = GAIN_SETTING_MAP[self.gain_var.get()].value
        self.gain = GAIN_SETTING_MAP[self.gain_var.get()].value
        logging.info("gain ={0}".format(self.gain))
        self.device.set_gain(self.gain)

    def integration_time_set(self, *args):
        self.integration_time = int(float(self.integration_time_var.get()))
        self.device.set_integration_time(self.integration_time)

    def LED_power_set(self, *args):
        logging.debug("Got LED power: ".format(self.LED_power_level_var.get()))
        self.LED_power_level = LED_POWER_MAP[self.LED_power_level_var.get()]
        logging.debug("New LED power level: ".format(self.LED_power_level.value))
        self.device.set_LED_power_level(self.LED_power_level.value)

    def toggle_LED(self, turn_LED_on: bool):
        if turn_LED_on:
            self.LED_on = True
        else:
            self.LED_on = False
        self.device.set_LED_power(self.LED_on)

    def read_period_set(self, *args):
        self.read_period = 1000.0 * float(self.read_period_var.get().split()[0])
        self.device.set_read_period(self.read_period)

    def toggle_read(self, graph, *args):
        self.graph = graph
        if self.reading.get():  # has just been set to true
            # run_integration_time = max(self.integration_time, 200)
            # logging.debug("starting continuous read with integration time: {0}".format(run_integration_time))
            self.device.start_continuous_read(graph)
        else:
            self.device.stop_read()

    def single_read(self, graph, flash=False):
        # save run settings
        self.run_settings = {'gain': self.gain, 'integration time': self.integration_time, 'flash': flash,
                             'power': self.LED_power_level, 'LED on': self.LED_on}
        # print('single read: ', self)
        self.device.read_once(graph, flash)
