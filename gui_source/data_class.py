# Copyright (c) 2017-2018 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

""" Data class to hold color sensor / spectrometer data """

# standard libraries
import logging
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
# local files
import device_settings
import main_gui_old
import psoc_spectrometers


__author__ = 'Kyle V. Lopin'

WAVELENGTH_AS7262 = [450, 500, 550, 570, 600, 650]
WAVELENGTH_AS7263 = [610, 680, 730, 760, 810, 860]

UW_PER_COUNT = 1 / 45.0  # AS7262 datasheet
CALIBRATION_INTEGRATION_SETTING = 166.0  # AS7262 datasheet
CALIBRATION_GAIN_SETTING = 16.0  # AS7262 datasheet

CONVERSION_eV_TO_nm = 1.0 / 1239.8  # nm (wavelength) * eV
CONVERSION_uJ_TO_eV = 6.241 * (10**12)  # eV / uJ
CONVERT_COUNT_TO_uMOLE = 1.0 / (6.022 * (10**17))
CONCENTRATION_SCALE_FACTOR = 10**8


class Data(object):
    def __init__(self, sensors: list):
        self.data = []
        self.wavelengths = None
        self.sensors = sensors
        if sensors:
            for sensor in sensors:  # type: psoc_spectrometers.AS726X
                self.data.append(SpectrometerData(sensor.settings))

        # figure out how to deal with current_data here
        self.current_data = None

    def update_data(self, data_counts):
        if self.sensors:
            for i, sensor in enumerate(self.sensors):
                # print('sensor_types:', sensor.sensor_type)
                if i == 0:
                    if sensor.sensor_type == "AS7262":
                        self.current_data = data_counts
                        self.wavelengths = WAVELENGTH_AS7262
                    elif sensor.sensor_type == "AS7263":
                        self.current_data = data_counts
                        self.wavelengths = WAVELENGTH_AS7263
                else:
                    if sensor.sensor_type == "AS7262":
                        self.wavelengths.append(WAVELENGTH_AS7262)
                    elif sensor.sensor_type == "AS7263":
                        self.wavelengths.append(WAVELENGTH_AS7263)

    def set_data_type(self, data_type):
        if not self.data:
            return
        for data in self.data:
            data.set_data_type(data_type)

    def calculate_conversion_factors(self):
        if not self.data:
            return
        for data in self.data:
            data.calculate_conversion_factors()


class SpectrometerData(object):
    def __init__(self, settings):
        # print('c1: ', settings)
        self.counts = None
        self.power_levels = None
        self.measurement_mode = None
        self.conc_levels = [0] * len(settings.wavelengths)
        self.current_data = None

        self.settings = settings  # type: device_settings.DeviceSettings_AS7262
        self.gain_var = settings.gain_var  # type: tk.StringVar
        self.integration_time_var = settings.integration_time_var  # type: tk.StringVar
        self.wavelengths = settings.wavelengths

        self.power_conversion = None
        self.concentration_conversion = None
        self.calculate_conversion_factors()

    def update_data(self, data_counts):
        logging.debug("updating data")
        self.counts = data_counts
        self.power_levels = [x*self.power_conversion for x in data_counts]
        for i, x in enumerate(self.counts[:5]):
            self.conc_levels[i] = x * self.concentration_conversion[i]

        self.measurement_mode = self.settings.measurement_mode_var.get()
        logging.debug("making current data of type: {0}".format(self.measurement_mode))
        logging.debug("Converted counts: {0}".format(self.counts))
        logging.debug("to {0} power levels".format(self.power_levels))
        logging.debug("and {0} mols".format(self.conc_levels))

        self.set_data_type()

    def set_data_type(self, measurement_mode=None):
        if not measurement_mode:
            measurement_mode = self.measurement_mode
        else:
            self.measurement_mode = measurement_mode
        # measurement_mode = self.settings.measurement_mode_var.get()
        logging.debug("setting data type: {0}".format(measurement_mode))
        if measurement_mode == main_gui_old.DisplayTypes.counts.value:
            logging.debug("setting data as counts")
            self.current_data = self.counts
        elif measurement_mode == main_gui_old.DisplayTypes.concentration.value:
            logging.debug("setting data as moles")
            self.current_data = self.conc_levels
        else:
            logging.debug("setting data as power")
            self.current_data = self.power_levels

    def calculate_conversion_factors(self):
        gain = float(self.gain_var.get())
        time = float(self.integration_time_var.get())
        logging.debug("conversion time: {0}".format(time))
        logging.debug("conversion gain: {0}".format(gain))
        power_conversion = (UW_PER_COUNT *
                            (CALIBRATION_INTEGRATION_SETTING / time) *
                            (CALIBRATION_GAIN_SETTING / gain))
        logging.debug("power conversion factor: {0}".format(power_conversion))

        mol_conversion = [power_conversion] * len(self.wavelengths)

        for i, wavelength in enumerate(self.wavelengths):
            mol_conversion[i] *= (CONVERSION_uJ_TO_eV * wavelength * CONVERSION_eV_TO_nm *
                                  CONCENTRATION_SCALE_FACTOR * CONVERT_COUNT_TO_uMOLE)

        logging.debug("mol convert: {0}".format(mol_conversion))

        self.power_conversion = power_conversion
        self.concentration_conversion = mol_conversion

    def save_data(self):
        if not self.current_data:  # if no data run has been called yet, just pass
            return
        SaveTopLevel(self.wavelengths, self.current_data,
                     self.settings.measurement_mode_var.get(),
                     self.settings)


class SaveTopLevel(tk.Toplevel):
    def __init__(self, wavelength_data: list, light_data: list,
                 data_type: str, settings):
        tk.Toplevel.__init__(self, master=None)
        # set basic attributes
        self.attributes('-topmost', 'true')
        self.geometry('450x380')
        self.title("Save data")

        # strings to display data to user
        self.full_data_string = tk.StringVar()  # string to store wavelength and data
        self.data_string = tk.StringVar()  # string to hold just the data

        self.full_data_string = "Wavelength (nm), {0}\n".format(data_type)
        self.data_string = "{0}\n".format(data_type)
        for i, _data in enumerate(wavelength_data):
            self.full_data_string += "{0}, {1:4.3f}\n".format(_data, light_data[i])
            self.data_string += "{0:4.3f}\n".format(light_data[i])

        # make the area
        self.text_box = tk.Text(self, width=50, height=8)
        self.text_box.insert(tk.END, self.full_data_string)
        self.text_box.pack(side='top', pady=6)

        # allow user to display just the data not the wavelengths all the time
        self.display_type = tk.IntVar()
        tk.Checkbutton(self, text="Show just data (no wavelengths)", command=self.toggle_data_display,
                       variable=self.display_type).pack(side='top', pady=6)

        # Allow the user to add comments to the data file
        tk.Label(self, text="Comments:").pack(side='top', pady=6)

        # make string prepopulated with settings
        # this dictionary loops are horrible
        power = None
        for value, power_setting in device_settings.LED_POWER_MAP.items():
            if power_setting == settings.run_settings['power']:
                power = value

        lighting_str = ""
        if settings.run_settings['LED on']:
            lighting_str += "LED on with {0}".format(power)
        elif settings.run_settings['flash']:
            lighting_str += "LED flash with {0}".format(power)
        else:
            lighting_str += "No lighting"
        # get gain settings
        gain = None
        for value, gain_setting in device_settings.GAIN_SETTING_MAP.items():
            if gain_setting.value == settings.run_settings['gain']:
                gain = value
            elif value == settings.run_settings['gain']:  # horrible hack but should work
                gain = value

        self.details_str = "gain: {0}, integration time: {1} ms\n{2}".format(gain,
                                                                             settings.run_settings['integration time'],
                                                                             lighting_str)

        self.comment = tk.Text(self, width=50, height=5)
        self.comment.insert(tk.END, self.details_str)
        self.comment.pack(side='top', pady=6)

        # allow user to remove the details
        self.add_details = tk.IntVar()
        tk.Checkbutton(self, text="Add run details", command=self.toggle_details,
                       variable=self.add_details).pack(side='top', pady=6)
        self.add_details.set(1)

        button_frame = tk.Frame(self)
        button_frame.pack(side='top', pady=6)
        tk.Button(button_frame, text="Save Data", command=self.save_data).pack(side='left', padx=10)
        tk.Button(button_frame, text="Close", command=self.destroy).pack(side='left', padx=10)

    def toggle_data_display(self):
        if self.display_type.get():  # button is checked
            self.text_box.delete(1.0, tk.END)
            self.text_box.insert(tk.END, self.data_string)
        else:
            self.text_box.delete(1.0, tk.END)
            self.text_box.insert(tk.END, self.full_data_string)

    def toggle_details(self):
        if not self.add_details.get():  # button is checked
            self.comment.delete(1.0, tk.END)
        else:
            self.comment.insert(tk.END, self.details_str)

    def save_data(self):
        try:
            _filename = open_file(self, 'saveas')  # open the file
        except Exception as error:
            messagebox.showerror(title="Error", message=error)
        self.attributes('-topmost', 'true')

        if not _filename:
            self.destroy()
        # a file was found so open it and add the data to it
        with open(_filename, mode='a', encoding='utf-8') as _file:

            if self.comment.get(1.0, tk.END):
                self.data_string += self.comment.get(1.0, tk.END)
            try:
                _file.write(self.data_string)
                _file.close()
                self.destroy()

            except Exception as error:

                messagebox.showerror(title="Error", message=error)
                self.lift()
                _file.close()


def open_file(parent, _type: str) -> str:
    """
    Make a method to return an open file or a file name depending on the type asked for
    :param parent:  master tk.TK or toplevel that called the file dialog
    :param _type:  'open' or 'saveas' to specify what type of file is to be opened
    :return: filename user selected
    """
    """ Make the options for the save file dialog box for the user """
    file_opt = options = {}
    options['defaultextension'] = ".csv"
    # options['filetypes'] = [('All files', '*.*'), ("Comma separate values", "*.csv")]
    options['filetypes'] = [("Comma separate values", "*.csv")]
    logging.debug("saving data: 1")
    if _type == 'saveas':
        """ Ask the user what name to save the file as """
        logging.debug("saving data: 2")
        _filename = filedialog.asksaveasfilename(parent=parent, confirmoverwrite=False, **file_opt)
        return _filename

    elif _type == 'open':
        _filename = filedialog.askopenfilename(**file_opt)
        return _filename
