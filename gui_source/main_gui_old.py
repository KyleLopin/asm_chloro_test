# Copyright (c) 2017-2018 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

""" File to start for a Graphical User interface for the AS726X Sparkfun breakout board connected
to a PSoC controller.  The data is saved in the data_class file, pyplot_embed has a
matplotlib graph embedded into a tk.Frame to display the data and usb_comm communicates with the device. """

import logging
import tkinter as tk
from tkinter import ttk
# standard libraries
from enum import Enum

# installed libraries
# local files
import arduino
import device_settings
import psoc_spectrometers
import pyplot_embed
import reg_toplevel
# import serial_comm
import spectro_frame


AS7265X_WAVELENGTHS = [610, 680, 730, 760, 810, 860,
                       560, 585, 645, 705, 900, 940,
                       410, 435, 460, 485, 510, 535]

# sort the wavelenghts of AS7265x
AS7265X_SORT_INDEX = sorted(range(len(AS7265X_WAVELENGTHS)),
                            key=AS7265X_WAVELENGTHS.__getitem__)


__author__ = 'Kyle Vitautas Lopin'

# if getattr(sys, 'frozen', False):
#     # we are running in a |PyInstaller| bundle
#     basedir = sys._MEIPASS
# else:
#     # we are running in a normal Python environment
#     basedir = os.path.dirname(__file__)


class DisplayTypes(Enum):
    counts = "Counts"
    power = u'\u03bcW / cm\u00B2'
    concentration = u"\u03bcmol / (cm\u00B2 \u00D7 s) (\u00D7 10\u207B\u2078)"


class SpectrometerGUI(tk.Tk):
    """ Class to display the controls and data of a color sensor or a spectrometer.  Currently displays the
     current spectrum.

     TODO: add a time course notebook and move the current spectrum to a separate notebook. """

    def __init__(self, parent=None):
        """
        Initialize the graphical user interface by:
        1) Start the logging module
        2) Attach the device using the psoc_spectrometer call, this module will abstract all operations with
        the actual device through the settings, which are all traced to call the proper functions when any of
        the variables are changed.
        3) Make the graph area using pyplot_embed that will display the intensity versus wavelength data.
        4) Make a frame that contains all the buttons used to control the device.
        5) Make a frame to display

        :param parent:  any parent program that could call this GUI
        """

        tk.Tk.__init__(self, parent)
        # logging.basicConfig(format='%(asctime)s %(module)s %(lineno)d: %(levelname)s %(message)s',
        #                     datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)

        self.device = arduino.ArduinoColorSensors()
        # check what devices are attached
        self.sensors = self.device.sensors

        self.graph = pyplot_embed.SpectroPlotterBasic(self)

        # make the status frame with the connect button and status information
        self.status_frame = StatusFrame(self, self.device)
        self.status_frame.pack(side='top', fill=tk.X)
        # self.device.initialize_device()

    def update_graph(self, data: tuple):
        """
        Allow user to call the master class to update the graph for any widget that does not
        have direct access to the graph

        :param data:  data to display on y-axis of graph
        """
        self.graph.update_data(data)

    def device_not_working(self):
        """
        If something goes wrong display an error to the user.
        :return:
        """
        self.status_frame.update_status("Error reading device")

    def write_message(self, message: str):
        """
        Write a message to the microcontroller that controls the sensor(s) / spectrometer(s)
        :param message: string - message to send through the USB to the device.  Check device API to see
        what messages you can send.
        """
        self.device.usb.usb_write(message)

    def reconnect_device(self):
        self.device = psoc_spectrometers.AS7262(self)
        self.settings = self.device.settings
        self.buttons_frame.update_settings(self.settings)
        self.device.initialize_device()

        if self.device.usb.spectrometer:
            return True
        return False


BUTTON_PADY = 5


class ButtonFrame(tk.Frame):

    def __init__(self, parent: tk.Frame, graph, device, sensors):
        """

        :param parent:
        :param settings:
        :param graph:
        :param device:
        """
        tk.Frame.__init__(self, parent)
        sensor_frame = tk.Frame(self)
        if sensors:  # make sure a sensor is attached to make its variable options
            for i, sensor in enumerate(sensors):
                AS726XButtonFrame(sensor_frame, sensor, graph, device).pack(side=tk.LEFT)

        sensor_frame.pack(side=tk.TOP)

        UniversalButtonsAS726X(self, device, graph).pack(side=tk.BOTTOM)


class AS726XButtonFrame(tk.Frame):
    """ Frame to contain all the buttons the user can use to control the settings and use of the device """

    def __init__(self, parent: tk.Frame, sensor: psoc_spectrometers,  # device_settings.DeviceSettings_AS7262): type hinting issue
                 graph, device):
        """
        Class to make all the buttons needed to control a AS7262 sensor that is controlled by a PSoC.

        :param parent:  tkinter Frame or Tk this frame is embedded in.
        :param seensor:  the settings for the PSoC controlled AS7262.  The settings are all traced to
        the approriat functions
        :param graph:  graph area, this is needed because it also contains the data class to be saved also
        """
        tk.Frame.__init__(self, parent)
        self.master = parent
        self.settings = sensor.settings  # type: device_settings.DeviceSettings_AS7262
        self.graph = graph
        self.device = device  # type: psoc_spectrometers.PSoC

        self.config(relief=tk.GROOVE, bd=3)
        tk.Label(self, text=sensor.sensor_type).pack(side='top', pady=BUTTON_PADY)
        # make all the buttons and parameters
        # Gain settings
        tk.Label(self, text="Gain Setting:").pack(side='top', pady=BUTTON_PADY)
        gain_var_options = ["1", "3.7", "16", "64"]
        # gain_var_options = device_settings.GAIN_SETTING_MAP.keys()
        tk.OptionMenu(self, self.settings.gain_var, *gain_var_options).pack(side='top', pady=BUTTON_PADY)

        # set the integrations time, give them just a few options not the full 255 options
        tk.Label(self, text="Integration time (ms):").pack(side='top', pady=BUTTON_PADY)
        custom_range = [1, 2, 4, 8, 16, 32, 64, 128, 255]
        integration_time_var = ["{:.1f}".format(x*5.6) for x in custom_range]

        tk.OptionMenu(self, self.settings.integration_time_var, *integration_time_var).pack(side='top', pady=BUTTON_PADY)

        # make LED control widgets
        tk.Label(self, text="LED power (mA):").pack(side='top', pady=BUTTON_PADY)
        LED_power_options = ["12.5 mA", "25 mA", "50 mA", "100 mA"]
        # LED_power_options = device_settings.LED_POWER_MAP.keys()
        tk.OptionMenu(self, self.settings.LED_power_level_var, *LED_power_options).pack(side='top', pady=BUTTON_PADY)

        self.LED_button = tk.Button(self, text="Turn LED On", command=self.LED_toggle)
        self.LED_button.pack(side='top', pady=BUTTON_PADY)

        # make buttons to control the device
        # button to make a single sensor read
        self.reading = False
        self.read_button = tk.Button(self, text="Single Read", command=self.read_once)
        self.read_button.pack(side="top", pady=BUTTON_PADY)

        # make a check box for if the LED should be flashed
        self.use_flash = tk.IntVar()
        self.flash_checkbutton = tk.Checkbutton(self, text="Use flash", variable=self.use_flash)
        self.flash_checkbutton.pack(side="top", pady=BUTTON_PADY)

        # button to continuously read from the sensor
        self.run_button = tk.Button(self, text="Start Reading", command=self.run_toggle)
        self.run_button.pack(side="top", pady=BUTTON_PADY)

        # button to save the data, this will open a toplevel with the data printed out, and an option to save to file
        tk.Button(self, text="Save Data", command=self.save_data).pack(side="top", pady=BUTTON_PADY)

    def update_settings(self, settings):  # hack
        self.settings = settings

    def get_message(self):
        self.device.usb.usb_write("E")
        print(self.device.usb.usb_read_data(20))

    def LED_toggle(self):
        """
        Toggle the LED of the color sensor that illuminates the sample.
        """
        if self.settings.LED_on:
            # turn off the LED and change the button
            self.settings.toggle_LED(False)
            self.LED_button.config(text="Turn LED on", relief=tk.RAISED)

            # re-activate the LED flash checkbox if it was inactivated earlier
            self.flash_checkbutton.config(state=tk.ACTIVE)
        else:
            self.settings.toggle_LED(True)
            self.LED_button.config(text="Turn LED off", relief=tk.SUNKEN)

            # inactivate the LED flash checkbox if the LED is already on
            self.flash_checkbutton.config(state=tk.DISABLED)

    def run_toggle(self):
        """
        Toggle the device to continuously read
        """
        # self.settings.reading is traced to device settings method toggle_read
        if self.settings.reading.get():  # stop reading
            self.settings.reading.set(False)
            self.read_button.config(state=tk.ACTIVE)

            self.run_button.config(text="Start Reading")
        else:  # start reading
            # don't let the user do a single read when doing continous reads
            self.read_button.config(state=tk.DISABLED)
            self.settings.reading.set(True)

            self.run_button.config(text="Stop Reading")

    def disable_read_buttons(self):
        self.run_button.config(state=tk.DISABLED)
        self.read_button.config(state=tk.DISABLED)

    def enable_read_buttons(self):
        self.run_button.config(state=tk.ACTIVE)
        self.read_button.config(state=tk.ACTIVE)

    def average_reads(self):
        """ Not implemented yet """
        print("average read: ", self.settings.average_reads.get())

    def read_once(self):
        """
        Take a single sensor read
        """
        self.read_button.config(state=tk.DISABLED)
        logging.debug("read once with flash: {0}".format(self.use_flash.get()))
        self.settings.single_read(self.use_flash.get())
        self.read_button.config(state=tk.ACTIVE)

    def read_just_data(self):
        self.settings.device.read_data()

    def save_data(self):
        """
        Save the set of data that is being displayed
        """
        self.graph.data.save_data()


class UniversalButtonsAS726X(tk.Frame):
    def __init__(self, parent: tk.Frame, device, graph):
        tk.Frame.__init__(self, parent)
        self.graph = graph
        # radio buttons to choose what data type to display
        tk.Label(self, text="Data Display Type:").pack(side="top", pady=BUTTON_PADY)
        # self.display_type = tk.StringVar()
        self.display_type = tk.StringVar()
        self.display_type.set(DisplayTypes.counts.value)
        self.display_type.trace("w", self.toggle_display_type)

        top_frame = tk.Frame(self)
        top_frame.pack(side=tk.TOP)
        ttk.Radiobutton(top_frame, text=DisplayTypes.counts.value, variable=self.display_type,
                        value=DisplayTypes.counts.value).pack(side=tk.LEFT, pady=BUTTON_PADY)
        ttk.Radiobutton(top_frame, text=DisplayTypes.power.value, variable=self.display_type,
                        value=DisplayTypes.power.value).pack(side=tk.LEFT, pady=BUTTON_PADY)
        ttk.Radiobutton(self, text=DisplayTypes.concentration.value, variable=self.display_type,
                        value=DisplayTypes.concentration.value).pack(side=tk.TOP, pady=BUTTON_PADY)

        # button to debug the sensor by writing or reading the sensors (virtual) registers
        tk.Button(self, text="Register Check", command=lambda: reg_toplevel.RegDebugger(self.master, device)
                  ).pack(side="top", pady=BUTTON_PADY)

        # tk.Button(self, text="Check USB Data", command=self.read_just_data).pack(side="top", pady=BUTTON_PADY)

        # tk.Button(self, text="Check commands", command=self.get_message).pack()

    def toggle_display_type(self, *args):
        logging.debug("Changing display type to: {0}".format(self.display_type.get()))
        self.graph.change_data_units(self.display_type.get())


class StatusFrame(tk.Frame):
    """ Frame to display information about the sensors and device attached """

    def __init__(self, parent: tk.Tk, device):
        """
        Make all the information the user should know about device available to them.

        :param parent:
        :param device:
        """
        tk.Frame.__init__(self, parent)
        self.master = parent  # type: SpectrometerGUI
        self.device = device  # type: psoc_spectrometers.PSoC
        self.status_str = tk.StringVar()
        if device.sensors:
            print('check2')
            # self.status_str.set("Spectrometer: {0} connected".format(device.usb.spectrometer.decode('utf-8')))
            # self.status_str.set("Spectrometer: {0} connected".format(device.usb.sensor_message))
        else:
            self.status_str.set("No sensors found")

        self.status_label = tk.Label(self, textvariable=self.status_str)
        self.status_label.pack(side='left')
        self.status_label.bind("<Button-1>", self.device_connection_test)

    def update_status(self, message):
        self.status_str.set(message)
        self.status_label = tk.Label(self, textvariable=self.status_str)

        if not self.device.usb.spectrometer:
            self.status_label.config(bg='red')
            self.status_label = tk.Label(self, textvariable=self.status_str)
            psoc_spectrometers.ConnectionStatusToplevel(self.master, self.status_str)

        # self.status_str.set(message)
        # self.status_label = tk.Label(self, textvariable=self.status_str)

    def device_connection_test(self, *args):
        logging.debug("Checking the status of the device")

        psoc_spectrometers.ConnectionStatusToplevel(self.master, self.status_str)


if __name__ == '__main__':
    app = SpectrometerGUI()
    app.title("Spectrograph")
    app.geometry("900x750")
    app.mainloop()
