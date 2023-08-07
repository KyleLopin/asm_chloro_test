# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Classes to keep track of settings and interact with AS7262, AS7263 (AS726X) and
AS7265x color sensors
"""

__author__ = "Kyle Vitatus Lopin"

# standard libraries
from datetime import date, datetime
import os
import tkinter as tk

WAVELENGTH_AS7262 = [450, 500, 550, 570, 600, 650]
WAVELENGTH_AS7263 = [610, 680, 730, 760, 810, 860]
WAVELENGTH_AS7265X = [410, 435, 460, 485, 510, 535,
                      565, 585, 610, 645, 680, 705,
                      730, 760, 810, 860, 900, 940]
LED_CURRENT = {0: "12.5 mA", 1: "25 mA", 2: "50 mA", 3: "100 mA"}


class AS7262():
    def __init__(self, arduino, button_attached: bool, mux_number: int):
        self.has_button = button_attached
        self.name = "AS7262"
        self.qwiic_port = mux_number
        self.indicator = "Off"
        self.device = arduino
        self.wavelengths = WAVELENGTH_AS7262
        self.data = SensorData(self)
        today = date.today()

        self.filename = "Data/{0}_{1}.csv".format(today, self.name)

    def __str__(self):
        button_str = "Button not found"
        if self.has_button:
            button_str = "Button found"
        string = "{0} sensor| port {1}\n{2}".format(self.name,
                                                      self.qwiic_port,
                                                      button_str)
        return string

    def display(self, master):
        pad_y = 5
        sensor_frame = tk.Frame(master, relief=tk.RIDGE, bd=5)
        text_str = self.__str__()
        tk.Label(sensor_frame, text=text_str).pack(side=tk.TOP, pady=pad_y)
        sensor_frame.pack(side=tk.LEFT, expand=2, fill=tk.BOTH, pady=pad_y)

        ind_options = ["No Indicator", "Indicator LED on", "Flash Indicator LED"]
        if self.has_button:
            ind_options.extend(["Button LED on", "Flash Button LED"])

        self.ind_opt_var = tk.StringVar()
        self.ind_opt_var.set(ind_options[0])

        tk.OptionMenu(sensor_frame, self.ind_opt_var, *ind_options,
                      command=self.indicator_options).pack(side=tk.TOP, pady=pad_y)

        # LED display options
        tk.Label(sensor_frame, text="Measurement\nLighting options:").pack(side=tk.TOP, pady=pad_y)
        led_options = ["No lights", "Light on", "Flash light"]

        self.led_opt_var = tk.StringVar()
        self.led_opt_var.set(led_options[2])

        tk.OptionMenu(sensor_frame, self.led_opt_var, *led_options,
                      command=self.set_led_option).pack(side=tk.TOP, pady=pad_y)

        tk.Button(sensor_frame, text="Read Sensor", command=self.read_sensor).pack(side=tk.TOP, pady=pad_y)

        self.tracker = TrackerFrame(self, sensor_frame)
        self.tracker.pack(side=tk.TOP)

    def read_sensor(self):
        print("read sensor: {0}".format(self.qwiic_port))
        self.device.write("Read:{0}".format(self.qwiic_port))

    def read(self, data):
        print(self.filename)

    def set_led_option(self, command):
        print(command, self.qwiic_port)

    def increase_read_num(self):
        print('check333')
        self.tracker.update_read(increase=True)

    def indicator_options(self, command):
        print(command, self.qwiic_port)
        print('foobar')
        msg = "{0}:{1}".format(command, self.qwiic_port).encode()
        print(self.device)

        if "Flash" in command:
            # this is not pretty below
            self.device.master.after(2000, self.turn_leds_off)
            msg = "{0}{1}:{2}".format(command.split("Flash ")[1], " on",
                                      self.qwiic_port).encode()
        self.device.write(msg)

    def turn_leds_off(self):
        self.device.write("No Indicator:{0}".format(self.qwiic_port))
        self.ind_opt_var.set("No Indicator")


class TrackerFrame(tk.Frame):
    def __init__(self, sensor, master):
        tk.Frame.__init__(self, master)
        print(type(master))
        print(type(self))
        self.read_num = 1
        self.leaf_num = tk.IntVar()
        self.leaf_num.set(1)
        self.read_label = tk.Label(self,
                          text="Read: {0}".format(self.read_num))
        self.read_label.pack(side=tk.TOP)
        # tk.Label(self, text="hello").pack(side=tk.TOP)
        leaf_num_frame = tk.Frame(self)
        tk.Label(leaf_num_frame, text="Leaf number:").pack(side=tk.LEFT)
        tk.Spinbox(leaf_num_frame, from_=0, textvariable=self.leaf_num,
                   command=self.increase_leaf, width=2).pack(side=tk.LEFT)
        leaf_num_frame.pack(side=tk.TOP)

    def update_read(self, increase: bool):
        print('update readpPp:', self.read_num)
        if increase:
            self.read_num += 1
        else:
            self.read_num = 1
        self.read_label.config(text="Read: {0}".format(self.read_num))
        print('update read', self.read_num)
    #
    def increase_leaf(self):
        self.update_read(False)
        self.leaf_num.set(self.leaf_num.get()+1)

    def get_read_num(self):
        return self.read_num

    def get_leave_num(self):
        return self.leaf_num.get()


class AS7263(AS7262):
    def __init__(self, *args):
        # AS7262.__init__(self, kw)
        print('++++')
        print(args)
        super(AS7263, self).__init__(*args)
        self.name = "AS7263"
        today = date.today()
        self.filename = "data/{0}_{1}.csv".format(today, self.name)
        print(self.filename)
        self.wavelengths = WAVELENGTH_AS7263
        self.data = SensorData(self)


class AS7265x(AS7262):
    def __init__(self, *args):
        super(AS7265x, self).__init__(*args)
        self.name = "AS7265x"
        today = date.today()
        self.filename = "data/{0}_{1}.csv".format(today, self.name)
        self.wavelengths = WAVELENGTH_AS7265X
        self.data = SensorData(self)


class SensorData():
    def __init__(self, sensor: AS7262):
        self.wavelengths = sensor.wavelengths
        self.sensor = sensor
        self.gain = 64  # this is default of Sparkfun library
        self.int_cycles = 150  # default for AS726x, 49 for AS7265x
        self.data = []
        self.led_current = ""  # default of program
        self.norm_data = []
        self.data_string = ""
        self.LED = "White LED"

    def set_led_current(self, new_current):
        self.led_current = LED_CURRENT[new_current]

    def set_int_cylces(self, new_cycles):
        self.int_cycles = new_cycles

    def set_LED(self, led):
        self.LED = led

    def set_data(self, new_data):
        self.data = new_data
        self.norm_data = [x*250/self.int_cycles for x in self.data]

    def save_data(self, saturation_check):
        if not os.path.isfile(self.sensor.filename):
            header = 'Leaf number,Read number,gain,integration time,'
            for wavelength in self.wavelengths:
                header += "{0} nm,".format(wavelength)
            header += "led,led current,saturation check,time"
            with open(self.sensor.filename, mode='a',
                      encoding='utf-8') as _file:
                _file.write(header + '\n')

        print('filename: ', self.sensor.filename)
        print('leave number: ', self.sensor.tracker.get_leave_num())
        print('read number: ', self.sensor.tracker.get_read_num())
        data_str = "{0},{1},{2},{3},".format(self.sensor.tracker.get_leave_num(),
                                             self.sensor.tracker.get_read_num(),
                                             self.gain, self.int_cycles)
        for data in self.norm_data:
            data_str += "{0:10.3f},".format(data)
        data_str += "{0},{1},{2},{3}\n".format(self.LED, self.led_current,
                                               saturation_check,
                                               datetime.now().strftime("%H:%M:%S"))
        print(data_str)
        with open(self.sensor.filename, mode='a',
                  encoding='utf-8') as _file:
            _file.write(data_str)
