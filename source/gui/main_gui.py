# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
GUI to interact with an arduino connected to multiple AS7262, AS7263, and/or AS7265x color sensors
"""


__author__ = "Kyle Vitatus Lopin"

# standard libraries
import tkinter as tk
import tkinter.font as tkFont
from tkinter import ttk
# installed libraries
import numpy as np
# local files
import arduino
import AS726XX  # for type hinting
import pyplot_embed

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

# _font = tkFont.Font(family="Helvetica", size=24)
# _font = tkFont.Font(family="Lucida Grande", size=24)

class SpectralSensorGUI(tk.Tk):
    def __init__(self, parent=None):
        tk.Tk.__init__(self, parent)

        style = ttk.Style(self)
        style.configure('lefttab.TNotebook', tabposition='ne')
        # access the class to control the Arduino
        # / Red board that the sensor are attached to
        self.device = arduino.ArduinoColorSensors(self)
        print("device:", self.device)
        while self.device.starting_up:
            pass
        # make the graph area
        self.graph = pyplot_embed.SpectroPlotterBasic(self)
        self.graph.pack(side=tk.LEFT, fill=tk.BOTH, expand=2)
        # ButtonFrame(self, self.device).pack(side=tk.BOTTOM, fill=tk.X)
        ButtonFrame(self, self.device).pack(side=tk.RIGHT, fill=tk.Y)
        self.after(100, self.look_for_data)

    def look_for_data(self):
        if self.device.graph_event.is_set():
            self.device.graph_event.clear()
            sensor, sat_check = self.device.graph_queue.get()  # type: AS726XX
            if sat_check == 'Clear':  # this is a hack, fix later
                self.graph.delete_data()
            else:
                data = sensor.data
                label = "{0}: {1} cycles, {2} {3} led current".format(sensor.name,
                                  data.int_cycles, data.led_current, data.LED)
                print("============>>>>>>>>>>>>>>>>>>> ", label)
                self.graph.update_data(data.wavelengths,
                                       data.norm_data,
                                       label)

        self.after(100, self.look_for_data)


class ButtonFrame(tk.Frame):
    def __init__(self, master, device: arduino.ArduinoColorSensors):

        tk.Frame.__init__(self, master=master)
        # notebook = ttk.Notebook(self, style='lefttab.TNotebook')
        self.notebook = ttk.Notebook(self)
        self.sensors = device.sensors
        print('=======', len(self.sensors))
        self.tabs = []
        self.tab_names = []
        for sensor in self.sensors:
            tab = tk.Frame(self)
            print(dir(sensor))
            sensor.display(tab)
            tab_display = "{0} [{1}]".format(sensor.name, sensor.qwiic_port)
            self.notebook.add(tab, text=tab_display)
            self.tabs.append(tab)
            self.tab_names.append(sensor.name)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=2)

        self.make_summary_frame()

    def make_summary_frame(self):
        pad_y = 5

        _font = tkFont.Font(family="Helvetica", size=10)
        for sensor in self.sensors:
            sum_frame = tk.Frame(self, relief=tk.RIDGE, bd=2)
            sum_frame.bind("<Button-1>", lambda event, s=sensor: self.label_press(event, s))
            _label = tk.Label(sum_frame, font=_font,
                              text="{0} on port: {1}".format(sensor.name, sensor.qwiic_port))
            _label.pack(side=tk.TOP, expand=1, fill=tk.BOTH,
                        pady=pad_y, padx=2)
            _label.bind("<Button-1>", lambda event, s=sensor: self.label_press(event, s))
            print("bind")
            # _label.bind("<Button-1>", lambda event, s=sensor: self.label_press(event, s))

            print("bind2")
            tk.Button(sum_frame, text="Read Sensor",
                      command=sensor.read_sensor
                      ).pack(side=tk.TOP, expand=1, fill=tk.BOTH,
                             pady=pad_y, padx=2)

            sum_frame.pack(side=tk.TOP, fill=tk.BOTH)

    def label_press(self, event, sensor):
        print(event.widget)
        print(dir(event))
        print(dir(event.widget))
        print(sensor)
        print(sensor.name)
        print(self.tab_names.index(sensor.name))
        tab_index = self.tab_names.index(sensor.name)
        self.notebook.select(self.tabs[tab_index])


class ButtonFrame_old(tk.Frame):
    def __init__(self, master, device):
        tk.Frame.__init__(self, master=master)
        # self.config(bg='red', bd=5)
        self.sensors = device.sensors
        print('=======', len(self.sensors))
        for sensor in self.sensors:
            print(sensor)
            # sensor_frame = tk.Frame(self, relief=tk.RIDGE, bd=5)
            # text_str = sensor.__str__()
            # tk.Label(sensor_frame, text=text_str).pack(side=tk.LEFT)
            # sensor_frame.pack(side=tk.LEFT)
            # tk.Checkbutton(sensor_frame, text="Turn on indicator", )
            sensor.display(master)

    def turn_on_indicator(self):
        pass



if __name__ == '__main__':
    app = SpectralSensorGUI()
    app.title("Spectrograph")
    app.geometry("800x650")
    app.mainloop()
