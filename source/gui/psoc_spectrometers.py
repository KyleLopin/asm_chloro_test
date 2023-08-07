# Copyright (c) 2017-2018 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

""" Classes to represent different color spectrometers, implimented so far: AS7262"""

# standard libraries
import logging
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox
# local files
import device_settings
import main_gui_old
import pyplot_embed
import spectro_frame
import usb_comm

__author__ = 'Kyle Vitautas Lopin'

DEVICE_TYPE = "WiPy"

class PSoC(object):
    def __init__(self, master_app: tk.Tk):
        self.master_app = master_app
        usb_device = USB()
        self.usb = usb_device.usb  # alias to easier attribute
        self.sensors_list = self.usb.spectrometer
        self.sensors = []
        logging.debug("sensor list: {0}".format(self.sensors_list))
        if self.sensors_list and "No Sensor" not in self.sensors_list[0]:
            for sensor in self.sensors_list:
                self.sensors.append(AS726X(self.usb, sensor, master_app))

    # def initialize_device(self, frame: spectro_frame.ColorSpectorFrame):
    #     if self.sensors:
    #         for sensor in self.sensors:
    #             sensor.set_gain(0)
    #             sensor.set_integration_time(1428)
    #             sensor.set_LED_power(False)
    #             sensor.set_LED_power(0)


class USB(object):
    """ USB communication port to communicate with a microcontroller / PSoC with """
    def __init__(self):
        self.INFO_IN_ENDPOINT = 1
        self.OUT_ENDPOINT = 2
        self.DATA_IN_ENDPOINT = 3
        self.USB_INFO_BYTE_SIZE = 32

        # the data reading from the USB will be on a separate thread so that polling
        # the USB will not make the program hang.
        self.data_queue = queue.Queue()
        self.data_acquired_event = threading.Event()
        self.termination_flag = False  # flag to set when streaming data should be stopped

        self.usb = usb_comm.PSoC_USB(self, self.data_queue, self.data_acquired_event,
                                     self.termination_flag)

    def data_process(self, *args):
        print("original process_data")
        pass


class AS726X(object):
    def __init__(self, usb_communication: USB, sensor_type, master_app):
        self.master = master_app  # type: main_gui_old.SpectrometerGUI
        self.usb = usb_communication
        self.sensor_type = sensor_type
        self.settings = device_settings.AS726X_Settings(self, sensor_type)

        self.integration_time_per_cycle = 5.6  # ms
        self.reading = None
        self.after_delay = int(max(float(self.settings.integration_time), 200))
        self.currently_running = False

    def initialize_device(self, new_master_graph):
        self.master.graph = new_master_graph
        # initialize all the settings in case the program restarts
        self.set_gain(0)
        self.set_integration_time(1428)
        self.set_LED_power(False)
        self.set_LED_power(0)

    def set_gain(self, gain_setting):
        self.usb.usb_write("{0}|GAIN|{1}".format(self.sensor_type, gain_setting))
        self.master.graph.update_data_conversion_factors()

    def set_integration_time(self, integration_time_ms):
        integration_cycles = int(integration_time_ms / self.integration_time_per_cycle)
        self.usb.usb_write("{0}|INTEGRATE_TIME|{1}".format(self.sensor_type, str(integration_cycles).zfill(3)))
        self.after_delay = int(max(float(self.settings.integration_time), 200))
        self.master.graph.update_data_conversion_factors()

    def set_read_period(self, read_period_ms: float):
        self.usb.usb_write("SET_CONT_READ_PERIOD|{0}".format(str(int(read_period_ms)).zfill(5)))

    def start_continuous_read(self, graph):
        self.reading = True
        self.reading_run(graph)

    def reading_run(self, graph):
        # self.reading = self.master.after((self.settings.read_period - 100), self.reading_run)
        # print("try to read", time.time())
        if self.reading:
            self.read_once(graph)
            self.master.after(int(self.settings.read_period - 100), self.reading_run)

    def data_read(self):
        while not self.termination_flag:
            logging.debug("data read call: {0}".format(self.termination_flag, hex(id(self.termination_flag))))
            self.data_acquired_event.wait(timeout=0.2)  # wait for the usb communication thread to
            self.data_acquired_event.clear()
            if not self.data_queue.empty():  # make sure there is data in the queue to process
                new_data = self.data_queue.get()
                self.master.update_graph(new_data)

        logging.debug("exiting data read")

    def data_process(self, _data):
        self.master.update_graph(_data)

    def stop_read(self):
        # self.usb.usb_write("AS7262|STOP")
        self.reading = False

    def read_once(self, graph, flash_on=False):
        logging.debug("read once")
        if flash_on:
            self.usb.usb_write("{0}|READ_SINGLE|FLASH".format(self.sensor_type))
        else:
            self.usb.usb_write("{0}|READ_SINGLE|NO_FLASH".format(self.sensor_type))
        # print(self.settings.integration_time)
        time.sleep(self.settings.integration_time/1000.+0.2)
        self.read_data(graph)

    def read_data(self, graph):
        data = self.usb.read_all_data()
        print(data)
        if data and (data[6] == 0):
            graph.update_data(data[:6])
        elif data and data[6] == 255:
            logging.info("sensor has problem, error byte set")
            messagebox.showerror("Error", "Error in getting data.  Please submit bug report")
        else:
            logging.info("device not working ch")
            self.master.device_not_working()

    def set_LED_power(self, LED_on):
        var_str = "OFF"
        if LED_on:
            var_str = "ON"
        self.usb.usb_write("{0}|LED_CTRL|{1}".format(self.sensor_type, var_str))

    def set_LED_power_level(self, power_level):
        self.usb.usb_write("{0}|POWER_LEVEL|{1}".format(self.sensor_type, power_level))


class ThreadedDataLoop(threading.Thread):
    def __init__(self, queue, event, flag):
        self.comm_queue = queue
        self.comm_event = event
        self.termination_flag = flag

    def run(self):
        while not self.termination_flag:
            new_data = self.comm_queue.get()
            self.master.update_graph(new_data)
        logging.debug("Ending threaded data loop")


class ConnectionStatusToplevel(tk.Toplevel):
    def __init__(self, master, status_str):
        tk.Toplevel.__init__(self)
        self.master = master  # type: main_gui_old.SpectrometerGUI
        self.status_str = status_str  # type: tk.StringVar
        self.geometry("300x300")
        self.attributes('-topmost', True)
        self.status_label = tk.Label(self, text=status_str.get())
        self.status_label.pack()
        tk.Button(self, text="Try to reconnect the device", command=self.reconnect).pack(side='top', pady=10)

    def reconnect(self):
        self.status_str.set("Reconnecting")
        reconnected = self.master.reconnect_device()

        if reconnected:
            self.status_str.set("Spectrometer: {0} connected".format(self.master.device.usb.spectrometer))
            time.sleep(0.2)
            self.destroy()
        else:
            self.status_str.set("Device not working correctly")
