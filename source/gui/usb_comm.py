# Copyright (c) 2017-2018 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

""" Classes to communication with a spectrophotometer"""

# standard libraries
from enum import Enum
import glob
import logging
import queue
import random
import serial
import struct
import sys
import threading
import time

import time
# installed libraries
import usb
import usb.core
import usb.util
import usb.backend

# local files
import psoc_spectrometers

PSOC_ID_MESSAGE = "PSoC-Spectrometer"
NO_SENSOR_ID_MESSAGE = "No Sensor"
AS7262_ID_MESSAGE = "AS7262"
AS7263_ID_MESSAGE = "AS7263"
BOTH__ID_MESSAGE = "Both AS726X"

END_CHAR = '\r'

USB_DATA_BYTE_SIZE = 40
IN_ENDPOINT = 0x81
OUT_ENDPOINT = 0x02

# Serial communication settings
BAUDRATE = 115200
# BAUDRATE = 114286
STOPBITS = serial.STOPBITS_ONE
PARITY = serial.PARITY_NONE
BYTESIZE = serial.EIGHTBITS

__author__ = 'Kyle Vitautas Lopin'


class DeviceTypes(Enum):
    usb = "USB"
    serial = "Serial"
    none = None


class PSoC_USB(object):
    def __init__(self, master, queue: queue.Queue, event: threading.Event(),
                 termination_flag: bool,
                 vendor_id=0x04B4, product_id=0x8051):
        self.master_device = master
        self.usb_device_found = False
        self.found = False
        self.device_type = None  # variable to display if a serial or usb protocol is to be used
        self.connected = False
        self.spectrometer = None
        self.received_message = "No device"
        self.device = self.connect_usb(vendor_id, product_id)
        # if the usb was not found
        if not self.usb_device_found:
            logging.info("No USB device find, looking for serial")
            self.device = self.connect_serial()

        if not self.device:
            # no device was found so just return and dont try a connection test
            return
        self.connection_test()
        # data_processing_function([1])
        # make an extra thread to poll the usb as this thread will hang on the timeouts of the usb
        self.data_ready_event = event
        self.data_queue = queue
        self.data_aquire_thread = ThreadedUSBDataCollector(self, master, self.data_queue,
                                                           self.data_ready_event,
                                                           termination_flag)

    def connect_usb(self, vendor_id, product_id):
        """
        Use the pyUSB module to find and set the configuration of a USB device

        This method uses the pyUSB module, see the tutorial example at:
        https://github.com/walac/pyusb/blob/master/docs/tutorial.rst
        for more details

        :param vendor_id: the USB vendor id, used to identify the proper device connected to
        the computer
        :param product_id: the USB product id
        :return: USB device that can use the pyUSB API if found, else returns None if not found
        """
        dev = usb.core.find(find_all=True)
        # for cfg in dev:
        #     print(cfg)
        device = usb.core.find(idVendor=vendor_id, idProduct=product_id)
        if device is None:
            logging.info("Device not found")
            return None
        else:  # device was found
            logging.info("PSoC found")
            self.usb_device_found = True
            self.found = True
            self.device_type = DeviceTypes.usb
            logging.info("Device type: {0}".format(self.device_type))

        # set the active configuration. the pyUSB module deals with the details
        device.set_configuration()
        return device

    def connection_test(self):
        """ Test if the device response correctly.  The device should return a message when
        given and identification call of 'I', and return the Spectrometer the device is connect to
        after given the command of 'i'.
        :return: True or False if the device is communicating correctly
        """
        # needed to make usb_write work, will be updated if not connected correctly
        self.connected = True

        # first test if the PSoC is connected correctly
        self.usb_write("ID")  # device should identify itself; only the first I is important
        received_message = self.usb_read_data(encoding='string')
        logging.debug('Received identifying message: {0}'.format(received_message))
        if received_message != PSOC_ID_MESSAGE:
            # set the connected state to false if the device is not working properly
            logging.debug("PSoC send wrong message")
            self.connected = False
            return
        logging.debug("PSoC send correct message")
        # test for the spectrometer if the PSoC is connected
        self.usb_write('ID-Spectrometer')  # device will return string of the spectrometer it is connected to
        received_message = self.usb_read_data(encoding='string')
        logging.debug('Received identifying message: {0}'.format(received_message))
        self.sensor_message = received_message
        if received_message == "Both AS726X":
            self.spectrometer = ["AS7262", "AS7263"]
        elif "No Sensor" in received_message:
            self.spectrometer = []
        else:
            self.spectrometer = [received_message[0:6]]

        logging.info("sensors attached: {0}".format(received_message))

    def connect_serial(self):
        available_ports = find_available_ports()
        logging.info("serial ports found: {0}".format(available_ports))
        for port in available_ports:
            try:

                logging.info("writing to port: {0}".format(port.port))
                device = serial.Serial(port.port, baudrate=BAUDRATE, stopbits=STOPBITS,
                                       parity=PARITY, bytesize=BYTESIZE, timeout=1)
                device.flushInput()
                device.flushOutput()
                sent_message = b"ID\r"
                device.write(sent_message)

                time.sleep(0.5)
                from_device = device.read_all().decode("utf-8")
                logging.info("From the device: {0}".format(from_device))
                if from_device == PSOC_ID_MESSAGE:
                    logging.info("Found serial device")
                    self.found = True
                    self.device_type = DeviceTypes.serial
                    self.connected = True
                    return device

                device.close()

            except Exception as exception:
                # not the correct device
                exc_type, exc_obj, exc_tb = sys.exc_info()
                logging.info("No device at port: {0}".format(port))
                logging.info("Exception: {0} at {1}".format(exception, exc_tb.tb_lineno))

    def usb_write(self, message, endpoint=OUT_ENDPOINT):
        # logging.debug("writing message: {0} with a device: {1}".format(message, self.device_type))
        if self.device_type == DeviceTypes.usb:
            self.usb_write_usb(message, endpoint=endpoint)
        elif self.device_type == DeviceTypes.serial:
            self.usb_write_serial(message)

    def usb_write_serial(self, message):
        if not self.device:
            logging.error("No device")
            return
        try:

            logging.debug("write to usb/uart: {0}{1}".format(message, END_CHAR))
            self.device.write("{0}{1}".format(message, END_CHAR).encode('utf-8'))
            while self.device.out_waiting != 0:
                print(self.device.out_waiting)

        except Exception as error:
            logging.error("USB writing error: {0}".format(error))
            self.connected = False

    def usb_write_usb(self, message, endpoint=OUT_ENDPOINT):
        time.sleep(0.1)
        if not self.device:
            logging.error("No device")
            return
        if not endpoint:
            endpoint = self.master_device.OUT_ENDPOINT

        try:
            logging.debug("write to usb: {0}".format(message))
            self.device.write(endpoint, message)

        except Exception as error:
            logging.error("USB writing error: {0}".format(error))
            # self.master_device.update_status("Device not connected")
            self.connected = False
            self.spectrometer = None
            self.device_type = None
            self.found = None

    def usb_read_info(self, info_endpoint=None, num_usb_bytes=None):
        if not info_endpoint:
            info_endpoint = self.master_device.INFO_IN_ENDPOINT
        if not num_usb_bytes:
            num_usb_bytes = self.master_device.USB_INFO_BYTE_SIZE

    def usb_read_data(self, num_bytes=USB_DATA_BYTE_SIZE,
                      endpoint=IN_ENDPOINT, encoding=None,
                      timeout=3000):
        if self.device_type == DeviceTypes.usb:
            return self.usb_read_data_usb(num_bytes, endpoint=endpoint,
                                          encoding=encoding, timeout=timeout)

        elif self.device_type == DeviceTypes.serial:
            return self.usb_read_data_serial(encoding=encoding)

    def usb_read_data_serial(self, encoding=None, timeout=3000):
        if not self.connected:
            logging.info("not working")
            return None
        try:
            # usb_input = self.device.readline(timeout=1.0)  # TODO fix this
            time.sleep(0.3)
            usb_input = self.device.read_all()
            logging.debug(usb_input)
        except Exception as error:
            logging.error("Failed data read")
            return None
        if encoding == 'uint16':
            pass
            # return convert_uint8_uint16(usb_input)
        elif encoding == "float32":
            if (len(usb_input)) == 25:
                # return struct.iter_unpack('f', usb_input)
                return struct.unpack('>ffffffB', usb_input)
            else:
                logging.error("Error in reading")
        elif encoding == 'string':
            # change bytes to string and change to regular string not byte string
            return usb_input.decode("utf-8")
        else:  # no encoding so just return raw data
            return usb_input

    def usb_read_data_usb(self, num_bytes=USB_DATA_BYTE_SIZE,
                          endpoint=IN_ENDPOINT, encoding=None,
                          timeout=3000):
        """ Read data from the usb and return it, if the read fails, log the miss and return None
        :param num_usb_bytes: number of bytes to read
        :param endpoint: hexidecimal of endpoint to read, has to be formatted as 0x8n where
        n is the hex of the encpoint number
        :param encoding: string ['uint16', 'signed int16', or 'string] what data format to return
        the usb data in
        :return: array of the bytes read
        """
        if not self.connected:
            logging.info("not working")
            return None
        try:
            usb_input = self.device.read(endpoint, num_bytes, timeout)  # TODO fix this
            logging.debug(usb_input)
        except Exception as error:
            logging.error("Failed data read")
            logging.error("No IN ENDPOINT: %s", error)
            return None
        if encoding == 'uint16':
            pass
            # return convert_uint8_uint16(usb_input)
        elif encoding == "float32":
            if (len(usb_input) % 4) == 0:
                # return struct.iter_unpack('f', usb_input)
                return struct.unpack('>ffffffB', usb_input)
            else:
                logging.error("Error in reading")
        elif encoding == 'string':
            # change bytes to string and change to regular string not byte string
            return usb_input.tostring().decode("utf-8")
        else:  # no encoding so just return raw data
            return usb_input

    def read_all_data(self):
        # mock a data read now
        return self.usb_read_data(num_bytes=24, encoding="float32")

    def start_reading(self):
        pass


class ThreadedUSBDataCollector(threading.Thread):
    def __init__(self, device: PSoC_USB, master: 'psoc_spectrometers.BaseSpectrometer()',
                 data_queue: queue.Queue, data_event: threading.Event,
                 function_call=None, termination_flag=False):
        threading.Thread.__init__(self)
        self.device = device
        self.master = master
        self.data_queue = data_queue
        self.data_event = data_event
        # self.func_call = function_call
        self.running = True  # bool: Flag to know when the data read should stop
        self.termination_flag = termination_flag  # Flag to know when the thread should stop

    def run(self):
        """ Poll the USB for new data packets and pass it through the queue and set the event flag so the main
        program will update the graph.  Structure the program so if the termination flag is set, collect the last
        data point to clear the data from the sensor."""

        while not self.termination_flag:
            logging.debug("In data collection thread")
            if self.running:
                data = self.device.read_all_data()
                logging.debug("Got data: {0}".format(data))
                # make sure there is data and that the previous data has been processed
                # if data and self.data_queue.empty():
                #     self.data_queue.put(data)
                #     self.data_event.set()
                if data:
                    # self.func_call(data)
                    self.master.data_processing(data)
                elif not data:
                    self.termination_flag = True
                    logging.error("=================== Device not working =====================")
            else:  # the main program wants to stop the program
                self.termination_flag = True
        logging.debug("exiting data thread call: {0}".format(self.termination_flag, hex(id(self.termination_flag))))
        # print(hex(id(self.termination_flag)))
        self.data_event.set()  # let the main program exit the data_read wait loop

    def stop_running(self):
        logging.debug("Stopping data stream")
        self.running = False

    def single_run(self):
        pass


def find_available_ports():

    # taken from http://stackoverflow.com/questions/12090503/listing-available-com-ports-with-python

    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i+1) for i in range(32)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    available_ports = []
    for port in ports:
        try:
            device = serial.Serial(port=port, write_timeout=0.5,
                                   inter_byte_timeout=1, baudrate=115200,
                                   parity=serial.PARITY_EVEN, stopbits=1)
            device.close()
            available_ports.append(device)
        except (OSError, serial.SerialException):
            pass
    return available_ports
