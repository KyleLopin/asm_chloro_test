# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Class to interact with an arduino
"""

__author__ = "Kyle Vitatus Lopin"
# installed libraries
import queue
import serial  # pyserial
import serial.tools.list_ports
import threading
import time
# local files
import AS726XX

# USB-UART Constants
DESCRIPTOR_NAME_WIN1 = "USB Serial Port"
DESCRIPTOR_NAME_WIN2 = "USB Serial Device"
DESCRIPTOR_NAME_MAC1 = "FT230X Basic UART"
DESCRIPTOR_NAME_MAC2 = "Expansion3"
DESCRIPTOR_NAME_MAC3 = "RedBoard Turbo"
DESCRIPTOR_NAME_MAC4 = "USB2.0-Serial"
DESCRIPTOR_NAME_ARTEMIS = "USB-SERIAL CH340"
BAUD_RATE = 115200
DESCRIPTOR_NAMES = [DESCRIPTOR_NAME_WIN1, DESCRIPTOR_NAME_WIN2,
                    DESCRIPTOR_NAME_MAC1, DESCRIPTOR_NAME_MAC2,
                    DESCRIPTOR_NAME_MAC3, DESCRIPTOR_NAME_MAC4,
                    DESCRIPTOR_NAME_ARTEMIS]

ID_NAME = b"Naresuan Color Sensor Setup"

# class Arduino_old:
#     def __init__(self):
#         self.device = self.auto_find_com_port()
#
#     @staticmethod
#     def auto_find_com_port():
#         available_ports = serial.tools.list_ports
#         for port in available_ports.comports():  # type: serial.Serial
#             print(port.device)
#             print('name: ', port.name)
#             print('desc:', port.description)
#             print(DESCRIPTOR_NAMES)
#             for name in DESCRIPTOR_NAMES:
#                 if name in port.description:
#                     print("found name in device")
#                 # if True:
#                     try:
#                         device = serial.Serial(port.device, baudrate=BAUD_RATE,
#                                                timeout=1)
#                         device.readline()  # clear anything it may respond with first
#                         device.write(b"Id")
#                         for i in range(5):
#                             input = device.readline()
#                             # it should respond with correct ID but may take a few lines
#                             if ID_NAME in input:
#                                 print("Found device")
#                                 return device  # a device could connect without an error so return
#
#                     except Exception as error:  # didn't work so try other ports
#                         print("Port access error: ", error)
#
#     def read_all(self):
#         try:
#             data_packet = self.device.readall()
#             data_packets = data_packet.split(b'\r\n')
#             return data_packets
#         except:
#             print("Error Reading")
#
#     def is_connected(self):
#         return self.device
#
#     def read_as7262(self):
#         self.write("R")
#         return(self.read_data(b"AS7262"))
#
#     def read_data(self, type):
#         data = b""
#         print('====================START')
#         while b"DONE" not in data:
#             data = self.device.readline()
#             print('data: ', data)
#             if b"Data:" in data:
#                 data_pkt = AS726XRead(type, self.sort_index)
#                 cal_data = data.split(b':')[1].split(b',')
#                 cal_data = [float(x) for x in cal_data]
#                 data_pkt.add_cal_data(cal_data)
#
#             elif b"Gain:" in data:
#                 gain_data_pre = data.split(b':')
#                 gain = int(gain_data_pre[1].split(b'|')[0])
#                 int_time = int(gain_data_pre[2].split(b'\r')[0])
#                 data_pkt.add_gain_n_integration(gain, int_time)
#         print('====================END')
#         return data_pkt
#
#     def write(self, message):
#         print('=======-----')
#         if type(message) is str:
#             message = message.encode()
#         print('writing message: ', message)
#         self.device.write(message)
#
#     def read_package(self, command):
#         self.write(command)
#         end_line = b"End " + command
#         input_line = b""
#         input = []
#         while end_line not in input_line:
#             input_line = self.device.readline()
#
#             input.append(input_line)
#         return input


class Arduino(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.device = self.auto_find_com_port()
        self.running = False
        self.output = queue.Queue()
        self.command = None
        self.package = []

    @staticmethod
    def auto_find_com_port():
        available_ports = serial.tools.list_ports
        for port in available_ports.comports():  # type: serial.Serial
            # print(port.device)
            # print('name:', port.name)
            # print('desc:', port.description)
            # print(DESCRIPTOR_NAMES)
            for name in DESCRIPTOR_NAMES:
                # print('name: {0}|{1}'.format(name, port.description))
                if name in port.description:
                    print('----')
                    try:
                        device = serial.Serial(port.device, baudrate=BAUD_RATE,
                                               timeout=1)
                        device.readline()  # clear anything it may respond with first
                        device.write(b"Id")
                        for i in range(10):
                            input = device.readline()
                            # it should respond with correct ID but may take a few lines
                            print("Got input: ", input)
                            if ID_NAME in input:
                                print("Found device")
                                return device  # a device could connect without an error so return

                    except Exception as error:  # didn't work so try other ports
                        print("Port access error: ", error)

    def run(self):
        self.running = True

        if not self.device:  # not device so return
            return None
        while self.running:
            # print('run loop')
            time.sleep(.05)  # ease the load on the computer
            if not self.output.empty():
                self.device.write(self.output.get())
            if self.device.in_waiting:
                # everything could be \r\n terminated
                data_line = self.device.readline().strip(b'\r\n')
                print(data_line)
                self.parse_input(data_line)
                # self.parse_package(data_line)

    def parse_input(self, dataline):
        """ Place holder, this should be overwritten in implementation """
        # a command has already been received so collect the data
        if b"End" in dataline:
            end_command = dataline.split(b' ')[1].split(b'\r\n')[0]
            print('end command: ', end_command, self.command)
            if end_command == self.command:
                # correct information
                self.parse_package(end_command, self.package)
                self.command = None
                self.package = []
            else:
                print("Error, ended the wrong command ", end_command, self.command)

        elif self.command:
            # print('have command: ', self.command)
            self.package.append(dataline)

        elif b"Starting" in dataline:
            self.command = dataline.split(b' ')[1].split(b'\r\n')[0]
            print('Got command: ', self.command)

    def write(self, message):
        if type(message) is str:
            message = message.encode()
        self.output.put(message)


class ArduinoColorSensors(Arduino):
    def __init__(self, master):
        print('check')
        Arduino.__init__(self)

        self.start()
        self.master = master
        self.graph_event = threading.Event()
        self.graph_queue = queue.Queue()
        self.sensors = []
        self.port_list = dict()
        print(self.device)
        if not self.device:
            return
        self.write(b"Setup")
        self.starting_up = True
        self.sensor = None
        self.new_data = False

    def parse_package(self, command, package):
        print("parse: ", command)
        print(package)
        if command == b'Setup':
            print("Setting up1")
            self.setup(package)
        elif command == b"Data":
            if self.new_data:
                self.graph_queue.put([self.sensor, "Clear"])
                self.graph_event.set()
                self.new_data = False
            print(self.sensors)
            print(self.port_list)
            self.data_read(package)

        elif command == b"Inc":
            print("inc read")
            self.sensor.increase_read_num()
            self.new_data = True

        else:
            print('===== NOT RECOGNIZED COMMAND ======')

    def setup(self, packet):
        print("Setting up2")
        for line in packet:
            print(line)
            has_button = True
            if b"No button attached" in line:
                has_button = False

            port = None
            if b"to port:" in line:
                port = int(line.split(b":")[1].split(b'|')[0])
            if b"AS7262 device attached" in line:
                self.sensors.append(AS726XX.AS7262(self, has_button, port))
                self.port_list[port] = self.sensors[-1]
            elif b"AS7263 device attached" in line:
                self.sensors.append(AS726XX.AS7263(self, has_button, port))
                self.port_list[port] = self.sensors[-1]
            elif b"AS7265x device attached" in line:
                self.sensors.append(AS726XX.AS7265x(self, has_button, port))
                self.port_list[port] = self.sensors[-1]
        for sensor in self.sensors:
            print(sensor)
        print(self.port_list)
        self.starting_up = False

    def data_read(self, package):
        sensor = None  #  set scope
        for line in package:
            print('data line: ', line)
            if b"Reading port" in line:
                port_number = int(line.split(b': ')[1])
                print('port number: ', port_number)
                print(self.port_list[port_number])
                self.sensor = self.port_list[port_number]  # type: AS726XX.AS7262
            elif b"Integration time" in line:
                integration_time = int(line.split(b": ")[1])
                print('integration time: ', integration_time)
                self.sensor.data.set_int_cylces(integration_time)
            elif b"LED current" in line:
                led_current = int(line.split(b": ")[1])
                print('led current: ', led_current)
                self.sensor.data.set_led_current(led_current)
            elif b"Data:" in line:
                data = line.split(b': ')[1].split(b',')
                data = [float(x) for x in data]
                print(data)
                self.sensor.data.set_data(data)
            elif b"LED:" in line:
                led = line.split(b": ")[1]
                print('led: ', led)
                self.sensor.data.set_LED(led)

            elif b'OK Read' in line or b"Saturated Read" in line:
                saturation = line
                # last line to recieve so save the data and update graph
                self.sensor.data.save_data(saturation)
                self.graph_queue.put([self.sensor, saturation])
                self.graph_event.set()
        # print('check222')
        # sensor.increase_read_num()

    def run_command(self, command):
        self.current_command = command
        self.write('Start: {0}'.format(command))

