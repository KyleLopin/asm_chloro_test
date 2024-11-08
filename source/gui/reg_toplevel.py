""" Top level to read and write registers of an AS726X that is connect to a PSoC device """

# standard libraries
import tkinter as tk

# installed libraries
# local files
# import main_gui_old  # for type hinting


__author__ = 'Kyle Vitautas Lopin'


REGISTERS = {"HW_VERSION1": "0x00",
             "HW_VERSION2": "0x01",
             "FW_VERSION1": "0x02",
             "FW_VERSION2": "0x03",
             "Control\Setup": "0x04",
             "Integration Time": "0x05",
             "Device Temp": "0x06",
             "LED Control": "0x07",
             "V_High": "0x08",
             "V Low": "0x09",
             "V_Cal 1": "0x14",
             "V_Cal 2": "0x15",
             "V_Cal 3": "0x16",
             "V_Cal 4": "0x17",
             "R_High": "0x12",
             "R Low": "0x13",
             "R_Cal 1": "0x28",
             "R_Cal 2": "0x29",
             "R_Cal 3": "0x2A",
             "R_Cal 4": "0x2B"}


class RegDebugger(tk.Toplevel):
    def __init__(self, master, device):  # masters is hacked to be the settings
        tk.Toplevel.__init__(self, master=master)
        self.device = device.usb
        self.geometry('400x300')
        self.title("Register Debug")

        self.sensor_type = tk.StringVar()
        type_frame = tk.Frame(self).pack(side=tk.TOP)
        tk.Radiobutton(type_frame, text="AS7262", variable=self.sensor_type, value="AS7262").pack(side=tk.LEFT)
        tk.Radiobutton(type_frame, text="AS7263", variable=self.sensor_type, value="AS7263").pack(side=tk.LEFT)

        tk.Label(self, text="Write register:").pack(side='top')
        self.reg_number = tk.StringVar()
        tk.OptionMenu(self, self.reg_number, *REGISTERS.keys()).pack(side='top')
        tk.Label(self, text="the value ix hex:").pack(side='top')

        foo_frame = tk.Frame(self)
        foo_frame.pack(side='top')
        tk.Label(foo_frame, text="0x").pack(side='left')
        self.reg_value = tk.StringVar()
        tk.Entry(foo_frame, textvariable=self.reg_value).pack(side='left')

        tk.Button(self, text="Write Register", command=self.write_reg).pack(side='top')

        tk.Label(self, text="").pack(side='top')
        tk.Label(self, text="Read Register:").pack(side='top')

        self.read_reg = tk.StringVar()

        tk.OptionMenu(self, self.read_reg, *REGISTERS.keys()).pack(side='top')
        tk.Button(self, text="Read", command=self.read_reg_call).pack(side='top')
        self.reg_read_value = tk.StringVar()
        self.reg_read_value.set("Register value: 0x")
        tk.Label(self, textvariable=self.reg_read_value).pack(side='top')

    def write_reg(self):
        if not self.reg_number.get():
            return
        self.device.usb_write("{0}|REG_WRITE|{1}|0x{2}".format(self.sensor_type.get(),
                                                                  REGISTERS[self.reg_number.get()],
                                                                  self.reg_value.get()))

    def read_reg_call(self):
        self.device.usb_write("{0}|REG_READ|{1}".format(self.sensor_type.get(),
                                                           REGISTERS[self.read_reg.get()]))
        return_str = self.device.usb_read_data(encoding="string")
        # print("Got back reg value: {:02X}".format(reg_value))
        self.reg_read_value.set(return_str[10:])
