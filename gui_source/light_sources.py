# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

# standard libraries
from collections import OrderedDict
import tkinter as tk
# local files
import main_wipy

__author__ = "Kyle Vitatus Lopin"

LP55231_LEDS_RIGHT = [390, 395, 400, 405, 410, 425, 525, 890, '----']
LP55231_RIGHT_ADDR = 0x34
LP55231_LEDS_LEFT = [455, 475, 480, 465, 470, 505, 630, '----', 940]
LP55231_LEFT_ADDR = 0x33

LP55231_START_LED_PWM_REG_ADDR = 0x16
LP55231_START_LED_CURRENT_REG_ADDR = 0x26

USE_SINGLE_LED = main_wipy.USE_SINGLE_LED
USE_MULTIPLE_LEDS = main_wipy.USE_MULTIPLE_LEDS


class LEDFrame(tk.Frame):
    def __init__(self, master):
        self.use_multiple_leds = master.use_multiple_leds
        if self.use_multiple_leds == USE_MULTIPLE_LEDS:
            pass
        elif self.use_multiple_leds == USE_SINGLE_LED:
            pass


class SingleLEDFrame(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Label(self, text="LED Source:").pack(side='top', padx=5, pady=5)
        self.lights = make_light_sources()
        print(self.lights)

        self._led_choices = tk.StringVar(self)
        self._led_choices.set("None")
        tk.OptionMenu(self, self._led_choices, *self.lights).pack(side='top', padx=5, pady=5)

    def get(self):
        _led_choice = self._led_choices.get()
        print('led choice: ', _led_choice, type(_led_choice))
        if _led_choice == "None":
            return "None", [], []
        _onboard_led, lp55231_leds = self.lights[_led_choice].get_single_led_str()

        # return _led_choice, self.lights[_led_choice]
        return _led_choice, _onboard_led, lp55231_leds


class MultiLEDFrame(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Label(self, text="Add LED Source:").pack(side='top', padx=5, pady=5)
        self.lights = make_light_sources()
        print(self.lights)

        self._led_choices = tk.StringVar(self)
        self._led_choices.set("None")
        tk.OptionMenu(self, self._led_choices, *self.lights,
                      command=self.add_led).pack(side='top', padx=5, pady=5)

        tk.Label(self, text="LED Sources added:\n(click to delete)").pack(side='top', padx=5, pady=5)
        self.added_leds_frame = tk.Frame(self)
        self.added_leds = []
        self.added_leds_frame.pack(side='top')

    def add_led(self, option):
        print('add led: ', option)
        new_led = tk.Label(self.added_leds_frame, text=option)
        new_led.pack(side='top')
        new_led.bind('<Button-1>', lambda event, _new_led=new_led: self.delete_led(_new_led))
        self.added_leds.append(new_led)

    def delete_led(self, _led):
        print('delete: ', _led)
        self.added_leds.remove(_led)
        _led.destroy()
        print(self.added_leds)

    def get(self):
        _led_names = []
        _onboard_leds = []
        _lp55231_leds = []
        for led in self.added_leds:
            _led_names.append(led['text'])
            _list1, _list2 = self.lights[_led_names[-1]].get_single_led_str()
            _onboard_leds.extend(_list1)
            _lp55231_leds.extend(_list2)
        print(_led_names, _onboard_leds, _lp55231_leds)
        return _led_names, _onboard_leds, _lp55231_leds


def make_light_sources():
    ordered_dict = OrderedDict()
    ordered_dict["None"] = None
    ordered_dict["White LED"] = Light("AS726X", device=0)
    ordered_dict["IR LED"] = Light("AS726X", device=1)
    ordered_dict["UV (405 nm) LED"] = Light("AS726X", device=2)

    _dict = dict()

    for i, LED in enumerate(LP55231_LEDS_RIGHT):
        key = "{0} nm LED".format(LED)
        _dict[key] = Light("LP55231", channel=i+9, i2c_adress=LP55231_RIGHT_ADDR)

    for i, LED in enumerate(LP55231_LEDS_LEFT):
        key = "{0} nm LED".format(LED)
        _dict[key] = Light("LP55231", channel=i, i2c_adress=LP55231_LEFT_ADDR)

    for key, value in sorted(_dict.items()):
        if key != '---- nm LED':
            ordered_dict[key] = value

    return ordered_dict


class Light:
    def __init__(self, type, i2c_adress=None, channel=None, device=None):
        self.type = type
        if type == "AS726X":
            self.device = device
        elif type == "LP55231":
            self.i2c_addr = i2c_adress
            self.channel = channel
        else:
            raise ValueError("only types of AS726X or LP55231 are accepted")

    def turn_on(self):
        if self.type == "AS726X":
            msg = "as7265x.enable_led({0})".format(self.device)
        elif type == "LP55231":
            msg = "i2c_2_write8({0}, {1}, {2}".format(self.i2c_addr,
                                                      hex(self.channel +
                                                          LP55231_START_LED_PWM_REG_ADDR),
                                                      0xFF)
            print(msg)

    def get_single_led_str(self):
        if self.type == "AS726X":
            # print("([{0}], [])".format(self.device))
            return [self.device], []
        elif self.type == "LP55231":
            return [], [self.channel]


