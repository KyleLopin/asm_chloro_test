#ifndef GLOBALS_H
#define GLOBALS_H


#define print Serial.print
#define println Serial.println
#include <SparkFun_AS7265X.h>
#include <AS726X.h>
#include <SparkFun_Qwiic_Button.h>


#include "AS726X.h" //Click here to get the library: http://librarymanager/All#SparkFun_AS726X
//#include "AS726X.cpp"
#include "SparkFun_AS7265X.h" //Click here to get the library: http://librarymanager/All#SparkFun_AS7265X
#undef private

#include <Wire.h>
//#include "mux_commands.h"
#define AS726X_ADDR 0x49
#define BUTTON_ADDR 0x6F

#define AS7262_CODE 0x3E
#define AS7263_CODE 0x3F
#define AS7265X_CODE 0x41

String input = "";

AS726X as726x;
AS7265X as7265x;
QwiicButton button;

// Parameters
const int MAX_CH_VALUE = 65000;
const byte NO_SENSOR = 0;
const byte AS7262_SENSOR = 1;
const byte AS7263_SENSOR = 2;
const byte AS7265X_SENSOR = 3;
byte sensor_type[8];

//float as7262_data[6];
bool saturated;

boolean use_mux = false;
boolean single_sensor = false;

typedef struct {
  byte led_current = 0;
  byte led_to_use = 0;
  byte int_time = 250;
  byte gain = 1;
  bool use_led = true;
  
} as726x_params;


#endif
