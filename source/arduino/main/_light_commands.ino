
//#include <SparkFun_Qwiic_Button.h>
//
//extern AS726X as726x;
//extern QwiicButton button;
#include "globals.h"

void turn_indicator_on(String input_cmd) {
  int mux_num = input_cmd.charAt(17) - '0';
  enableMuxPort(mux_num);
  as726x.enableIndicator();  // even if sensor is as7265x, this will work
  disableMuxPort(mux_num);
}


void turn_indicator_off(String input_cmd) {
  int mux_num = input_cmd.charAt(13) - '0';
  enableMuxPort(mux_num);
  as726x.disableIndicator();  // even if sensor is as7265x, this will work
  if (button.isConnected()) {
    button.LEDoff();
  }
  disableMuxPort(mux_num);
}

void turn_button_on(String input_cmd) {
  int mux_num = input.charAt(14) - '0';
  enableMuxPort(mux_num);
  button.LEDon(100);  // even if sensor is as7265x, this will work
  disableMuxPort(mux_num);
}

void turn_button_off(String input_cmd) {
  int mux_num = input.charAt(14) - '0';
  enableMuxPort(mux_num);
  button.LEDoff();  // even if sensor is as7265x, this will work
  disableMuxPort(mux_num);
}
