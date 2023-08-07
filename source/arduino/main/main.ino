
//#define private public  // We want access to a few private methods

//#define Serial SerialUSB
//#define print SerialUSB.print
//#define println SerialUSB.println

//#define print Serial.print
//#define println Serial.println
//#include <SparkFun_AS7265X.h>
//#include <AS726X.h>
//#include <SparkFun_Qwiic_Button.h>
//
//
//#include "AS726X.h" //Click here to get the library: http://librarymanager/All#SparkFun_AS726X
////#include "AS726X.cpp"
//#include "SparkFun_AS7265X.h" //Click here to get the library: http://librarymanager/All#SparkFun_AS7265X
//#undef private
//
//#include <Wire.h>
////#include "mux_commands.h"
//#define AS726X_ADDR 0x49
//#define BUTTON_ADDR 0x6F
//
//#define AS7262_CODE 0x3E
//#define AS7263_CODE 0x3F
//#define AS7265X_CODE 0x41
//
//String input = "";
//
//AS726X as726x;
//AS7265X as7265x;
//QwiicButton button;
//
//// Parameters
//const int MAX_CH_VALUE = 65000;
//const byte NO_SENSOR = 0;
//const byte AS7262_SENSOR = 1;
//const byte AS7263_SENSOR = 2;
//const byte AS7265X_SENSOR = 3;
//byte sensor_type[8];
//
////float as7262_data[6];
//bool saturated;
//
//boolean use_mux = false;
//boolean single_sensor = false;

#include "globals.h"


void setup() {
  Serial.begin(115200); // Initialize Serial Monitor USB

  while (!Serial) ; // Wait for Serial monitor to open

  // Send a welcome message to the serial monitor:
  println("Send character(s) to relay it over Serial1");
  Wire.begin(); 
  button.begin();
  print("Use mux: "); println(use_mux);
  println(false);
  if (check_mux()) {
    println("Has mux");
    use_mux = true;
    for (int i=0; i <= 7; i++) {
        enableMuxPort(i);
        if (button.isConnected()) {
          button.clearEventBits();
        }
    }
  }  // else use_mux is already false
  print("Use mux: "); println(use_mux);
  println(false, true);
//  print("Use mux: "); println(use_mux);
}


void loop() {
//  println("loop");
//  Wire.requestFrom(0x49, 1);
//  print("check:"); println(Wire.available());
  if(Serial.available()){
    input = Serial.readString();
    print("got input: "); println(input);
  }
//  println("loop2");
  if (input == "Setup") {
    println("Starting Setup");
    get_sensor_info();
  }
  else if (input == "Id") {
    println("Naresuan Color Sensor Setup");
  }
  else if (input.startsWith("Read:")) {
    int mux_num = input.charAt(5) - '0';
    print("reading mux: "); println(mux_num);
    enableMuxPort(mux_num);
    read_sensor(mux_num);
    disableMuxPort(mux_num);
  }
  else if (input.startsWith("Indicator LED on:")) {
    turn_indicator_on(input);
  }
  else if (input.startsWith("No Indicator:")) {
    turn_indicator_off(input);
  }
  else if (input.startsWith("Button LED on:")) {
    turn_button_on(input);
  }
  else if (input.startsWith("Button LED off:")) {
    turn_button_off(input);
  }
  else if (input == "") {
    // pass on blank line
  }
  else {
    println("Not valid input:");
    println(input);
  }
//  println("loop3");
  delay(10);
//  println("loop3b");
  input = "";
  if (use_mux == true) {
    for (int i=0; i <= 7; i++) {
//      print("Mux cycle:"); println(i);
      enableMuxPort(i);
      if (button.isConnected()) {
        if (button.hasBeenClicked()) {
          button.LEDon(50);
//          print(sensor_type[i]);
          if (sensor_type[i] == AS7262_SENSOR) {
            println("Sensor AS7262");
            run_as726x_scan(i);
          }
          else if (sensor_type[i] == AS7263_SENSOR) {
            println("Sensor AS7263");
            run_as726x_scan(i);
          }
          else if (sensor_type[i] == AS7265X_SENSOR) {
            println("Sensor AS7265x");
            run_as7265x_scan(i);
          }
          button.LEDoff();
          // Don't let user click button during a read
          button.clearEventBits(); 
        }
      }
    disableMuxPort(i);
    }
  }
  else {
//    println("Check ppo");
    if (button.isConnected()) {
      if (button.hasBeenClicked()) {
//        println("Check ppo2");
        button.clearEventBits();
        if (sensor_type[0] == AS7262_SENSOR) {
          println("Sensor AS7262");
          run_as726x_scan(0); // no mux so used 0 as port number as placeholder
        }
        else if (sensor_type[0] == AS7263_SENSOR) {
          println("Sensor AS7263");
          run_as726x_scan(0); // no mux so used 0 as port number as placeholder
        }
        else if (sensor_type[0] == AS7265X_SENSOR) {
          println("Sensor AS7265x");
          run_as7265x_scan(0); // no mux so used 0 as port number as placeholder
        }
      }
    }
  }
}

void read_sensor(int channel) {
  if (sensor_type[channel] == AS7262_SENSOR) {
    println("Sensor AS7262");
    run_as726x_scan(channel);
  }
  if (sensor_type[channel] == AS7263_SENSOR) {
    println("Sensor AS7263");
    run_as726x_scan(channel);
  }
  else if (sensor_type[channel] == AS7265X_SENSOR) {
    println("Sensor AS7265x");
    run_as7265x_scan(channel);
  }
  else {
    println("No sensor on this channel");
  }
}

void run_as726x_scan(int channel) {
  for (int j=50; j <= 250; j += 50) {
    for (int k=0; k <=3; k++) {
      println("Starting Data Read");
      print("Reading port: "); println(channel);
      print("Integration time: "); println(j);
      print("LED current: "); println(k);
      as726x.setBulbCurrent(k);
      as726x.setIntegrationTime(j);
      run_as726x();
      println("End Data Read");
    }
  }
  println("Starting Inc read");
  println("End Inc read");
}

void run_as7265x_scan(int channel) {
  for (int j=50; j <= 150; j += 50) {
    for (int k=0; k <=3; k++) {
      int k_max = 6;
      if (k >= 3) {
        k_max = 2;
      }
      println("Starting Data Read");
      print("Reading port: "); println(channel);
      print("Integration time: "); println(j);
      print("LED current: "); println(k);
      as7265x.setBulbCurrent(k, AS7265x_LED_WHITE);
      as7265x.setBulbCurrent(k, AS7265x_LED_IR);
      as7265x.setBulbCurrent(k, AS7265x_LED_UV);
      as7265x.setIntegrationCycles(j);
      run_as7265x(k_max);
      println("End Data Read");
    }
  }
  println("Starting Inc read");
  println("End Inc read");
}
