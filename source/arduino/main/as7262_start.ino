

#include <AS726X.h>

//#include <SparkFun_AS7265X.h>
//
//#include "AS726X.h" //Click here to get the library: http://librarymanager/All#SparkFun_AS726X
//#include "SparkFun_AS7265X.h" //Click here to get the library: http://librarymanager/All#SparkFun_AS7265X
//#include <Wire.h>
//
//#define Serial SerialUSB
//#define print SerialUSB.print
//#define println SerialUSB.println
//
//char input;
//
//AS726X sensor1;
//AS7265X sensor2;
//
//// Parameters
//const int MUX_ADDR = 0x70;
//const int AS7262_CHANNEL = 7;
//const int AS7265X_CHANNEL = 6;
//const int MAX_CH_VALUE = 65000;
//
//float as7262_data[6];
//bool saturated;
//uint8_t as7262_integration_time = 255;
//uint8_t as7262_gain = 3;
//uint8_t as7265x_integration_time = 255;
//uint8_t as7265x_gain = 3;
//
//void setup() {
//  //sensor.begin();
//  Serial.begin(115200); // Initialize Serial Monitor USB
//  //Serial1.begin(9600); // Initialize hardware serial port, pins 0/1
//
//  while (!SerialUSB) ; // Wait for Serial monitor to open
//
//  // Send a welcome message to the serial monitor:
//  println("Send character(s) to relay it over Serial1");
//  Wire.begin();
//  enableMuxPort(AS7262_CHANNEL);
//  sensor1.begin(Wire, 3);
//  sensor1.setIntegrationTime(as7262_integration_time);
//  sensor1.setBulbCurrent(0b01);
//  disableMuxPort(AS7262_CHANNEL);
//
//  enableMuxPort(AS7265X_CHANNEL);
//  sensor2.begin(Wire);  // defualt gain is 64
//  sensor2.setIntegrationCycles(as7265x_integration_time);
//  sensor2.setBulbCurrent(AS7265X_LED_CURRENT_LIMIT_50MA, AS7265x_LED_WHITE);
//  sensor2.setBulbCurrent(AS7265X_LED_CURRENT_LIMIT_100MA, AS7265x_LED_IR);
//  sensor2.setBulbCurrent(AS7265X_LED_CURRENT_LIMIT_50MA, AS7265x_LED_UV);
//  sensor2.disableIndicator();
//  disableMuxPort(AS7265X_CHANNEL);
//  
//}
//
//void loop() {
//  if(Serial.available()){
//    input = Serial.read();
//    println(input);
//  }
//  switch (input) {
//    case 'I':
//      print("Arduino AS7262 sensor");
//      break;
//    case 'R':
//      // read AS7262
//      enableMuxPort(AS7262_CHANNEL);
//      sensor1.takeMeasurementsWithBulb();
//      get_as7262_data();
//      saturated = check_as7262_saturation();
//      if (saturated == 1) {
//        println("OK Read");
//      }
//      else {
//        println("Saturated Read");
//      }
//      disableMuxPort(AS7262_CHANNEL);
//      as7262_get_info();
//      println("DONE");
//      break;
//    case 'S':
//      // read triad
//      println("AS7265X Triad read");
//      enableMuxPort(AS7265X_CHANNEL);
//
//      run_as7265x();
//      saturated = check_as7265x_saturation();
//      if (saturated == 1) {
//        println("OK Read");
//      }
//      else {
//        println("Saturated Read");
//      }
//        disableMuxPort(AS7265X_CHANNEL);
//      break;
//      
//    case 'T':
//      // set integration time
//      while (!Serial.available()) ;
//      println("set integration");
//
//      char direction = Serial.read();
//      print(direction);
//      if (direction == '-') {
//        print("Decrement");
//        as7262_integration_time -= 50;
//      }
//      else if (direction == '+') {
//        if (as7262_integration_time <= 205) {
//          as7262_integration_time += 50;
//          print("Increment");
//        }
//      }
//      enableMuxPort(AS7262_CHANNEL);
//      sensor1.setIntegrationTime(as7262_integration_time);
//      disableMuxPort(AS7262_CHANNEL);
//      print(as7262_integration_time);
//      break;
//  }
//  input = 0;
//}
//
//
//void run_as7265x() {
//  for (int i=0; i<=2; i++) {
//    print("i = "); println(i);
//    sensor2.enableBulb(i);
//    sensor2.takeMeasurements();
//    sensor2.disableBulb(i);
//    get_as7265x_data();
//    saturated = check_as7262_saturation();
//    if (saturated == 1) {
//      println("OK Read");
//    }
//    else {
//      println("Saturated Read");
//    }
//    as7265x_get_info();
//    println("DONE");
//  }
//}
//
//
//void get_as7265x_data() {
//  print("Data: ");
//  print(sensor2.getCalibratedA()); print(", ");
//  print(sensor2.getCalibratedB()); print(", ");
//  print(sensor2.getCalibratedC()); print(", ");
//  print(sensor2.getCalibratedD()); print(", ");
//  print(sensor2.getCalibratedE()); print(", ");
//  print(sensor2.getCalibratedF()); print(", ");
//
//  print(sensor2.getCalibratedG()); print(", ");
//  print(sensor2.getCalibratedH()); print(", ");
//  print(sensor2.getCalibratedI()); print(", ");
//  print(sensor2.getCalibratedJ()); print(", ");
//  print(sensor2.getCalibratedK()); print(", ");
//  print(sensor2.getCalibratedL()); print(", ");
//
//  print(sensor2.getCalibratedR()); print(", ");
//  print(sensor2.getCalibratedS()); print(", ");
//  print(sensor2.getCalibratedT()); print(", ");
//  print(sensor2.getCalibratedU()); print(", ");
//  print(sensor2.getCalibratedV()); print(", ");
//  println(sensor2.getCalibratedW());
//}
//
//
//void get_as7262_data() {
//  print("Data: ");
//  print(sensor1.getCalibratedViolet(), 4); print(", ");
//  print(sensor1.getCalibratedBlue(), 4); print(", ");
//  print(sensor1.getCalibratedGreen(), 4); print(", ");
//  print(sensor1.getCalibratedYellow(), 4); print(", ");
//  print(sensor1.getCalibratedOrange(), 4); print(", ");
//  println(sensor1.getCalibratedRed(), 4); 
//}
//
//bool check_as7262_saturation() {
//  if (sensor1.getViolet() >= MAX_CH_VALUE) {
//    return false;
//  }
//  if (sensor1.getBlue() >= MAX_CH_VALUE) {
//    return false;
//  }
//  if (sensor1.getGreen() >= MAX_CH_VALUE) {
//    return false;
//  }
//  if (sensor1.getYellow() >= MAX_CH_VALUE) {
//    return false;
//  }
//  if (sensor1.getOrange() >= MAX_CH_VALUE) {
//    return false;
//  }
//  if (sensor1.getRed() >= MAX_CH_VALUE) {
//    return false;
//  }
//  return true;  
//}
//
//bool check_as7265x_saturation() {
//  if (sensor2.getA() >= MAX_CH_VALUE) {
//    return false;
//  }
//  if (sensor2.getB() >= MAX_CH_VALUE) {
//    return false;
//  }
//  if (sensor2.getC() >= MAX_CH_VALUE) {
//    return false;
//  }
//  if (sensor2.getD() >= MAX_CH_VALUE) {
//    return false;
//  }
//  if (sensor2.getE() >= MAX_CH_VALUE) {
//    return false;
//  }
//  if (sensor2.getF() >= MAX_CH_VALUE) {
//    return false;
//  }
//  if (sensor2.getG() >= MAX_CH_VALUE) {
//    return false;
//  }
//  if (sensor2.getH() >= MAX_CH_VALUE) {
//    return false;
//  }
//  if (sensor2.getI() >= MAX_CH_VALUE) {
//    return false;
//  }
//  if (sensor2.getJ() >= MAX_CH_VALUE) {
//    return false;
//  }
//  if (sensor2.getK() >= MAX_CH_VALUE) {
//    return false;
//  }
//  if (sensor2.getL() >= MAX_CH_VALUE) {
//    return false;
//  }
//  if (sensor2.getR() >= MAX_CH_VALUE) {
//    return false;
//  }
//  if (sensor2.getS() >= MAX_CH_VALUE) {
//    return false;
//  }
//  if (sensor2.getT() >= MAX_CH_VALUE) {
//    return false;
//  }
//  if (sensor2.getU() >= MAX_CH_VALUE) {
//    return false;
//  }
//  if (sensor2.getV() >= MAX_CH_VALUE) {
//    return false;
//  }
//  if (sensor2.getW() >= MAX_CH_VALUE) {
//    return false;
//  }
//  return true;  
//}
//
//
//void as7262_get_info() {
//  print("Gain: "); print(as7262_gain);
//  print("| Int time: "); println(as7262_integration_time);
//}
//
//
//void as7265x_get_info() {
//  print("Gain: "); print(as7265x_gain);
//  print("| Int time: "); println(as7265x_integration_time);
//}
// 
//
//
