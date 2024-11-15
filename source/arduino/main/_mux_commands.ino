
#include <Wire.h>
extern QwiicButton button;

const int MUX_ADDR = 0x70;  
#define AS726X_ADDR 0x49

bool check_mux() {
//  Wire.requestFrom(MUX_ADDR, 1);
//  return Wire.available();
  Wire.beginTransmission(MUX_ADDR);
  byte end_trans = Wire.endTransmission();
  print("end Trans: ");  println(end_trans);
  if (end_trans == 0) {  // Successful connection 
    return true;
  }
  return false;  // else not connected

  Wire.beginTransmission(AS726X_ADDR);
  end_trans = Wire.endTransmission();
  if (end_trans == 0) {  // There is an AS726x or AS7265x, no mux
    println("Has sensor and no mux");
    return false;
  }
}

void get_sensor_info() {
  if (use_mux) {
    println("Have mux");
    for (int i=0; i <= 7; i++) {
      enableMuxPort(i);
      int avail = check_channel(AS726X_ADDR);
      print("port: "); println(i);
      print("avail: "); println(avail);
//      Wire.requestFrom(0x49, 1);
//      print("check:"); println(Wire.available());
      if (avail) {
        println("check get sensor info1");
        sensor_type[i] = get_sensor_type(i);
      }
      println("check get sensor info2");
      disableMuxPort(i);
    }
  }
  else {
    println("No mux");
    int avail = check_channel(AS726X_ADDR);
//    print("check channel:"); println(avail);
    if (avail) {
      single_sensor = true;
      sensor_type[0] = get_sensor_type(0);  // just put in a place holder for the channel
    }
  }
  println("End Setup");
}

bool check_channel(int address){
//  Wire.requestFrom(0x49, 1);
//  println(Wire.available());
//  Wire.requestFrom(AS726X_ADDR, 1);
//  available_channel = Wire.available();
  Wire.beginTransmission(address);
  int end_trans = Wire.endTransmission();
  print("end Trans AS7262: ");  println(end_trans);
  if (end_trans == 0) {
    return true;
  }
  return false;
}

byte get_sensor_type(int channel) {
  println("get senor type");
  byte _sensor_type = NO_SENSOR;
  if (check_channel(AS726X_ADDR) ) {
    as726x.begin(Wire);
    print("Wire on");
    as726x.setIntegrationTime(150);
    as726x.setBulbCurrent(0b11);
  }
  print("No sensor");
  uint8_t hw_type = as726x.getVersion();
  print("Hardware type: 0x");
  println(hw_type, HEX);

  if (hw_type == AS7262_CODE) {
    _sensor_type = AS7262_SENSOR;
    print("AS7262 device attached to port: "); 
    print(channel); print("|  ");
    
  }
  else if (hw_type == AS7263_CODE) {
    _sensor_type = AS7263_SENSOR;
    print("AS7263 device attached to port: "); 
    print(channel); print("|  ");
  }
  else if (hw_type == AS7265X_CODE) {
    _sensor_type = AS7265X_SENSOR;
    as7265x.begin();
    as7265x.setBulbCurrent(AS7265X_LED_CURRENT_LIMIT_100MA, AS7265x_LED_WHITE);
    as7265x.setBulbCurrent(AS7265X_LED_CURRENT_LIMIT_100MA, AS7265x_LED_IR);
    as7265x.setBulbCurrent(AS7265X_LED_CURRENT_LIMIT_100MA, AS7265x_LED_UV);
    as7265x.disableIndicator();
    as7265x.setIntegrationCycles(150);
    print("AS7265x device attached to port: "); 
    print(channel); print("|  ");
  }
  if (button.isConnected() == false) {
    println("No button attached to device.");
  }
  else {
    println("Button attached to device.");
    button.clearEventBits();  // Clear any clicks before being setup
  }
  delay(1000);
  return _sensor_type;
}

//Enables a specific port number
bool enableMuxPort(byte portNumber)
{
  if(portNumber > 7) portNumber = 7;

  //Read the current mux settings
  Wire.requestFrom(MUX_ADDR, 1);
  if(!Wire.available()) return(false); //Error
  byte settings = Wire.read();

  //Set the wanted bit to enable the port
  settings |= (1 << portNumber);
 
  Wire.beginTransmission(MUX_ADDR);
  Wire.write(settings);
  Wire.endTransmission();

  return(true);
}

//Disables a specific port number
bool disableMuxPort(byte portNumber)
{
  if(portNumber > 7) portNumber = 7;

  //Read the current mux settings
  Wire.requestFrom(MUX_ADDR, 1);
  if(!Wire.available()) return(false); //Error
  byte settings = Wire.read();

  //Clear the wanted bit to disable the port
  settings &= ~(1 << portNumber);

  Wire.beginTransmission(MUX_ADDR);
  Wire.write(settings);
  Wire.endTransmission();

  return(true);
}
