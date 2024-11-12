/*
 * File: test_main.ino
 * Purpose: Go through each current level allowed by the AS7265x sensor
 * Author: Kyle Vitatuas Lopin
 * Date: 11-11-2024
 *
 * Description:
 *   - Connect an AS7265x sensor and Qwiic button to the arduino
 *   - When the button is pressed the AS7265x will go through all 4 current levels (12.5, 25, 50, and 100 mA) and turn on the LED for 100 msecs
 *
 * Usage:
 *   - Used to measure LED drive pin of the AS765x ICs to test if the voltage us above V_LED voltage specified by the datasheet
 *
 */

#define print Serial.print
#define println Serial.println
#include "SparkFun_AS7265X.h"
#include <SparkFun_Qwiic_Button.h>

#define BUTTON_ADDR 0x6F
#define AS7265X_CODE 0x41


AS7265X as7265x;
QwiicButton button;

uint8_t led_input = 0;
String input = "";
unsigned long previousMillis = 0;     // Stores last time LED was toggled
const unsigned long ON_TIME = 100;    // LED on duration in milliseconds
const unsigned long OFF_TIME = 50;    // LED off duration in milliseconds


void setup() {
  Serial.begin(115200); // Initialize Serial Monitor USB

  while (!Serial) ; // Wait for Serial monitor to open

  // Send a welcome message to the serial monitor:
  println("Send character(s) to relay it over Serial1");
  if (as7265x.begin() == false)
  {
    println("Sensor does not appear to be connected. Please check wiring. Freezing...");
    while (1)
      ;
  }
  button.begin();
}


void loop() {

  if(Serial.available()){
    input = Serial.readString();
    print("got input: "); println(input);
    led_input = input.charAt(0) - '0';  // Convert the string number to uint8_t
    print("led set to: "); println(led_input);
  }


  if (button.hasBeenClicked()) {
    println("pressed");
    button.clearEventBits();  // stop if from triggeing .hasBeenClicked() again
    run_led_scan(led_input);
  }

}

void run_led_scan(uint8_t led) {
  int k = 0;
  unsigned long next_step = millis();
  as7265x.setBulbCurrent(k, led);  // Set the bulb current based on k
  do {
    as7265x.enableBulb(led);
    next_step += ON_TIME;
    
    while (millis() < next_step) {
      // wait
    }

    as7265x.disableBulb(led);
    next_step += OFF_TIME;
    // handle next step settings now
    k++;
    as7265x.setBulbCurrent(k, led);  // Set the bulb current based on k
    
    while (millis() < next_step) {
      // wait
    }
  } while (k <= 3);

}
