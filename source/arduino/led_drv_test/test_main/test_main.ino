#define print Serial.print
#define println Serial.println
#include "SparkFun_AS7265X.h"
#include <SparkFun_Qwiic_Button.h>

#define BUTTON_ADDR 0x6F
#define AS7265X_CODE 0x41


AS7265X as7265x;
QwiicButton button;

#define LED AS7265x_LED_IR


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
  as7265x.disableBulb(LED);
}


void loop() {
  if (button.hasBeenClicked()) {
    println("pressed");
    button.clearEventBits();  // stop if from triggeing .hasBeenClicked() again

    for (int k=0; k <=3; k++) { // loop through the 12.5, 25, 50, and 100 mA currents
      as7265x.setBulbCurrent(k, LED);
      as7265x.enableBulb(LED);
      delay(100);
      as7265x.disableBulb(LED);
      delay(50);
    }
  }

}
