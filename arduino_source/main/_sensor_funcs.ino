String LEDS[3] = {"White", "IR", "UV"};

void run_as7265x(int k_max) {
  println("AS7265X read");
  for (int k=0; k<=k_max; k++) {
    println("Starting Data Read");
    if (k <= 2) {
      as7265x.enableBulb(k);
      print("LED: "); println(LEDS[k]);
    }
    else if (k == 3) {
      as7265x.enableBulb(AS7265x_LED_WHITE);
      as7265x.enableBulb(AS7265x_LED_IR);
      println("LED: White IR");
    }
    else if (k == 4) {
      as7265x.enableBulb(AS7265x_LED_WHITE);
      as7265x.enableBulb(AS7265x_LED_UV);
      println("LED: White UV");
    }
    else if (k == 5) {
      as7265x.enableBulb(AS7265x_LED_UV);
      as7265x.enableBulb(AS7265x_LED_IR);
      println("LED: UV IR");
    }
    else if (k == 6) {
      as7265x.enableBulb(AS7265x_LED_WHITE);
      as7265x.enableBulb(AS7265x_LED_UV);
      as7265x.enableBulb(AS7265x_LED_IR);
      println("LED: White UV IR");
    }
    
    as7265x.takeMeasurements();
    // Just turn all the bulbs off instead of another if else sequence
    as7265x.disableBulb(AS7265x_LED_WHITE);
    as7265x.disableBulb(AS7265x_LED_IR);
    as7265x.disableBulb(AS7265x_LED_UV);
    get_as7265x_data();
    saturated = check_as7262_saturation();
    if (saturated == 1) {
      println("OK Read");
    }
    else {
      println("Saturated Read");
    }
    println("End Data Read");
  }
}

void run_as726x() {
  as726x.takeMeasurementsWithBulb();
  get_as7262_data();
  saturated = check_as7262_saturation();
  if (saturated == 1) {
      println("OK Read");
    }
    else {
      println("Saturated Read");
    }
}


void get_as7265x_data() {
  print("AS7265x Data: ");

  print(as7265x.getCalibratedA()); print(", ");  // 410 nm 
  print(as7265x.getCalibratedB()); print(", ");  // 435 nm
  print(as7265x.getCalibratedC()); print(", ");  // 460 nm
  print(as7265x.getCalibratedD()); print(", ");  // 485 nm
  print(as7265x.getCalibratedE()); print(", ");  // 510 nm
  print(as7265x.getCalibratedF()); print(", ");  // 535 nm
  
  print(as7265x.getCalibratedG()); print(", ");  // 565 nm
  print(as7265x.getCalibratedH()); print(", ");  // 585 nm

  print(as7265x.getCalibratedR()); print(", ");  // 610 nm
  
  print(as7265x.getCalibratedI()); print(", ");  // 645 nm

  print(as7265x.getCalibratedS()); print(", ");  // 680 nm
  
  print(as7265x.getCalibratedJ()); print(", ");  // 705 nm

  print(as7265x.getCalibratedT()); print(", ");  // 730 nm
  print(as7265x.getCalibratedU()); print(", ");  // 760 nm
  print(as7265x.getCalibratedV()); print(", ");  // 810 nm
  print(as7265x.getCalibratedW()); print(", ");  // 860 nm

  print(as7265x.getCalibratedK()); print(", ");  // 900 nm
  println(as7265x.getCalibratedL());  // 940 nm
}


void get_as7262_data() {
  print("AS7262 Data: ");
  print(as726x.getCalibratedViolet(), 4); print(", ");
  print(as726x.getCalibratedBlue(), 4); print(", ");
  print(as726x.getCalibratedGreen(), 4); print(", ");
  print(as726x.getCalibratedYellow(), 4); print(", ");
  print(as726x.getCalibratedOrange(), 4); print(", ");
  println(as726x.getCalibratedRed(), 4); 
}

void get_as7263_data() {
  print("AS7263 Data: ");
  print(as726x.getCalibratedR(), 4); print(", ");
  print(as726x.getCalibratedS(), 4); print(", ");
  print(as726x.getCalibratedT(), 4); print(", ");
  print(as726x.getCalibratedU(), 4); print(", ");
  print(as726x.getCalibratedV(), 4); print(", ");
  println(as726x.getCalibratedW(), 4); 
}

bool check_as7262_saturation() {
  if (as726x.getViolet() >= MAX_CH_VALUE) {
    return false;
  }
  if (as726x.getBlue() >= MAX_CH_VALUE) {
    return false;
  }
  if (as726x.getGreen() >= MAX_CH_VALUE) {
    return false;
  }
  if (as726x.getYellow() >= MAX_CH_VALUE) {
    return false;
  }
  if (as726x.getOrange() >= MAX_CH_VALUE) {
    return false;
  }
  if (as726x.getRed() >= MAX_CH_VALUE) {
    return false;
  }
  return true;  
}

bool check_as7263_saturation() {
  if (as726x.getViolet() >= MAX_CH_VALUE) {
    return false;
  }
  if (as726x.getBlue() >= MAX_CH_VALUE) {
    return false;
  }
  if (as726x.getGreen() >= MAX_CH_VALUE) {
    return false;
  }
  if (as726x.getYellow() >= MAX_CH_VALUE) {
    return false;
  }
  if (as726x.getOrange() >= MAX_CH_VALUE) {
    return false;
  }
  if (as726x.getRed() >= MAX_CH_VALUE) {
    return false;
  }
  return true;  
}

bool check_as7265x_saturation() {
  if (as7265x.getA() >= MAX_CH_VALUE) {
    return false;
  }
  if (as7265x.getB() >= MAX_CH_VALUE) {
    return false;
  }
  if (as7265x.getC() >= MAX_CH_VALUE) {
    return false;
  }
  if (as7265x.getD() >= MAX_CH_VALUE) {
    return false;
  }
  if (as7265x.getE() >= MAX_CH_VALUE) {
    return false;
  }
  if (as7265x.getF() >= MAX_CH_VALUE) {
    return false;
  }
  if (as7265x.getG() >= MAX_CH_VALUE) {
    return false;
  }
  if (as7265x.getH() >= MAX_CH_VALUE) {
    return false;
  }
  if (as7265x.getI() >= MAX_CH_VALUE) {
    return false;
  }
  if (as7265x.getJ() >= MAX_CH_VALUE) {
    return false;
  }
  if (as7265x.getK() >= MAX_CH_VALUE) {
    return false;
  }
  if (as7265x.getL() >= MAX_CH_VALUE) {
    return false;
  }
  if (as7265x.getR() >= MAX_CH_VALUE) {
    return false;
  }
  if (as7265x.getS() >= MAX_CH_VALUE) {
    return false;
  }
  if (as7265x.getT() >= MAX_CH_VALUE) {
    return false;
  }
  if (as7265x.getU() >= MAX_CH_VALUE) {
    return false;
  }
  if (as7265x.getV() >= MAX_CH_VALUE) {
    return false;
  }
  if (as7265x.getW() >= MAX_CH_VALUE) {
    return false;
  }
  return true;  
}


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
