/*
  ESP32 mDNS based environmental sensor
  https://www.notion.so/giorgiogilestro/ESP8266-Sensor-mesh-fbf3c45a68034634a02316458a0336b5
*/
#if defined(ESP32)
#include <ESPmDNS.h>
#include "esp_system.h"
#endif


//For I2C communication
#include <Wire.h>

//For sensor-specific routines
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>
#include <SerialCommand.h>

//Initialise the I2C environmental sensors
Adafruit_BME280 bme;
SerialCommand SCmd;
const float VERSION = 1.2;

typedef struct {
  float temperature;
  float humidity;
} environment;

environment env;

void setup(void)
{
  Serial.begin(57600);

  // Initialise BME280 and BH1750 I2C sensors
  bme.begin(0x76);
  SCmd.addCommand("T", teach);
  SCmd.addCommand("D", printValues);
}

void loop(void){
  SCmd.readSerial();
  delay(250);
}

void readEnv(void) {
  env.temperature = bme.readTemperature();
  env.humidity = bme.readHumidity();
}


void teach() {
  Serial.print("{\"name\": \"Environmental sensor\", \"version\": \"");
  Serial.print(VERSION);
  Serial.println("\"}");
}


void printValues() {
  readEnv();
  Serial.print("{");
  Serial.print("\"temperature\": ");
  Serial.print(env.temperature);
  Serial.print(",");
  Serial.print("\"pressure\": ");
  Serial.print(0 / 100.0F);
  Serial.print(",");
  Serial.print("\"altitude\": ");
  Serial.print(0);
  Serial.print(",");
  Serial.print("\"humidity\": ");
  Serial.print(env.humidity);
  Serial.print(",");
  Serial.print("\"light\": ");
  Serial.print(0);
  Serial.print("}");
  Serial.println();
}
