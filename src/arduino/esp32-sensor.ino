/*
  ESP32/8266 mDNS based environmental sensor
  https://www.notion.so/giorgiogilestro/ESP8266-Sensor-mesh-fbf3c45a68034634a02316458a0336b5
  
  This is an example of an HTTP server that is accessible
  via mDNS responder.

  Soldier together the two sensors boards (BME280 and BH1750), using the same header
  then connect to the ESP32 via I2C cable
  VCC -> 3.3V
  Gnd -> Gnd
  SDA -> D21
  SCL -> D22

  or (reccomended) ESP8266 D1 mini Pro
  VCC -> 3.3V
  Gnd -> Gnd
  SDA -> D2
  SCL -> D1

  Instructions:
  - Update WiFi SSID and password as necessary.
  - Flash the sketch to the board
  - Install host software:
    - For Linux, install Avahi (http://avahi.org/).
    - For Windows, install Bonjour (http://www.apple.com/support/bonjour/).
    - For Mac OSX and iOS support is built in through Bonjour already.
  - Point your browser to http://etho_sensor.local, you should see a response.

  - To memorize multiple SSIDs options instead of only one
 https://tttapa.github.io/ESP8266/Chap10%20-%20Simple%20Web%20Server.html


*/
#if defined(ESP32)
  #include <ESPmDNS.h>
  #include "esp_system.h"

  // Ideally, on ESP32 we would also use EEPROM rotate for consistency
  // https://github.com/xoseperez/eeprom32_rotate
  // but it seems to be too buggy
  //#include <EEPROM32_Rotate.h>
  //EEPROM32_Rotate EEPROMr;

  // To save data in the EEPROM use the following
  // http://tronixstuff.com/2011/03/16/tutorial-your-arduinos-inbuilt-eeprom/
  #include <EEPROM.h>

#elif defined(ESP8266)
  #include <ESP8266mDNS.h>
  #include <ESP8266WebServer.h>

  ESP8266WebServer server(80);

  // For ESP8266, EEPROM rotate seems the only library that actually works
  // https://github.com/xoseperez/eeprom_rotate
  #include <EEPROM_Rotate.h>
  EEPROM_Rotate EEPROMr;
#endif

#define EEPROM_SIZE 4096 //total size of the eeprom we want to use
#define EEPROM_START 128 //from which byte on we start writing data

//For I2C communication
#include <Wire.h>

//For sensor-specific routines
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>
#include <BH1750FVI.h>

//Initialise the I2C environmental sensors
Adafruit_BME280 bme;

//We want to use a watchdog for ESP32 to avoid recurrent crashes
#if defined(ESP32)
  const int button = 0;         //gpio to use to trigger delay
  const int wdtTimeout = 3000;  //time in ms to trigger the watchdog
  hw_timer_t *timer = NULL;

  void IRAM_ATTR resetModule() {
    esp_restart();
  }
#endif

typedef struct {
  float temperature;
  float humidity;
  float pressure;
  uint16_t lux;  
} environment;

typedef struct {
  char location[20];
  char sensor_name[20];
} configuration;

configuration cfg = {"", "flyhostel_sensor"};
environment env;

void setup(void)
{  
    Serial.begin(57600);

#if defined(ESP8266)    
    Serial.println("Initialising EEPROM.");
    EEPROMr.begin(EEPROM_SIZE);
    delay(1000);
    Serial.println("EEPROM initialised.");
#endif

#if defined(ESP32)
    if (!EEPROM.begin(EEPROM_SIZE)) {
        Serial.println("Failed to initialise EEPROM");
        Serial.println("Restarting...");
        delay(1000);
        ESP.restart();
    }
    Serial.println("ESP32 detected");
#endif

    loadConfiguration();

    // Initialise BME280 and BH1750 I2C sensors
    bme.begin(0x76);
    readEnv();
    Serial.println(SendJSON());

#if defined(ESP32)
    //Starting a watchdog timer
    timer = timerBegin(0, 80, true);                  //timer 0, div 80
    timerAttachInterrupt(timer, &resetModule, true);  //attach callback
    timerAlarmWrite(timer, wdtTimeout * 1000, false); //set time in us
    timerAlarmEnable(timer);                          //enable interrupt
#endif
}

void loop(void)
{
  
  #if defined(ESP32)
      timerWrite(timer, 0); //reset timer (feed watchdog)
  #endif
  readEnv();
  Serial.println(SendJSON());
  delay(1000);
}

void loadConfiguration()
{

#if defined(ESP8266)
    if (EEPROMr.read(EEPROM_START) != 1) { saveConfiguration(); }
    EEPROMr.get(EEPROM_START+2, cfg);
    delay(500);
#endif

#if defined(ESP32)

    if (EEPROM.read(EEPROM_START) != 1) { saveConfiguration(); } 
    EEPROM.get(EEPROM_START+2, cfg);
#endif

    Serial.println("Configuration loaded.");

}

void saveConfiguration()
{
#if defined(ESP8266)
    EEPROMr.write(EEPROM_START, 1);
    delay(250);
    EEPROMr.put(EEPROM_START+2, cfg);
    delay(250);
    EEPROMr.commit();
#endif

#if defined(ESP32)
    EEPROM.write(EEPROM_START, 1);
    EEPROM.put(EEPROM_START+2, cfg);
    EEPROM.commit();
#endif

    Serial.println("Configuration saved.");
}

//to rename from commandline use the following command
//curl -d location=Incubator_1A -d sensor_name=etho_sensor_1A -G http://DEVICE_IP/set
//DEVICE_IP can be found opening the serial port


void readEnv(void){
    env.temperature = bme.readTemperature();
    env.humidity = bme.readHumidity();
  }

String SendJSON(){
    String ptr = "{\"id\": \"";
    ptr += getMacAddress();
    ptr += "\", \"name\" : \"";
    ptr += cfg.sensor_name;
    ptr += "\", \"location\" : \"";
    ptr += cfg.location;
    ptr += "\", \"temperature\" : \"";
    ptr += env.temperature;
    ptr += "\", \"humidity\" : \"";
    ptr += env.humidity;
    ptr += "\"}";
    return ptr; 
}


String getMacAddress(void) {
    uint8_t baseMac[6];

    // Get the MAC address as UID
    #if defined(ESP32)
      esp_read_mac(baseMac, ESP_MAC_WIFI_STA);
    #endif
    
    char baseMacChr[18] = {0};
    sprintf(baseMacChr, "%02X:%02X:%02X:%02X:%02X:%02X", baseMac[0], baseMac[1], baseMac[2], baseMac[3], baseMac[4], baseMac[5]);
    return String(baseMacChr);
}
