#include <SoftwareSerial.h>
// https://github.com/kroimon/Arduino-SerialCommand - not in the repo
#include <SerialCommand.h>
#define BAUD 9600
const float VERSION = 1.1; 

#include<string.h>

SerialCommand SCmd;

// pin where the transistor connects to the Arduino board
int LIGHT_PIN= 6;
int INTENSITY = 1;

int t0 = 0;
int t1 = 0;
unsigned int tick = 0;

// a function that runs once, at startup
void setup() 
{
  int intensity = INTENSITY;
  Serial.begin(BAUD);
  Serial.println("Starting Arduino");
  SCmd.addCommand("S", set);
  pinMode(LIGHT_PIN, OUTPUT);
  analogWrite(LIGHT_PIN, intensity); // ?? mW / sqcm
}

// a function that runs over and over again, forever, after the setup function
void loop() {
  SCmd.readSerial(); 
  delay(50);
  t0 = t1;
  t1 = millis();
  tick =  t1 - t0;
}

char *next() {
  char *arg;
  arg = SCmd.next();
  Serial.println(arg);
  return arg;
}


void set() {
  char *arg;
  char *hardware;

  unsigned int intensity = 0;
  unsigned int selected = 0;
   
  arg = SCmd.next();

  if (arg != NULL) {
    hardware = arg;
  } else {
    return;
  }

  if (strcmp(hardware, "L") == 0) {
    selected = LIGHT_PIN;
  } else {
    return ;
  }
  
  arg = SCmd.next();
  
  if (arg != NULL) {
    intensity = atoi(arg);
  } else {
    return;
  }
  
  if (selected == LIGHT_PIN) {
      analogWrite(selected, intensity);
      Serial.print("Settting LIGHT_PIN to ");
      Serial.println(intensity);
    }
}

