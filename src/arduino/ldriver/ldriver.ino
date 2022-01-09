#include <SoftwareSerial.h>
// https://github.com/kroimon/Arduino-SerialCommand - not in the repo
#include <SerialCommand.h>
#define BAUD 9600
const float VERSION = 1.1; 

#include<string.h>

SerialCommand SCmd;

// pin where the transistor connects to the Arduino board
int LIGHT_PIN= 6;
int IR_PIN = 5;

int INTENSITY = 0;
int t0 = 0;
int t1 = 0;
unsigned int tick = 0;

// a function that runs once, at startup
void setup() 
{
  int intensity = 255;
  Serial.begin(BAUD);
  Serial.println("Starting Arduino");
  SCmd.addCommand("L", activate_light);  
  SCmd.addCommand("I", activate_ir);
  SCmd.addCommand("S", set);
  SCmd.addCommand("T", teach);

  pinMode(LIGHT_PIN, OUTPUT);
  pinMode(IR_PIN, OUTPUT);
  pinMode(13, OUTPUT);

  analogWrite(LIGHT_PIN, intensity); // ?? mW / sqcm
  analogWrite(IR_PIN, intensity); // ?? mW / sqcm
  teach();
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

  if (strcmp(hardware, "I") == 0) {
    selected = IR_PIN;
  } else if (strcmp(hardware, "L") == 0) {
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

  Serial.print("Setting ");
  Serial.print(selected);
  Serial.print(" --> ");
  Serial.println(intensity);
  analogWrite(selected, intensity);
}

void activate_light() {
  char *arg;

  unsigned int duration = 0;
  unsigned int intensity = 0;
  
  arg = SCmd.next();

  if (arg != NULL) {
    duration = atoi(arg);
  } else {
    return;
  }

  arg = SCmd.next();
  
    if (arg != NULL) {
    intensity = atoi(arg);
  } else {
    return;
  }

  activate(LIGHT_PIN, duration, intensity);
}

void activate_ir() {
  char *arg;

  unsigned int duration = 0;
  unsigned int intensity = 0;
  
  arg = SCmd.next();

  if (arg != NULL) {
    duration = atoi(arg);
  } else {
    return;
  }

  arg = SCmd.next();
  
    if (arg != NULL) {
    intensity = atoi(arg);
  } else {
    return;
  }

  activate(IR_PIN, duration, intensity);
}

void activate(unsigned int selected, unsigned int duration, unsigned int intensity) {
  analogWrite(selected, intensity); // ?? mW / sqcm
  analogWrite(13, intensity); // ?? mW / sqcm
  delay(duration);
  analogWrite(selected, 0); // ?? mW / sqcm
  analogWrite(13, 0); // ?? mW / sqcm
  return;
}

void teach() {
  Serial.print("{\"name\": \"LED driver\", \"version\":  \"");
  Serial.print(VERSION);
  Serial.print("\", \"channels\": {\"IR\": ");
  Serial.print(IR_PIN);
  Serial.print(", \"VISIBLE\": ");
  Serial.print(LIGHT_PIN);
  Serial.println("}}");
  return;
}
