High resolution monitoring in Drosophila
===============================================


Dev notes
==========

* To build: `python setup.py sdist bdist_wheel`

* To publish: `python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/* --verbose`




### Sensors

This module contains information on how to set up an environmental sensor for behavior experiments

The sensor measures

* **temperature**
* **humidity**
* **light**

Components

1. BME280 sensor (temperature and humidity)
2. Photoresistor (light)
3. Arduino Uno
4. RaspberryPi

NOTE. A BMP280 sensor exists which does not measure humidity.

# Where to get the BME280 sensor

https://www.adafruit.com/product/2652

# How the PHR is designed

https://create.arduino.cc/projecthub/MisterBotBreak/how-to-use-a-photoresistor-46c5eb


# How to set up the sensor connections


### Photoresistor

* 10kOhm resistor and photoresistor need to be connected in series
* The remaining pins are connected to 5V and GND.
* The series connected pins should be connected to A0

### BME280

4 headers need to be connected: 5V, GND, SDI (A4) and SCK (A5).

|  Arduino board |    Headers    | Component |
|----------------|---------------|-----------|
|   5V           |   VIN         |  BME280   |
|   NA           |   3V          |  BME280   |
|   GND          |   GND         |  BME280   |
|   A5           |   SCK         |  BME280   |
|   NA           |   SDO         |  BME280   |
|   A4           |   SDI         |  BME280   |
|   NA           |   CS          |  BME280   |
|   A0           |   PHR2/10kOhm |  PHResist |
|   5V           |   PHR1        |  PHResist |
|   GND          |   10 kOhm     |  PHResist |
|----------------|---------------|-----------|


# TODO

* If the Arduino is not connected on ACM0, figure it out and change the port
