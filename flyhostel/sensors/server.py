import threading
import logging
import socket
import json
import os
from flyhostel.sensors.sensor import Sensor
from flyhostel.sensors.constants import HTML_PORT, JSON_PORT

MAX_COUNT = 10

class Server(threading.Thread):

    _SERVER_HOST = "0.0.0.0"
    _html = """
        <!DOCTYPE HTML><html>
        <head>
        <title>BME280 Web Server at %s</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.7.2/css/all.css" integrity="sha384-fnmOCqbTlWIlj8LyTjo7mOUStjsKC4pOpQbqyi7RrhN7udi9RwhKkMHpvLbHG9Sr" crossorigin="anonymous">
        <link rel="icon" href="data:,">

        </head>
        <body>
        <div class="topnav">
            <h3>BME280 WEB SERVER at %s</h3>
            <h3>Last timestamp: %s</h3>
        </div>
        <div class="content">
            <div class="cards">
            <div class="card temperature">
                <h4><i class="fas fa-thermometer-half"></i> TEMPERATURE</h4><p><span class="reading"><span id="temp">%s</span> &deg;C</span></p>
            </div>
            <div class="card humidity">
                <h4><i class="fas fa-tint"></i> HUMIDITY</h4><p><span class="reading"><span id="humid">%s %%</span></span></p>
            </div>
             <div class="card light">
                <h4><i class="fas fa-sun-o"></i> LIGHT</h4><p><span class="reading"><span id="light">%s (higher -> more light)</span></span></p>
            </div>

            <div class="card pressure">
                <h4><i class="fas fa-angle-double-down"></i> PRESSURE</h4><p><span class="reading"><span id="pres">%s</span> hPa</span></p>
            </div>
            </div>
        </div>

        <div>
        <img src="%s">
        </div>

        </body>
        </html>
        """

    def __init__(self, sensor, server_type="html", port=HTML_PORT, *args, **kwargs):

        with open("/etc/hostname", "r") as fh:
            self._hostname = fh.read().strip("\n")

        self._server_type = server_type
        self._port = port
        # Run sensor
        self._sensor = sensor

        # Create socket
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
        )
        try:
            self._server_socket.bind((self._SERVER_HOST, self._port))
        except Exception as error:
            logging.warning(f"Cannot bind to port {self._port}")
            raise error
            
        self._server_socket.listen(1)
        self._count = 0


        print("Listening on port %s ..." % self._port)
        super().__init__(*args, **kwargs)

    def _fill_html(self, src=""):

        response = self._html % (
            self._hostname,
            self._hostname,
            self._sensor.datetime,
            self._sensor.temperature,
            self._sensor.humidity,
            self._sensor.light,
            self._sensor.pressure,
            src,
        )
        return response

    def run(self):
        # Define socket host and port
        while True:
            client_connection, client_address = self._server_socket.accept()
            request = client_connection.recv(1024).decode()
            if self._sensor.temperature == 0:
                self._count += 1
                if self._count > MAX_COUNT:
                    os.system("reboot")
            else:
                self._count = 0

            if self._server_type == "html":
                response = self._fill_html()

            if self._server_type == "json":

                json_data = {
                    "hostname": self._hostname,
                    "temperature": self._sensor.temperature,
                    "humidity": self._sensor.humidity,
                    "light": self._sensor.light,
                    "pressure": self._sensor.pressure,
                    "time": self._sensor.datetime,
                }

                response = json.dumps(json_data)

            response = "HTTP/1.0 200 OK\n\n" + response
            print(response)
            # Send HTTP response
            client_connection.sendall(response.encode())
            client_connection.close()

        # Close socket
        self._server_socket.close()


if __name__ == "__main__":

    sensor = Sensor("/dev/ttyACM0")
    sensor.start()
    html_server = Server(sensor, server_type="html", port=HTML_PORT)
    json_server = Server(sensor, server_type="json", port=JSON_PORT)

    # html_server.start()
    json_server.start()
