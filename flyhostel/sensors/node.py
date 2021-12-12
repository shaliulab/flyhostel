import argparse
import urllib.request
import logging
import json
import traceback

# import os.path
import bottle

# For plots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pandas as pd
import numpy as np
import seaborn as sns

sns.set(rc={"figure.figsize": (11, 4)})

# To inject images in html
import base64


## https://stackoverflow.com/questions/250283/how-to-scp-in-python
import paramiko
from scp import SCPClient

from server import Server


def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client


PORT = 7575
DEBUG = True

api = bottle.Bottle()

#############
### CLASSS TO BE REMOVED IF BOTTLE CHANGES TO 0.13
############
class CherootServer(bottle.ServerAdapter):
    def run(self, handler):  # pragma: no cover
        from cheroot import wsgi
        from cheroot.ssl import builtin

        self.options["bind_addr"] = (self.host, self.port)
        self.options["wsgi_app"] = handler
        certfile = self.options.pop("certfile", None)
        keyfile = self.options.pop("keyfile", None)
        chainfile = self.options.pop("chainfile", None)
        server = wsgi.Server(**self.options)
        if certfile and keyfile:
            server.ssl_adapter = builtin.BuiltinSSLAdapter(
                certfile, keyfile, chainfile
            )
        try:
            server.start()
        finally:
            server.stop()


############


def download_logs(ip, filename):
    server = ip
    port = 22
    user = "root"
    password = "root"
    ssh = createSSHClient(server, port, user, password)
    scp = SCPClient(ssh.get_transport())
    scp.get("/sensor_log.txt", f"/ethoscope_data/sensors/{filename}")
    df = pd.read_csv(
        f"/ethoscope_data/sensors/{filename}",
        sep="\t",
        header=None,
        names=[
            "datetime",
            "temperature",
            "humidity",
            "light",
            "pressure",
            "altitude",
        ],
        parse_dates=True,
        index_col=0,
    )

    return df


@api.get("/plot/<ip>/<incubator>")
def plot(ip, incubator, timeout=10):
    try:
        filename = f"sensor-log_{ip}.txt"
        plotname = f"sensor-plot_{ip}.png"
        df = download_logs(ip, filename=filename)
        cols_plot = ["temperature", "humidity", "light"]
        limits = [[20, 30], [0, 100], [0, 1000]]
        axes = df[cols_plot].plot(
            marker=".",
            alpha=0.5,
            linestyle="None",
            figsize=(11, 9),
            subplots=True,
            title=f"Incubator {incubator} sensor data",
        )
        i = 0
        for ax in axes:
            ax.set_ylabel(cols_plot[i])
            ax.set_ylim(limits[i])
            ax.xaxis.set_major_locator(
                mdates.HourLocator(byhour=range(0, 24, 2))
            )
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d_%H:%M"))

            i += 1
        plot_abspath = f"/ethoscope_data/sensors/{plotname}"
        plt.savefig(plot_abspath)

        data = get_data(ip)
        html_page = Server._html
        img_url = f"file:///{plot_abspath}"

        with open(plot_abspath, "rb") as image_file:
            # From https://stackoverflow.com/questions/3715493/encoding-an-image-file-with-base64
            encoded_string = base64.b64encode(image_file.read())
            # From https://stackoverflow.com/questions/8499633/how-to-display-base64-images-in-html
            base64_image = f"data:image/png;base64, {encoded_string.decode()}"

        html_page = html_page % (
            f"incubator {incubator}",
            f"incubator {incubator}",
            data["temperature"],
            data["humidity"],
            data["light"],
            data["pressure"],
            base64_image,
        )
        return html_page

    except Exception as error:
        logging.error(error)
        logging.warning(traceback.print_exc())
        return {"code": 1}


@api.get("/data/<ip>")
def get_data(ip, timeout=10):
    try:
        url = f"http://{ip}:9001"
        print(f"GET - {url}")
        req = urllib.request.Request(url)
        f = urllib.request.urlopen(req, timeout=timeout)
        message = f.read()
        if not message:
            # logging.error("URL error whist scanning url: %s. No message back." % self._id_url)
            logging.warning("No message back")
        try:
            resp = json.loads(message)
        except ValueError:
            logging.warning("Could not parse json")
    except Exception as error:
        logging.warning(error)

    return resp


#######TO be remove when bottle changes to version 0.13
server = "cherrypy"
try:
    from bottle.cherrypy import wsgiserver
except:
    # Trick bottle into thinking that cheroot is cherrypy
    bottle.server_names["cherrypy"] = CherootServer(host="0.0.0.0", port=PORT)
    logging.warning(
        "Cherrypy version is bigger than 9, we have to change to cheroot server"
    )
    pass
#########

if __name__ == "__main__":
    bottle.run(api, host="0.0.0.0", port=PORT, debug=DEBUG, server="cherrypy")
