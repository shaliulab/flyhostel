import logging
import subprocess
import re

import smtplib, ssl
from flyhostel.configuration import load_config

logger = logging.getLogger(__name__)

source_address = load_config()["email"]["source"]
HEADERS = f"""Subject: flyhostel file system disk space usage
From: {source_address}
Content-Type: text/html; charset="utf8"
"""
BODY="The partition where the videos are being saved has {} % of memory left"

MAIL_TEMPLATE = HEADERS + "\n" + BODY

def send_mail_subprocess(message, address):
    cmd = ["sendmail", "-v", address]
    logger.debug(f"Executing command {cmd}")
    logger.debug(f"Sending message {message}")
    subprocess.run(cmd, input = bytes(message, "utf-8"))


def get_email_password(config_file="/etc/ssmtp/ssmtp.conf"):
    with open(config_file, "r") as filehandle:
        while True:
            try:
                line = filehandle.read().rstrip("\n")
            except:
                break

            hit = re.search("AuthPass=(.*)", line).group(1)
            if hit:
                password = hit
                logger.debug(f"Password could be parsed from {config_file}")
                break

        if not hit:
            raise Exception(f"Password could not be parsed from {config_file}")

    return password


def send_mail_python(message, address):
    # from https://realpython.com/python-send-email/

    port = 465  # For SSL
    password = get_email_password()
    
    # Create a secure SSL context
    context = ssl.create_default_context()
    
    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login(source_address, password)
        # TODO: Send email here
        server.sendmail(source_address, address, message)




def send_mail(message, address):
    return send_mail_python(message, address)

def notify_free_fraction(fraction, address):

    message  = MAIL_TEMPLATE.format(fraction)
    send_mail(message, address)

