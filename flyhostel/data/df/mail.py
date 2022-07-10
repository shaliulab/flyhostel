import subprocess
from flyhostel.configuration import load_config

source_address = load_config()["email"]["source"]
HEADERS = f"""Subject: flyhostel file system disk space usage
From: {source_address}
Content-Type: text/html; charset="utf8"
"""
BODY="The partition where the videos are being saved has {} % of memory left"

MAIL_TEMPLATE = HEADERS + "\n" + BODY

def notify_free_fraction(fraction, address):

    message  = MAIL_TEMPLATE.format(fraction)

    subprocess.run([
        "sendmail", address
    ], input = bytes(message, "utf-8")
    )
