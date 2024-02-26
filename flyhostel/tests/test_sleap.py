import os

from flyhostel.data.sqlite3.idtrackerai import IdtrackeraiExporter
from flyhostel.data.sqlite3.constants import NODES

basedir="./static_data/test_sleap/videos/FlyHostel1/5X/2023-05-23_14-00-00/"

dataset=IdtrackeraiExporter(
    basedir=basedir,
    deepethogram_data=None,
    framerate=None
)

chunks=[50]
reset=True
dbfile=os.path.join(basedir, "input.db")
tables=["POSE"]
dataset.export(dbfile=dbfile, chunks=chunks, tables=tables, behaviors=[], nodes=NODES, reset=reset, local_identities=[1])
