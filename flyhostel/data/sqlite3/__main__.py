import os.path
import re
import glob
from .idtrackerai import IdtrackeraiExporter
from .constants import PRESETS, BEHAVIORS, NODES
from .utils import parse_experiment_properties

def export_dataset(store_path, chunks, reset=True, framerate=None, tables=None):
    """
    Store all data available for a FlyHostel experiment into a single SQLite (.db) file

    Args:

        store_path (str): Pointer to the metadata.yaml file of a FlyHostel recording
        chunks (list): Chunks of the recording to be included
        reset (bool): If True, the SQLite file will be remade from scratch
        tables (list): If passed, only these tables will be written/updated
    """

    basedir = os.path.dirname(store_path)
    (_, _), (flyhostel_id, number_of_animals, date_time) = parse_experiment_properties()
    dbfile_basename = f"FlyHostel{flyhostel_id}_{number_of_animals}X_{date_time}.db"

    dbfile = os.path.join(basedir, dbfile_basename)
    basedir=os.path.realpath(basedir)
    print("Initializing exporter")
    dataset = IdtrackeraiExporter(
        basedir,
        deepethogram_data=os.environ["DEEPETHOGRAM_DATA"],
        sleap_data=os.environ["SLEAP_DATA"],
        framerate=framerate
    )
    if len(tables)== 1 and tables[0] in PRESETS:
        tables=PRESETS[tables[0]]


    print(f"Start export of tables {tables}")
    dataset.export(dbfile=dbfile, chunks=chunks, tables=tables, behaviors=BEHAVIORS, nodes=NODES, reset=reset)
