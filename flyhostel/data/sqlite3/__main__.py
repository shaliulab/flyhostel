import os.path
from .idtrackerai import IdtrackeraiExporter
from .constants import PRESETS, BEHAVIORS

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
    dbfile_basename = "_".join(basedir.split(os.path.sep)[-3:]) + ".db"

    dbfile = os.path.join(basedir, dbfile_basename)
    basedir=os.path.realpath(basedir)
    print("Initializing exporter")
    dataset = IdtrackeraiExporter(basedir, deepethogram_data=os.environ["DEEPETHOGRAM_DATA"], framerate=framerate)
    mode="a"
    if len(tables)== 1 and tables[0] in PRESETS:
        if tables[0] == "default":
            mode = "w"
        tables=PRESETS[tables[0]]


    print(f"Start export of tables {tables}")
    dataset.export(dbfile=dbfile, chunks=chunks, tables=tables, behaviors=BEHAVIORS, mode=mode, reset=reset)