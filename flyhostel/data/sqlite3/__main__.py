import os.path
from .idtrackerai import IdtrackeraiExporter

def export_dataset(store_path, chunks, reset=True, tables=None):
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
    dataset = IdtrackeraiExporter(basedir, deepethogram_data=os.environ["DEEPETHOGRAM_DATA"])
    dataset.export(dbfile=dbfile, mode="a", chunks=chunks, reset=reset, tables=tables)
