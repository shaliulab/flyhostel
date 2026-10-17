import os.path
import re
import glob
from .idtrackerai import IdtrackeraiExporter
from .constants import PRESETS, BEHAVIORS, NODES
from .utils import parse_experiment_properties
from flyhostel.utils.experiment import get_experiment_identifier, parse_experiment_name

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
    # FlyHostelN_NX_YYYY-MM-DD_HH-MM-SS[_arenaN].db
    # resolve the metadata file itself (not just its folder) so that it works
    # when metadata.yaml is a symlink staged in a nextflow work directory
    dbfile_basename = get_experiment_identifier(os.path.dirname(os.path.realpath(store_path))) + ".db"
    fields=parse_experiment_name(dbfile_basename[:-len(".db")])
    assert fields["flyhostel_id"]==flyhostel_id and fields["date_time"]==date_time, f"{dbfile_basename} does not match pattern"
    # in multi arena experiments, the idtrackerai conf holds the number of animals of one arena
    # while the experiment name holds the number of animals in the whole flyhostel
    if fields["arena"] is None:
        assert fields["number_of_animals"]==number_of_animals, f"{dbfile_basename} does not match pattern"

    dbfile = os.path.join(basedir, dbfile_basename)
    basedir=os.path.realpath(basedir)
    print("Initializing exporter")
    dataset = IdtrackeraiExporter(
        basedir,
        framerate=framerate
    )
    if len(tables)== 1 and tables[0] in PRESETS:
        tables=PRESETS[tables[0]]


    print(f"Start export of tables {tables}")
    dataset.export(dbfile=dbfile, chunks=chunks, tables=tables, behaviors=BEHAVIORS, nodes=NODES, reset=reset)
