"""
Conversion between experiment names and basedirs

A basedir has one of these forms

    $FLYHOSTEL_VIDEOS/FlyHostel1/6X/2023-05-23_14-00-00/
    $FLYHOSTEL_VIDEOS/FlyHostel1/4X/2026-10-13_17-00-00/arenas/2/

and the corresponding experiment names are

    FlyHostel1_6X_2023-05-23_14-00-00
    FlyHostel1_4X_2026-10-13_17-00-00_arena2

The second form is used by multi-arena (4X throughput) recordings,
where each arena is cropped into its own imgstore under arenas/N

Arena 0 means the recording is not split into arenas, i.e. it lives in the
standard basedir (never in arenas/0), and its experiment name has no _arena suffix
"""
import glob
import logging
import os
import re

logger=logging.getLogger(__name__)

ARENAS_FOLDER="arenas"
ARENA_PREFIX="arena"
ID_PREFIX_LENGTH=26

EXPERIMENT_REGEX=re.compile(
    r"(?P<flyhostel>FlyHostel(?P<flyhostel_id>\d+))_"
    r"(?P<number_of_animals>\d+)X_"
    r"(?P<date_time>(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2}))"
    rf"(?:_{ARENA_PREFIX}(?P<arena>\d+))?"
)


def normalize_arena(arena):
    """
    Return the arena as an int, or None if the recording is not split into arenas (None or 0)
    """
    if arena is None:
        return None
    try:
        arena=int(arena)
    except (TypeError, ValueError):
        # NaN or empty values from metadata tables
        return None
    if arena==0:
        return None
    return arena


def make_experiment_name(flyhostel_id, number_of_animals, date_time, arena=None):
    experiment=f"FlyHostel{flyhostel_id}_{number_of_animals}X_{date_time}"
    arena=normalize_arena(arena)
    if arena is not None:
        experiment+=f"_{ARENA_PREFIX}{arena}"
    return experiment


def match_experiment(text):
    """
    Return the experiment name at the start of text, or None

    text can be an experiment name or any string that starts with one,
    such as a data entry (experiment_chunk_local-identity) or a file name
    """
    match=EXPERIMENT_REGEX.match(os.path.basename(text.rstrip(os.path.sep)))
    if match is None:
        return None
    return match.group(0)


def parse_experiment_name(experiment):
    """
    Return a dict with the fields flyhostel, flyhostel_id, number_of_animals,
    date_time, date, time and arena (None for single arena experiments)
    """
    match=EXPERIMENT_REGEX.fullmatch(experiment)
    if match is None:
        raise ValueError(f"{experiment} is not a valid experiment name")
    fields=match.groupdict()
    fields["flyhostel_id"]=int(fields["flyhostel_id"])
    fields["number_of_animals"]=int(fields["number_of_animals"])
    fields["arena"]=normalize_arena(fields["arena"])
    return fields


def split_arena(experiment):
    """
    FlyHostel1_4X_2026-10-13_17-00-00_arena2 -> (FlyHostel1_4X_2026-10-13_17-00-00, 2)
    FlyHostel1_6X_2023-05-23_14-00-00 -> (FlyHostel1_6X_2023-05-23_14-00-00, None)
    FlyHostel1_6X_2023-05-23_14-00-00_arena0 -> (FlyHostel1_6X_2023-05-23_14-00-00, None)
    """
    match=re.match(rf"(.*)_{ARENA_PREFIX}(\d+)$", experiment)
    if match is None:
        return experiment, None
    return match.group(1), normalize_arena(match.group(2))


def experiment_to_relpath(experiment):
    """
    FlyHostel1_6X_2023-05-23_14-00-00 -> FlyHostel1/6X/2023-05-23_14-00-00
    FlyHostel1_4X_2026-10-13_17-00-00_arena2 -> FlyHostel1/4X/2026-10-13_17-00-00/arenas/2
    """
    experiment, arena=split_arena(experiment)
    tokens=experiment.split("_")
    relpath=os.path.join(tokens[0], tokens[1], "_".join(tokens[2:4]))
    if arena is not None:
        relpath=os.path.join(relpath, ARENAS_FOLDER, str(arena))
    return relpath


def get_experiment_identifier(basedir):
    """
    .../FlyHostel1/6X/2023-05-23_14-00-00 -> FlyHostel1_6X_2023-05-23_14-00-00
    .../FlyHostel1/4X/2026-10-13_17-00-00/arenas/2 -> FlyHostel1_4X_2026-10-13_17-00-00_arena2
    """
    parts=os.path.normpath(basedir).split(os.path.sep)
    if len(parts)>=5 and parts[-2]==ARENAS_FOLDER:
        return "_".join(parts[-5:-2]) + f"_{ARENA_PREFIX}{parts[-1]}"
    return "_".join(parts[-3:])


def get_id_prefix(experiment):
    """
    Return the experiment part of an animal id (experiment|identity)

    Always ID_PREFIX_LENGTH (26) characters long

    FlyHostel1_6X_2023-05-23_14-00-00 -> FlyHostel1_6X_2023-05-23_14
    FlyHostel1_4X_2026-10-13_17-00-00_arena2 -> FH1_4X_261013_17-00_arena2
    """
    fields=parse_experiment_name(experiment)
    if fields["arena"] is None:
        return experiment[:ID_PREFIX_LENGTH]

    date_time=fields["date"].replace("-", "")[2:] + "_" + fields["time"][:5]
    prefix=f"FH{fields['flyhostel_id']}_{fields['number_of_animals']}X_{date_time}_{ARENA_PREFIX}{fields['arena']}"
    if len(prefix)!=ID_PREFIX_LENGTH:
        raise ValueError(f"id prefix {prefix} of {experiment} is not {ID_PREFIX_LENGTH} characters long")
    return prefix


ARENA_ID_PREFIX_REGEX=re.compile(
    rf"FH(?P<flyhostel_id>\d+)_(?P<number_of_animals>\d+)X_"
    rf"(?P<yy>\d{{2}})(?P<mm>\d{{2}})(?P<dd>\d{{2}})_(?P<hh>\d{{2}})-(?P<mi>\d{{2}})_"
    rf"{ARENA_PREFIX}(?P<arena>\d+)"
)


ID_PREFIX_REGEX=re.compile(
    r"(?P<flyhostel>FlyHostel\d+)_(?P<number_of_animals>\d+X)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<hour_digit>\d)"
)


def experiment_from_id(id, videos_folder=None):
    """
    Recover the experiment name of an animal id

    FH1_4X_261013_17-00_arena2|01 -> FlyHostel1_4X_2026-10-13_17-00-00_arena2 (seconds are assumed to be 00)
    FlyHostel1_6X_2023-05-23_1|01 -> FlyHostel1_6X_2023-05-23_1X-00-00

    For single arena ids, the second digit of the hour is truncated,
    so it is recovered from the folder FlyHostel1/6X/2023-05-23_1X-00-00 in videos_folder
    (default $FLYHOSTEL_VIDEOS). If several folders match, the latest one is used

    Returns None if the experiment cannot be recovered
    """
    prefix=id.split("|")[0]
    match=ID_PREFIX_REGEX.fullmatch(prefix)
    if match is not None:
        return _experiment_from_single_arena_id(match.groupdict(), videos_folder=videos_folder)

    match=ARENA_ID_PREFIX_REGEX.fullmatch(prefix)
    if match is None:
        return None
    f=match.groupdict()
    return make_experiment_name(
        f['flyhostel_id'], f['number_of_animals'],
        f"20{f['yy']}-{f['mm']}-{f['dd']}_{f['hh']}-{f['mi']}-00",
        arena=f['arena']
    )


def _experiment_from_single_arena_id(f, videos_folder=None):
    if videos_folder is None:
        videos_folder=os.environ.get("FLYHOSTEL_VIDEOS", "/flyhostel_data/videos")

    pattern=os.path.join(
        videos_folder, f["flyhostel"], f["number_of_animals"],
        f"{f['date']}_{f['hour_digit']}[0-9]-00-00"
    )
    basedirs=sorted(filter(os.path.isdir, glob.glob(pattern)))
    if not basedirs:
        logger.warning("No experiment folder matches %s", pattern)
        return None
    if len(basedirs) > 1:
        logger.warning("%s experiment folders match %s. Using the latest one: %s", len(basedirs), pattern, basedirs[-1])
    return get_experiment_identifier(basedirs[-1])


def make_id(experiment, identity):
    return get_id_prefix(experiment) + "|" + str(identity).zfill(2)
