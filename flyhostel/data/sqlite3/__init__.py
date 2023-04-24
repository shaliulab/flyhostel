"""
Export information in an flyhostel experiment to a centralized sqlite file

The module expects:

* video_object.npy, preprocessing/blob_collection.npy for every session folder
* metadata.yaml, index.db to produce METADATA, SNAPSHOT, STORE_INDEX, behaviors
* OPTIONAL: preprocessing/fragments.npy to produce the AI table
* OPTIONAL: .pfs file to store camera metadata in METADATA
* OPTIONAL: .extra.json to produce ENVIRONMENT


"""
from .__main__ import export_dataset
from .idtrackerai import IdtrackeraiExporter
