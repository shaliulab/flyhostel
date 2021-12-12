import logging

import imgstore

logger = logging.getLogger(__name__)
# imgstore_logger = logging.getLogger("imgstore")

def read_store_metadata(experiment_folder, chunk_numbers=None):
    
    store = imgstore.new_for_filename(experiment_folder, chunk_numbers=chunk_numbers)
    store_metadata = store._metadata

    chunks = list(store._index.chunks)
    chunk_metadata = {chunk: store._get_chunk_metadata(chunk) for chunk in chunks}

    return store_metadata, chunk_metadata
