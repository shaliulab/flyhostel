# import logging
# import hashlib
# from pathlib import Path
# import pandas as pd
# import numpy as np
# logger = logging.getLogger(__name__)

# # Cache location: per-experiment, hidden, easy to nuke.
# _CACHE_VERSION = 1  # bump if you change _walk_frames semantics


# def _load_data_cache_dir(basedir):
#     d = Path(basedir) / ".load_data_cache"
#     d.mkdir(exist_ok=True)
#     return d


# def _video_fingerprint(basedir):
#     """
#     Cheap content fingerprint of the recording: (mtime_ns, size) of metadata.yaml
#     plus the first chunk's video file. Won't catch in-place re-encodes of later
#     chunks, but those are rare in this workflow.
#     """
#     meta = Path(basedir) / "metadata.yaml"
#     st = meta.stat()
#     parts = [("metadata.yaml", st.st_mtime_ns, st.st_size)]
#     # Add the first .mp4/.avi/whatever in basedir as a cheap proxy for the data.
#     for p in sorted(Path(basedir).glob("*.mp4")):
#         s = p.stat()
#         parts.append((p.name, s.st_mtime_ns, s.st_size))
#         break
#     return parts


# def _df_hash(df, cols=None):
#     """Order-independent content hash of a DataFrame (optionally restricted to cols)."""
#     if df is None or df.empty:
#         return "empty"
#     sub = df[cols] if cols is not None else df
#     # Sort so row order doesn't affect the hash.
#     sub = sub.sort_values(list(sub.columns), kind="mergesort").reset_index(drop=True)
#     h = pd.util.hash_pandas_object(sub, index=False).values
#     return hashlib.sha1(h.tobytes()).hexdigest()


# def _contours_hash(annotated_contours):
#     """Hash a dict[int, np.ndarray] by hashing each array's bytes in key order."""
#     h = hashlib.sha1()
#     for k in sorted(annotated_contours.keys()):
#         arr = np.ascontiguousarray(annotated_contours[k])
#         h.update(str(k).encode())
#         h.update(arr.shape.__repr__().encode())
#         h.update(arr.dtype.str.encode())
#         h.update(arr.tobytes())
#     return h.hexdigest()


# def _load_data_cache_key(basedir, tasks, frames_from_annotation, first_frame_number, last_frame_number, image_format):
#     """
#     Fingerprint everything _walk_frames depends on. Cheap relative to running
#     _walk_frames itself (hashing a few MB of dataframes takes ~10ms; the walk
#     takes seconds to minutes).
#     """
#     parts = {
#         "version": _CACHE_VERSION,
#         "video": _video_fingerprint(basedir),
#         # TODO

#     }
#     blob = repr(sorted(parts.items())).encode()
#     return hashlib.sha1(blob).hexdigest()


# def _load_data_cache_key(
#     basedir,
#     tasks,
#     frames_from_annotation,
#     first_frame_number,
#     last_frame_number,
#     image_format,
# ):
#     """
#     Fingerprint everything _walk_frames depends on.

#     Cheap relative to running _walk_frames itself
#     (hashing a few MB of dataframes takes ~10ms; the walk
#     takes seconds to minutes).
#     """

#     def _stable_task_repr(task):
#         """
#         Convert task objects into a deterministic representation.

#         Falls back progressively depending on the object type.
#         """
#         if isinstance(task, dict):
#             return tuple(sorted((k, _stable_task_repr(v)) for k, v in task.items()))

#         if isinstance(task, (list, tuple)):
#             return tuple(_stable_task_repr(x) for x in task)

#         if hasattr(task, "__dict__"):
#             return (
#                 task.__class__.__name__,
#                 tuple(
#                     sorted(
#                         (k, _stable_task_repr(v))
#                         for k, v in vars(task).items()
#                     )
#                 ),
#             )

#         return repr(task)

#     def _hash_annotation_frames(obj):
#         """
#         Create a compact deterministic hash for annotation-derived data.
#         Supports pandas objects, numpy arrays, and generic python objects.
#         """
#         if obj is None:
#             return None

#         try:
#             import pandas as pd

#             if isinstance(obj, (pd.DataFrame, pd.Series)):
#                 h = pd.util.hash_pandas_object(obj, index=True).values.tobytes()
#                 return hashlib.sha1(h).hexdigest()
#         except Exception:
#             pass

#         try:
#             import numpy as np

#             if isinstance(obj, np.ndarray):
#                 return hashlib.sha1(obj.tobytes()).hexdigest()
#         except Exception:
#             pass

#         return hashlib.sha1(repr(obj).encode()).hexdigest()

#     parts = {
#         "version": _CACHE_VERSION,
#         "video": _video_fingerprint(basedir),
#         "tasks": tuple(_stable_task_repr(t) for t in tasks),
#         "frames_from_annotation": _hash_annotation_frames(frames_from_annotation),
#         "first_frame_number": first_frame_number,
#         "last_frame_number": last_frame_number,
#         "image_format": image_format,
#     }

#     blob = repr(sorted(parts.items())).encode()
#     return hashlib.sha1(blob).hexdigest()