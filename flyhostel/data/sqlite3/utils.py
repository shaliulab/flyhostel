import sqlite3
import cv2

def table_is_not_empty(dbfile, table_name):
    """
    Returns True if the passed table has at least one row
    """

    with sqlite3.connect(dbfile, check_same_thread=False) as conn:
        cur=conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        count=cur.fetchone()[0]

    return count > 0


def ensure_type(var, var_name, typ):

    if not isinstance(var, typ):
        try:
            var=typ(var)
        except ValueError as exc:
            raise ValueError(f"{var_name} {var} with type {type(var)} cannot be coerced to {typ}") from exc

    return var

def serialize_arr(arr, path):
    """
    Transform an image (np.array) to bytes for export in SQLite
    """

    cv2.imwrite(path, arr, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

    with open(path, "rb") as filehandle:
        bstring = filehandle.read()

    return bstring
