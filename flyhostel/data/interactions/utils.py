import sqlite3
from flyhostel.data.interactions.load_data import get_sqlite_file
def load_metadata_prop(prop, animal=None, dbfile=None):

    if dbfile is None:
        dbfile = get_sqlite_file(animal)

    with sqlite3.connect(dbfile) as connection:
        cursor = connection.cursor()
        cursor.execute(f"SELECT value FROM METADATA WHERE field = '{prop}';")
        prop = cursor.fetchone()[0]
    return prop

def load_roi_width(dbfile):
    with sqlite3.connect(dbfile) as conn:
        cursor=conn.cursor()

        cursor.execute(
            """
        SELECT w FROM ROI_MAP;
        """
        )
        [(roi_width,)] = cursor.fetchall()
        cursor.execute(
            """
        SELECT h FROM ROI_MAP;
        """
        )
        [(roi_height,)] = cursor.fetchall()

    roi_width=int(roi_width)
    roi_height=int(roi_height)
    roi_width=max(roi_width, roi_height)
    return roi_width

def parse_identity(id):
    return int(id.split("|")[1])
