import sqlite3
from flyhostel.data.interactions.load_data import get_sqlite_file
def get_metadata_prop(animal, prop):
    with sqlite3.connect(get_sqlite_file(animal)) as connection:
        cursor = connection.cursor()
        cursor.execute(f"SELECT value FROM METADATA WHERE field = '{prop}';")
        prop = cursor.fetchone()[0]
    return prop