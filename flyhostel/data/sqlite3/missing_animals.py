import argparse
import os.path
import sqlite3

import numpy as np

def get_parser():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dbfile")
    return ap

def list_frames_with_no_animals(dbfile):

    with sqlite3.connect(dbfile) as conn:
        cursor=conn.cursor()
        
        cursor.execute("SELECT value FROM METADATA where field = 'chunksize';")
        chunksize=int(float(cursor.fetchone()[0]))


        cursor.execute("""
            SELECT IDX.frame_number
                FROM STORE_INDEX IDX
                LEFT JOIN ROI_0 R0 ON IDX.frame_number = R0.frame_number
                WHERE R0.frame_number IS NULL;
        """)

        frames = [row[0] for row in cursor.fetchall()]
 
    chunks = [frame// chunksize for frame in frames]
    chunks, counts = np.unique(chunks, return_counts=True)
    for i, chunk in enumerate(chunks):
        print(chunk, ": ", counts[i])


def main():
    ap = get_parser()
    args=ap.parse_args()
    assert os.path.exists(args.dbfile)
    list_frames_with_no_animals(args.dbfile)

if __name__ == "__main__":
    main()
