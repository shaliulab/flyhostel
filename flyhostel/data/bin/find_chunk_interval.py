import argparse
import os.path
import sqlite3

from flyhostel.data.pose.main import FlyHostelLoader
from flyhostel.data.pose.export import load_concatenation_table, parse_number_of_animals

def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("basedir", type=str)
    ap.add_argument("flyhostel_db", type=str)
    return ap

def main():

    ap = get_parser()

    args=ap.parse_args()
    basedir = args.basedir
    flyhostel_db = args.flyhostel_db

    experiment2="_".join(basedir.split(os.path.sep)[-3:])

    with sqlite3.connect(f'file:{flyhostel_db}?mode=ro', uri=True) as conn:
        cur=conn.cursor()
        number_of_animals=parse_number_of_animals(cur)
    identities=[number_of_animals-1]

    loader=FlyHostelLoader(
        experiment=experiment2,
        identity=identities[0],
        chunks=range(0, 400),
        identity_table="IDENTITY",
        roi_0_table="ROI_0"
    )
        
    metadata=loader.get_simple_metadata().iloc[0]

    if metadata["number_of_animals"]==1:
        concatenation_table_name="CONCATENATION"
    else:
        concatenation_table_name="CONCATENATION_VAL"

    with sqlite3.connect(f'file:{flyhostel_db}?mode=ro', uri=True) as conn:
        cur=conn.cursor()
        concatenation_table=load_concatenation_table(cur, basedir, concatenation_table=concatenation_table_name)
        start_chunk=concatenation_table["chunk"].iloc[0]
        end_chunk=concatenation_table["chunk"].iloc[-1]

    with open('chunks.txt', 'w') as fout:
        for chunk in range(start_chunk, end_chunk+1):
            fout.write(f"{basedir},{flyhostel_db},{concatenation_table_name},{str(chunk).zfill(6)}\n")

if __name__ == "__main__":
    main()