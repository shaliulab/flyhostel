from abc import ABC
import sqlite3
import os.path
import pandas as pd

class ConcatenationExporter(ABC):

    _basedir = None

    def init_concatenation_table(self, dbfile, reset=True):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            if reset:
                cur.execute("DROP TABLE IF EXISTS CONCATENATION;")
            cur.execute("CREATE TABLE IF NOT EXISTS CONCATENATION (chunk int(3), in_frame_index int(2), in_frame_index_after int(2), local_identity int(2), local_identity_after int(2), identity int(2));")

    def write_concatenation_table(self, dbfile, chunks):
        csv_file = os.path.join(self._basedir, "idtrackerai", "concatenation-overlap.csv")

        if os.path.exists(csv_file):
            concatenation=pd.read_csv(csv_file, index_col=0)
            with sqlite3.connect(dbfile, check_same_thread=False) as conn:
                cur = conn.cursor()
                for _, row in concatenation.iterrows():
                    args = (
                        row["chunk"], row["in_frame_index_before"], row["in_frame_index_after"],
                        row["local_identity"], row["local_identity_after"], row["identity"]
                    )
                    args=tuple([e.item() for e in args])
                    chunk=args[0]
                    if chunk in chunks:
                        cur.execute(
                            "INSERT INTO CONCATENATION (chunk, in_frame_index, in_frame_index_after, local_identity, local_identity_after, identity) VALUES (?, ?, ?, ?, ?, ?);",
                            args
                        )
                    else:
                        pass
        else:
            if "1X" in self._basedir:
                with sqlite3.connect(dbfile, check_same_thread=False) as conn:
                    cur=conn.cursor()
                    for chunk in chunks:
                        cur.execute(
                                "INSERT INTO CONCATENATION (chunk, in_frame_index, in_frame_index_after, local_identity, local_identity_after, identity) VALUES (?, ?, ?, ?, ?, ?);",
                                (chunk, 0, 0, 1, 1, 1)
                            )

            else:
                raise FileNotFoundError(
                    f"""concatenation_overlap.csv not found.
                    Please make sure idtrackerai_concatenation step is run for {self._basedir}
                    """
                )

        print("CONCATENATION table done")