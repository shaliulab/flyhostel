from abc import ABC
import sqlite3
import os.path
import pandas as pd

class ConcatenationExporter(ABC):

    _basedir = None
    cmd = "INSERT INTO CONCATENATION (chunk, in_frame_index, in_frame_index_after, local_identity, local_identity_after, is_inferred, is_broken, identity) VALUES (?, ?, ?, ?, ?, ?, ?, ?);"


    def init_concatenation_table(self, dbfile, reset=True):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            if reset:
                print("Dropping CONCATENATION")
                cur.execute("DROP TABLE IF EXISTS CONCATENATION;")
            cur.execute("CREATE TABLE IF NOT EXISTS CONCATENATION (id INTEGER PRIMARY KEY AUTOINCREMENT, chunk int(3), in_frame_index int(2), in_frame_index_after int(2), local_identity int(2), local_identity_after int(2), is_inferred int(1), is_broken int(1), identity int(2));")

    def write_concatenation_table(self, dbfile, chunks):
        csv_file = os.path.join(self._basedir, "idtrackerai", "concatenation-overlap.csv")

        if os.path.exists(csv_file):
            concatenation=pd.read_csv(csv_file, index_col=0)
            data=[]
            with sqlite3.connect(dbfile, check_same_thread=False) as conn:
                cur = conn.cursor()
                for _, row in concatenation.iterrows():
                    args = (
                        row["chunk"], row["in_frame_index_before"], row["in_frame_index_after"],
                        row["local_identity"], row["local_identity_after"],
                        row["is_inferred"], row["is_broken"],
                        row["identity"],
                    )
                    args2=[]
                    for e in args:
                        try:
                            e = e.item()
                        except AttributeError:
                            pass
                        args2.append(e)
                    args=tuple(args2)
                    chunk=args[0]
                    if chunk in chunks:
                        data.append(args)
                    else:
                        pass

                if data:
                    conn.executemany(
                        self.cmd,
                        data
                    )

        else:
            if "1X" in self._basedir:
                with sqlite3.connect(dbfile, check_same_thread=False) as conn:
                    cur=conn.cursor()
                    for chunk in chunks:
                        cur.execute(
                                self.cmd
                                (chunk, 0, 0, 1, 1, 0, 0, 1)
                            )

            else:
                raise FileNotFoundError(
                    f"""concatenation_overlap.csv not found.
                    Please make sure idtrackerai_concatenation step is run for {self._basedir}
                    """
                )
