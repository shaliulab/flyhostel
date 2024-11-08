import logging
import sqlite3
from sqlalchemy import create_engine

logger=logging.getLogger(__name__)

IF_EXISTS="replace"

def write_validated_roi0(df, dbfile):
    engine = create_engine(f"sqlite:///{dbfile}")
    df["area"]=0
    table_name='ROI_0_VAL'

    logger.debug("Writing %s to %s", table_name, dbfile)

    df.to_sql(
        table_name, con=engine,
        if_exists=IF_EXISTS,
        index=True, index_label="id"
    )

    logger.debug("Writing index;")
    with sqlite3.connect(dbfile) as conn:
        cursor=conn.cursor()
        cursor.execute("CREATE INDEX ROI0_val_fn ON ROI_0_VAL (frame_number);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_roi_0_val_frame_index ON ROI_0_VAL (frame_number, in_frame_index);")
    return df


def write_validated_identity(df, dbfile):
    engine = create_engine(f"sqlite:///{dbfile}")

    table_name='IDENTITY_VAL'

    logger.debug("Writing %s to %s", table_name, dbfile)
    df.to_sql(
        table_name, con=engine,
        if_exists=IF_EXISTS,
        index=True, index_label="id",
    )

    logger.debug("Writing index;")
    with sqlite3.connect(dbfile) as conn:
        cursor=conn.cursor()
        cursor.execute("CREATE INDEX id_val_fn ON IDENTITY_VAL (frame_number);")
        cursor.execute("CREATE INDEX id_val_lid ON IDENTITY_VAL (local_identity);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_identity_val_frame_index_identity ON IDENTITY_VAL (frame_number, in_frame_index, identity);")

    return df


def write_validated_concatenation(df_concatenation, dbfile):
    engine = create_engine(f"sqlite:///{dbfile}")
    table_name='CONCATENATION_VAL'

    logger.debug("Writing %s to %s", table_name, dbfile)

    df_concatenation.to_sql(
        table_name, con=engine,
        if_exists=IF_EXISTS,
        index=True, index_label="id",
    )
