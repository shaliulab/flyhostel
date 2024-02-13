import logging
import sqlite3
from sqlalchemy import create_engine

logger=logging.getLogger(__name__)

def write_validated_roi0(df, dbfile):
    engine = create_engine(f"sqlite:///{dbfile}")
    df["area"]=0
    df_roi0=df[["frame_number", "in_frame_index", "x", "y", "fragment", "area", "modified", "class_name", "validated"]].reset_index(drop=True)
    table_name='ROI_0_VAL'
    
    logger.debug("Writing %s to %s", table_name, dbfile)  
    df_roi0.to_sql(
        table_name, con=engine,
        if_exists='fail',
        index=True, index_label="id"
    )

    logger.debug("Writing index;")
    cursor.execute("CREATE INDEX ROI0_val_fn ON ROI_0_VAL (frame_number);")
    return df_roi0

def write_validated_identity(df, dbfile):
    engine = create_engine(f"sqlite:///{dbfile}")
    df_identity=df[["frame_number", "in_frame_index", "local_identity", "identity", "validated"]].reset_index(drop=True)
    table_name='IDENTITY_VAL'
    
    logger.debug("Writing %s to %s", table_name, dbfile)
    
    df_identity.to_sql(
        table_name, con=engine,
        if_exists='fail',
        index=True, index_label="id",
    )

    logger.debug("Writing index;")
    with sqlite3.connection(dbfile) as conn:
        cursor=conn.cursor()
        cursor.execute("CREATE INDEX id_val_fn ON IDENTITY_VAL (frame_number);")
        cursor.execute("CREATE INDEX id_val_lid ON IDENTITY_VAL (local_identity);")

    return df_identity