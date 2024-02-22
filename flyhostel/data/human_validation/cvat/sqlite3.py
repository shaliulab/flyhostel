import logging
import sqlite3
from sqlalchemy import create_engine

logger=logging.getLogger(__name__)

def write_validated_roi0(df, dbfile):
    engine = create_engine(f"sqlite:///{dbfile}")
    df["area"]=0
    table_name='ROI_0_VAL'
    
    logger.debug("Writing %s to %s", table_name, dbfile)  
    df.to_sql(
        table_name, con=engine,
        if_exists='fail',
        index=True, index_label="id"
    )

    logger.debug("Writing index;")
    with sqlite3.connection(dbfile) as conn:
        cursor=conn.cursor()
        cursor.execute("CREATE INDEX ROI0_val_fn ON ROI_0_VAL (frame_number);")

    return df


def write_validated_identity(df, dbfile):
    engine = create_engine(f"sqlite:///{dbfile}")
    
    table_name='IDENTITY_VAL'
    
    logger.debug("Writing %s to %s", table_name, dbfile)
    
    df.to_sql(
        table_name, con=engine,
        if_exists='fail',
        index=True, index_label="id",
    )

    logger.debug("Writing index;")
    with sqlite3.connection(dbfile) as conn:
        cursor=conn.cursor()
        cursor.execute("CREATE INDEX id_val_fn ON IDENTITY_VAL (frame_number);")
        cursor.execute("CREATE INDEX id_val_lid ON IDENTITY_VAL (local_identity);")

    return df


def write_validated_concatenation(df_concatenation, dbfile):
    engine = create_engine(f"sqlite:///{dbfile}")
    table_name='CONCATENATION_VAL'
    
    logger.debug("Writing %s to %s", table_name, dbfile)
    
    df_concatenation.to_sql(
        table_name, con=engine,
        if_exists='fail',
        index=True, index_label="id",
    )