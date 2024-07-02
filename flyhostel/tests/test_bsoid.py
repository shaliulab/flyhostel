import sqlite3
import numpy as np
import pytest
import sys
import shutil
import os
import logging
sys.path.append("/home/vibflysleep/opt/B-SOID/")
from flyhostel.data.pose import parse_number_of_animals
import codetiming

dfbile = "flyhostel.db"


def test_parse_number_of_animals():
    with sqlite3.connect(dbfile) as conn:
        cur=conn.cursor()
        number_of_animals=parse_number_of_animals(cur)

        assert number_of_animals==5

    
if __name__ == "__main__":
    test_parse_number_of_animals()
