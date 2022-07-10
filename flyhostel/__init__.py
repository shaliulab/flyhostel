__version__ = '1.1.3'

import os
import sys
sys.path.insert(0, os.getcwd())
from confapp import conf

try:
    import local_settings # type: ignore
    conf += local_settings
except:
    pass

import flyhostel.constants
conf += flyhostel.constants