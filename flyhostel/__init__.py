__version__ = '1.1.3'

from confapp import conf
import os
import sys
sys.path.append(os.getcwd())

try:
    import local_settings
    conf += local_settings
except:
    pass

conf += "flyhostel.constants"
