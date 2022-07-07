__version__ = '1.1.3'

from confapp import conf
conf += "idtrackerai.constants"

try:
    import local_settings #type: ignore
    conf += local_settings

except Exception:
    pass