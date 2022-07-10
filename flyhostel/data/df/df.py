import os

FACTOR={"GB": 1024**3, "MB": 1024**2, "KB": 1024}

def unit(bytes_, size_unit="GB"):
    return bytes_ / FACTOR[size_unit]


def size_free(statvfs):
   return round(unit(statvfs.f_frsize*statvfs.f_bavail), 2)


def size_total(statvfs):
   return round(unit(statvfs.f_frsize*statvfs.f_blocks), 2)


def get_free_fraction(path):
    statvfs = os.statvfs(path)
    fraction = size_free(statvfs) / size_total(statvfs)
    return fraction
