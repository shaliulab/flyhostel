import os.path

def add_suffix(filename, suffix=""):

    if suffix != "":
        basename, ext = os.path.splitext(filename)
        filename = basename + "_" + suffix + ext
    
    return filename

    

