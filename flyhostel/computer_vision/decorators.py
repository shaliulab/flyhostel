from functools import wraps
from time import time

def timeit(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        ts = time()
        result = f(*args, **kwargs)
        te = time()
        msec = (te-ts)*1000
        return result, msec
    
    wrap.unwrapped = f
    return wrap