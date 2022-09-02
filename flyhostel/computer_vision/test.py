from codetiming import Timer
import time

with Timer(text="something took {:.8f}", logger=logger.debug):
    time.sleep(1)
