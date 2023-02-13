import threading
import sqlite3
import time

class AsyncSQLiteWriter(threading.Thread):

    _MIN_QUEUE_SIZE=100

    def __init__(self, dbfile, table_name, queue, stop_event, *args, **kwargs):

        self._queue=queue
        self._table_name = table_name
        self._dbfile = dbfile
        self._stop_event = stop_event
        super(AsyncSQLiteWriter, self).__init__(*args, **kwargs)

    @property
    def needs_flushing(self):
        return self._queue.qsize() > self._MIN_QUEUE_SIZE


    def flush(self):

        queue_size=self._queue.qsize()
        data=[]
        for _ in range(queue_size):
            data.append(str(self._queue.get()))

        value_string=",".join(data)        
        before=time.time()
        with sqlite3.connect(self._dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            cur.execute(f"INSERT INTO {self._table_name} VALUES {value_string};")
        after=time.time()
        print(f"Wrote {queue_size} rows in {after-before} seconds")

    def run(self):
        while not self._stop_event.is_set():
            if self.needs_flushing:
                self.flush()

        self.flush()
