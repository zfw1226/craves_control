import os, sys
import errno
import time

def mkdir_p(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def isfile(fname):
    return os.path.isfile(fname) 

def isdir(dirname):
    return os.path.isdir(dirname)

def join(path, *paths):
    return os.path.join(path, *paths)


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exception_type, exception_value, traceback):
        elapse = time.time() - self.start
        # print(self.name + ': Elapsed time: ' + str(elapse))
        sys.stdout.flush()