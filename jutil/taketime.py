import time

class TakeTime(object):
    """
    Measure the time consumed by an entire block and write to stdout.

    Usage:

    with TakeTime("Doing something") as timer:
        do_something
        if timer.dt > time_limit:
            print("Oh no, running late already!")
        do_even_more

    The 'as' part is optional.
    """

    def __init__(self, name):
        self.name = name
        self.msg = ""

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, type, value, traceback):
        print("%s: %.3f sec    %s" % (self.name, self.dt, self.msg))

    @property
    def dt(self):
        return time.time() - self.t0

