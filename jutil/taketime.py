import time
import logging


class TakeTime(object):
    """
    Measure the time consumed by an entire block and write to stdout.::

        with TakeTime("Doing something") as timer:
            do_something
            if timer.dt > time_limit:
                print("Oh no, running late already!")
            do_even_more

    The 'as' part is optional.
    """

    def __init__(self, name):
        self._name = name
        self.msg = ""
        self._log = logging.getLogger(__name__)

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self._log.info("{}: {:.3f} sec    {}".format(self._name, self.dt, self.msg))

    @property
    def dt(self):
        """
        returns time passes sind the timer was started.
        """
        return time.time() - self.t0

