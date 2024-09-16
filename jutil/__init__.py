#
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Ungermann
#

import logging

from . import cg
from . import cgne
from . import costfunction
from . import diff
from . import fft
from . import linalg
from . import lnsrch
from . import lsqr
from . import minimizer
from . import misc
from . import norms
from . import operator
from . import preconditioner
from . import splitbregman
from . import taketime
try:
    from ._version import __version__
except ImportError:
    __version__ = "unbuilt-dev"

LOG = logging.getLogger(__name__)
LOG.info("Starting JUTIL V%s", __version__)
