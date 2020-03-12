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
    from .version import VERSION as __version__
    from .version import REVISION as __revision__
except ImportError:
    __version__ = "unbuilt-dev"
    __revision__ = "unbuilt-dev"

LOG = logging.getLogger(__name__)
LOG.info("Starting JUTIL V%s REV%s", __version__, __revision__)
