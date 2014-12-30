# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Ungermann
#
"""Docstring Jutil

"""  # TODO: Docstring!

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
from .version import version as __version__
from .version import hg_revision as __hg_revision__

_log = logging.getLogger(__name__)
_log.info("Starting JUTIL V{} HG{}".format(__version__, __hg_revision__))
