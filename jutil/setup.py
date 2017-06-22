#
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Ungermann
#

import os


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('jutil', parent_package, top_path)

    base_path = os.path.join(os.getcwd(), config.local_path)
    mod = [d for d in os.listdir(base_path) if '.' not in d]

    for m in mod:
        config.add_subpackage(m)

        if os.path.isdir(os.path.join(config.local_path, m, 'tests')):
            add_path = os.path.join(m, 'tests')
            # print(add_path)
            config.add_data_dir(add_path)

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    config = configuration(top_path='').todict()
    setup(**config)
