#!/bin/env python
DISTNAME = 'jutil'
DESCRIPTION = 'Juelich Tomographic Inversion Library'
MAINTAINER = 'Joern Ungermann'
MAINTAINER_EMAIL = 'j.ungermann@fz-juelich.de'
VERSION = '0.1.0-dev'

import os
import subprocess
import setuptools
from numpy.distutils.core import setup
from distutils.command.build_py import build_py


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=True)

    config.add_subpackage('jutil')

    return config


def hg_version():
    try:
        hg_rev = subprocess.check_output(['hg', 'id', '--id']).strip()
    except:
        hg_rev = "???"
    return hg_rev


def write_version_py(filename='jutil/version.py'):
    version_string = "# THIS FILE IS GENERATED FROM THE JUTIL SETUP.PY\n" + \
        'version = "{}"\n'.format(VERSION) + \
        'HG_REVISION = "{}"\n'.format(hg_version())
    with open(os.path.join(os.path.dirname(__file__),
                           filename), 'w') as vfile:
        vfile.write(version_string)


if __name__ == "__main__":
    write_version_py()

    setup(
        name=DISTNAME,
        description=DESCRIPTION,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        version=VERSION,
        test_suite="nose.collector",
        setup_requires=["numpy>=1.6",
                        "nose"],
        install_requires=["numpy>=1.6"],

        classifiers=[
            'Development Status :: 3 - alpha',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
        ],

        configuration=configuration,

        packages=setuptools.find_packages(exclude=['doc']),
        include_package_data=True,
        zip_safe=False,
        cmdclass={'build_py': build_py},
    )
