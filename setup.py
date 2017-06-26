#!/bin/env python
DISTNAME = 'jutil'
DESCRIPTION = 'Juelich Tomographic Inversion Library'
MAINTAINER = 'Joern Ungermann'
MAINTAINER_EMAIL = 'j.ungermann@fz-juelich.de'
VERSION = '0.2.0-dev'

import os
import subprocess
import setuptools
from distutils.core import setup
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


def git_version():
    try:
        git_rev = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
    except:
        git_rev = "???"
    return git_rev


def write_version_py(filename='jutil/version.py'):
    version_string = "# THIS FILE IS GENERATED FROM THE JUTIL SETUP.PY\n" + \
        'VERSION = "{}"\n'.format(VERSION) + \
        'GIT_REVISION = "{}"\n'.format(git_version())
    with open(os.path.join(os.path.dirname(__file__), filename), 'w') as vfile:
        vfile.write(version_string)


if __name__ == "__main__":
    write_version_py()

    setup(
        name=DISTNAME,
        description=DESCRIPTION,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        version=VERSION,
        setup_requires=["numpy>=1.6", "pytest-runner", "pytest-cov", "pytest-flake8"],
        tests_require=['pytest', 'nose'],
        install_requires=["numpy>=1.6", "tqdm", "scipy"],
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
