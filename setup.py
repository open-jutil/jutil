#!python
descr = """Gloripy
ToDo
"""


DISTNAME = 'jutil'
DESCRIPTION = 'Juelich Tomographic Inversion Library'
LONG_DESCRIPTION = descr
MAINTAINER = 'Joern Ungermann'
MAINTAINER_EMAIL = 'j.ungermann@fz-juelich.de'
URL = ''
VERSION = '0.1.0-dev'
PYTHON_VERSION = (2, 7)
DEPENDENCIES = {
    'numpy': (1, 6),
    'cython': (0, 19),
}

import os
import subprocess
import sys
import re
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
        HG_REV = subprocess.check_output(['hg', 'id', '--id']).strip()
    except:
        HG_REV = "???"
    return HG_REV


def write_version_py(filename='jutil/version.py'):
    version_string = "# THIS FILE IS GENERATED FROM THE JUTIL SETUP.PY\n" \
        'version = "{}"\n' \
        'HG_REVISION = "{}"\n'.format(VERSION, hg_version())
    with open(os.path.join(os.path.dirname(__file__),
                           filename), 'w') as vfile:
        vfile.write(version_string)


def get_package_version(package):
    version = []
    for version_attr in ('version', 'VERSION', '__version__'):
        if (hasattr(package, version_attr) and
                isinstance(getattr(package, version_attr), str)):
            version_info = getattr(package, version_attr, '')
            for part in re.split('\D+', version_info):
                try:
                    version.append(int(part))
                except ValueError:
                    pass
    return tuple(version)


def check_requirements():
    if sys.version_info < PYTHON_VERSION:
        raise SystemExit('You need Python version %d.%d or later.'
                         % PYTHON_VERSION)

    for package_name, min_version in DEPENDENCIES.items():
        dep_error = False
        try:
            package = __import__(package_name)
        except ImportError:
            dep_error = True
        else:
            package_version = get_package_version(package)
            if min_version > package_version:
                dep_error = True

        if dep_error:
            raise ImportError('You need `%s` version %d.%d or later.'
                              % ((package_name, ) + min_version))


if __name__ == "__main__":
    check_requirements()

    write_version_py()

    setup(
        name=DISTNAME,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        url=URL,
        download_url=URL,
        version=VERSION,
        test_suite="nose.collector",

        classifiers=[
            'Development Status :: 3 - alpha',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Programming Language :: C',
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
