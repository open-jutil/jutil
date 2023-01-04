import argparse
import shutil
import glob
import os
import stat
import tempfile
import sys
import subprocess as sp


def git_info():
    git_rev = sp.Popen(['git', 'describe', '--always', '--dirty'], stdout=sp.PIPE).communicate()[0].decode().strip()
    git_count = sp.Popen(['git', 'rev-list', 'HEAD', '--count'], stdout=sp.PIPE).communicate()[0].decode().strip()
    git_count_str = str(int(git_count)).zfill(5)
    return git_rev, git_count_str


def _main(args):
    print("make_exe.py")
    git_rev, git_count_str = git_info()
    identifier = f"V{git_count_str}_R{git_rev}"
    if "dirty" in git_rev:
        raise RuntimeError("There are uncommited changes in the repository!")
    newfexe = os.path.abspath(
        os.path.join(args.directory, f"jutil_{identifier}"))
    if os.path.exists(newfexe):
        raise RuntimeError("'{}' exists already?".format(newfexe))
    os.system("python setup.py bdist_egg")
    tmpdir = tempfile.mkdtemp()
    olddir = os.getcwd()
    eggs = glob.glob(os.path.abspath(os.path.join("dist", "jutil-*.egg")))
    if len(eggs) == 0:
        print("There was a problem with the build process, no egg found! Abort.")
        sys.exit(1)
    if len(eggs) > 1:
        print("There are multiple eggs presents. Please remove all and repeat the action. Abort.")
        sys.exit(1)
    egg = eggs[0]
    os.chdir(tmpdir)
    os.system("unzip " + egg)
    for fn in (glob.glob(os.path.join("EGG-INFO", "scripts", "*py"))
               + glob.glob(os.path.join("EGG-INFO", "scripts", "*sh"))):
        shutil.copyfile(fn, os.path.basename(fn))
        os.chmod(os.path.basename(fn),
                 stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP |
                 stat.S_IROTH | stat.S_IXOTH)
    os.chdir(olddir)
    shutil.move(tmpdir, newfexe)
    os.chmod(newfexe,
             stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP |
             stat.S_IROTH | stat.S_IXOTH)
    print("Created new fexe directory at {}".format(newfexe))


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="""
        Create a directory containing all jutil scripts and modules. The
        scripts may be directly called from that directory.
    """)
    _parser.add_argument("directory", help="directory to place jutil EXE in.")
    _main(_parser.parse_args())
