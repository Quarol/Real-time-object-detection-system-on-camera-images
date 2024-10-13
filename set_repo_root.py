import os
import sys

def set_repo_root():
    repo_root = os.path.abspath(os.path.dirname(__file__))
    os.chdir(repo_root)
    sys.path.append(repo_root)

    return repo_root

set_repo_root()