import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import set_repo_root

from detector.app import App


def run_app():
    app = App()
    app.run()


if __name__ == '__main__':
    run_app()