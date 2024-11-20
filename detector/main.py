import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import set_repo_root

import threading
import traceback

from detector.app import App


def global_exception_handler(exc_type, exc_value, exc_tb):
    print("An unhandled exception occurred!")
    print(f"Exception type: {exc_type}")
    print(f"Exception message: {exc_value}")
    traceback.print_tb(exc_tb)
    
    sys.exit(1)


def main():
    sys.excepthook

    app = App()
    app.run()


if __name__ == '__main__':
    main()