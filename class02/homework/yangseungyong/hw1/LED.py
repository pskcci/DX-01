import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import cv2
import numpy as np

from iotdemo import FactoryController

def main():
    global FORCE_STOP

    parser = ArgumentParser(prog='python3 factory.py',
                            description="Factory tool")

    parser.add_argument("-d",
                        "--device",
                        default=None,
                        type=str,
                        help="Arduino port")
    args = parser.parse_args() 
    with FactoryController(args.device) as ctrl:
        ctrl.red = True
        ctrl.orange = False
        ctrl.green = True

        ctrl.close()

if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()
