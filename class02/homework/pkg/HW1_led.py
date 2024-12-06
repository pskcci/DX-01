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

    with FactoryController('/dev/ttyACM0') as ctrl:
        ctrl.red = False
        ctrl.orange = True
        ctrl.green = False
        ctrl.conveyor = False
        ctrl.push_actuator(0)
        ctrl.close()

if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()