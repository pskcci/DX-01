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

        # while(1):
            # ctrl.orange = True
            # ctrl.green = True
            # ctrl.buzzer = True
            # ctrl.conveyor = True
            
            # num = int(input("3~8 중 1개를 입력하세요(종료는 0): "))
            # if num == 3 :
            #     ctrl.orange = False
            # elif num == 4:
            #     ctrl.green = False
            # elif num == 5:
            #     ctrl.buzzer = False
            # elif num == 8:
            #     ctrl.conveyor = False
        
        ctrl.orange = True
        ctrl.green = True
        ctrl.buzzer = True
        ctrl.conveyor = True

        ctrl.close()

if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()