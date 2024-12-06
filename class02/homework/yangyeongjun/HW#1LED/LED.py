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

    # TODO: HW2 Create a Queue

    # TODO: HW2 Create thread_cam1 and thread_cam2 threads and start them.

    with FactoryController(args.device) as ctrl:
        ctrl.red = True
        pb1 = True
        pb2 = True 
        pb3 = True
        while True:
            
            n = int(input("red=3/orange=4/green=5 : "))
            if n == 3:
                pb1 = not pb1
                
            elif n == 4:
                pb2 = not pb2
                
            elif n == 5:
                pb3 = not pb3
            else:
                pass
            ctrl.red = pb1
            ctrl.orange = pb2
            ctrl.green = pb3
            n = 0
                


            



if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()
