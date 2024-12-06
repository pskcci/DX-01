import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep
from iotdemo import FactoryController

def main():
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
        ctrl.orange = True
        ctrl.green = True
        ctrl.conveyor = True
        while True:
            ctrl.conveyor = False
            print("1 = R / 2 = Orag / 3 = Gr / 4 = End")
            num = int(input("num = "))
            if  num == 1:
               ctrl.red = False
               ctrl.orange = True
               ctrl.green = True
            elif num == 2:
               ctrl.red = True
               ctrl.orange = False
               ctrl.green = True
            elif num == 3:
               ctrl.red = True
               ctrl.orange = True
               ctrl.green = False
            elif num == 4:
                ctrl.red = True
                ctrl.orange = True
                ctrl.green = True
                ctrl.conveyor = True
                break
            else:
                pass

    ctrl.close()

if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()