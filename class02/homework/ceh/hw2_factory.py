#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep
import cv2
import numpy as np
from openvino.runtime import Core
from iotdemo import FactoryController
from iotdemo.motion.motion_detector import MotionDetector
from iotdemo.color.color_detector import ColorDetector
from pathlib import Path
import openvino as ov
from typing import Tuple

video = "resources/conveyor.mp4"

#FORCE_STOP은 프로그램에서 **스레드나 작업을 강제로 중지하기 위해 사용하는 플래그(flag)**입니다. 
#FORCE_STOP가 True면, 해당 값을 주기적으로 확인하는 스레드나 루프가 이를 감지하고 실행을 중단합니다.
FORCE_STOP = False

base_model_dir = Path("resources")
detection_model_name = "model"
model_xml = base_model_dir / f"{detection_model_name}.xml"
model_bin = base_model_dir / f"{detection_model_name}.bin"
core = ov.Core()

cam1_detect = 0
cam2_detect = 0

def model_init(model_path: str) -> Tuple:
    model = core.read_model(model=model_path)
    compiled_model = core.compile_model(model=model, device_name="CPU")
    input_keys = compiled_model.input(0)
    output_keys = compiled_model.output(0)
    return input_keys, output_keys, compiled_model

def thread_cam1(q):#motion  -- o,x추론
    global cam1_detect 
    motion1 = MotionDetector()
    motion1.load_preset("resources/motion.cfg")
    input_keys_mo, output_keys_mo, compiled_model_mo = model_init(model_xml)


    # TODO: HW2 Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture(video)
    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break
        # TODO: HW2 Enqueue "VIDEO:Cam1 live", frame info
        q.put(('VIDEO:Cam1 live', frame))


        # TODO: Motion detect
        detected1 = motion1.detect(frame)

        #print(frame)
        #print(detected1)
        if detected1 is None:
            pass
        else:
            q.put(('VIDEO:Cam1 detected', detected1))
            cam1_detect += 1


        # TODO: Calculate ratios
        #print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        # TODO: in queue for moving the actuator 1


    cap.release()
    q.put(('DONE', None))
    exit()


def thread_cam2(q):#motion  -- color추론9e
    global cam2_detect
    motion2 = MotionDetector()
    motion2.load_preset("resources/motion.cfg")
    color = ColorDetector()

    # TODO: HW2 Open "resources/conveyor.mp4" video clip
    cap2 = cv2.VideoCapture(video)
    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap2.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam2 live", frame info
        q.put(('VIDEO:Cam2 live', frame))

        # TODO: Detect motion
        detected2 = motion2.detect(frame)
        if detected2 is None:
            pass
        else:
            q.put(('VIDEO:Cam2 detected', detected2))
            check_color = color.detect(frame)
            print(check_color)
            cam2_detect += 1


        # TODO: Enqueue "VIDEO:Cam2 detected", detected info.

        # TODO: Detect color
        #findColor =  color.detect(frame)
        #print(findColor)
        # TODO: Compute ratio
        #print(f"{name}: {ratio:.2f}%")

        # TODO: Enqueue to handle actuator 2


    cap2.release()
    q.put(('DONE', None))
    exit()


def imshow(title, frame, pos=None):
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)


def main():
    global FORCE_STOP

    q = Queue()

    parser = ArgumentParser(prog='python3 factory.py',
                            description="Factory tool")

    parser.add_argument("-d",
                        "--device",
                        default=None,
                        type=str,
                        help="Arduino port")
    args = parser.parse_args()

    t1 = threading.Thread(target=thread_cam1, args=(q,))
    t2 = threading.Thread(target=thread_cam2, args=(q,))

    t1.start()
    t2.start()
    # TODO: HW2 Create a Queue
    # TODO: HW2 Create thread_cam1 and thread_cam2 threads and start them.
    motion1 = MotionDetector()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                FORCE_STOP = True
                break

            name, frame = q.get() 

            # TODO: HW2 show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.
            if name == 'VIDEO:Cam1 live':
                imshow('Cam1 live', frame, pos=(100, 100))

            if name == 'VIDEO:Cam1 detected':
                imshow(f'Cam1 live - Detected: {cam1_detect}', frame, pos=(100, 700))

            if name == 'VIDEO:Cam2 live':
                imshow('Cam2 live', frame, pos=(800, 100))

            if name == 'VIDEO:Cam2 detected':
                imshow(f'Cam2 live - Detected: {cam2_detect}', frame, pos=(800, 700))


            # TODO: HW2 get an item from the queue. You might need to properly handle exceptions.
            # de-queue name and data

            # TODO: Control actuator, name == 'PUSH'

            if name == 'DONE':
                FORCE_STOP = True

            q.task_done()
    t1.join()
    t2.join()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()