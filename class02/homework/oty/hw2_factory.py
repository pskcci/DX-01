#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep
import queue

import cv2
import numpy as np

from iotdemo import FactoryController, MotionDetector

FORCE_STOP = False


def thread_cam1(q):
    # TODO: MotionDetector
    cam01 = MotionDetector()
    cam01.load_preset("/home/intel/git-training/DX-01/class02/smart-factory/resources/motion.cfg")
    # TODO: Load and initialize OpenVINO

    # TODO: HW2 Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture("smart-factory/resources/conveyor.mp4")
    print("cam1")
    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam1 live", frame info
        q.put(('VIDEO:cam1 live',frame))
        # TODO: Motion detect
        detected = cam01.detect(frame)
        if detected is None:
            continue
        
        q.put(('VIDEO:Cam1 detected', detected))
        
        # TODO: Enqueue "VIDEO:Cam1 detected", detected info.
        # # abnormal detect
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # reshaped = detected[:, :, [2, 1, 0]]
        # np_data = np.moveaxis(reshaped, -1, 0)
        # preprocessed_numpy = [((np_data / 255.0) - 0.5) * 2]
        # batch_tensor = np.stack(preprocessed_numpy, axis=0)
        # TODO: Inference OpenVINO

        # # TODO: Calculate ratios
        # print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        # TODO: in queue for moving the actuator 1

    cap.release()
    q.put(('DONE', None))
    exit()

def thread_cam2(q):
    # TODO: MotionDetector
    cam02 = MotionDetector()
    cam02.load_preset("/home/intel/git-training/DX-01/class02/smart-factory/resources/motion.cfg")
    # TODO: ColorDetector


    # TODO: HW2 Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture("smart-factory/resources/conveyor.mp4")
    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam2 live", frame info
        q.put(('VIDEO:CAM2 live',frame))

        # TODO: Detect motion
        detected = cam02.detect(frame)
        if detected is None:
            continue
        # TODO: Enqueue "VIDEO:Cam2 detected", detected info.
        q.put(('VIDEO:Cam2 detected', detected))
        # TODO: Detect color

        # TODO: Compute ratio
        # print(f"{name}: {ratio:.2f}%")

        # TODO: Enqueue to handle actuator 2

    cap.release()
    q.put(('DONE', None))
    exit()


def imshow(title, frame, pos=None):
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)


def main():
    global FORCE_STOP

    # parser = ArgumentParser(prog='python3 factory.py',
    #                         description="Factory tool")

    # parser.add_argument("-d",
    #                     "--device",
    #                     default=None,
    #                     type=str,
    #                     help="Arduino port")
    # args = parser.parse_args()

    # TODO: HW2 Create a Queue

    que = queue.Queue()

    # TODO: HW2 Create thread_cam1 and thread_cam2 threads and start them.
    t1 = threading.Thread(target = thread_cam1, args=(que,))
    t2 = threading.Thread(target = thread_cam2, args=(que,))

    t1.start()
    t2.start()


    # with FactoryController(args.device) as ctrl:
    while not FORCE_STOP:
        if cv2.waitKey(10) & 0xff == ord('q'):
            break

        # TODO: HW2 get an item from the queue. You might need to properly handle exceptions.
        # de-queue name and data
        try:
            name, data = que.get_nowait()

        except Empty:
            continue
        

        # TODO: HW2 show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.
        if name[:6] == 'VIDEO:':
            imshow(name[6:], data)
        
        # TODO: Control actuator, name == 'PUSH'

        elif name == 'DONE':
            FORCE_STOP = True

        que.task_done()

    cv2.destroyAllWindows()
    t1.join()
    t2.join()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()
