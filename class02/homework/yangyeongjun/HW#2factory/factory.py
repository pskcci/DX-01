#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import cv2
import numpy as np
from iotdemo import FactoryController
from iotdemo.motion.motion_detector import MotionDetector

FORCE_STOP = False

def thread_cam1(q):
    # MotionDetector 인스턴스 생성
    motion_detector = MotionDetector()
    motion_detector.load_preset('/home/intel/git-training/DX-01/class02/smart-factory/resources/motion.cfg')
    
    # Open video clip 'resources/conveyor.mp4'
    cap = cv2.VideoCapture('./resources/conveyor.mp4')

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break
        
        # 모션 감지 및 크롭
        cropped_frame = motion_detector.detect(frame)  # detect 메서드로 모션 감지된 부분 크롭
        if cropped_frame is not None:
            # 크롭된 이미지 화면에 띄우기
            q.put(("VIDEO:Cam1 cropped", cropped_frame))
        
        # Enqueue "VIDEO:Cam1 live", frame info
        q.put(("VIDEO:Cam1 live", frame))

    cap.release()
    q.put(('DONE', None))
    exit()


def thread_cam2(q):
    # MotionDetector 인스턴스 생성
    motion_detector = MotionDetector()
    motion_detector.load_preset('/home/intel/git-training/DX-01/class02/smart-factory/resources/motion.cfg')

    # Open video clip 'resources/conveyor.mp4'
    cap = cv2.VideoCapture('./resources/conveyor.mp4')

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # 모션 감지 및 크롭
        cropped_frame = motion_detector.detect(frame)  # detect 메서드로 모션 감지된 부분 크롭
        if cropped_frame is not None:
            # 크롭된 이미지 화면에 띄우기
            q.put(("VIDEO:Cam2 cropped", cropped_frame))
        
        # Enqueue "VIDEO:Cam2 live", frame info
        q.put(("VIDEO:Cam2 live", frame))

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

    parser = ArgumentParser(prog='python3 factory.py',
                            description="Factory tool")

    parser.add_argument("-d",
                        "--device",
                        default=None,
                        type=str,
                        help="Arduino port")
    args = parser.parse_args()

    # Create a Queue
    q = Queue()

    # Create thread_cam1 and thread_cam2 threads and start them
    cam1_thread = threading.Thread(target=thread_cam1, args=(q,))
    cam2_thread = threading.Thread(target=thread_cam2, args=(q,))

    cam1_thread.start()
    cam2_thread.start()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            # Get an item from the queue. Handle Empty exception
            try:
                name, data = q.get(timeout=1)  # 큐에서 항목을 가져옴
            except Empty:
                continue

            # Show videos with titles of 'Cam1 live' and 'Cam2 live' respectively
            if name == "VIDEO:Cam1 live":
                imshow("Cam1 live", data, pos=(0, 0))
            elif name == "VIDEO:Cam2 live":
                imshow("Cam2 live", data, pos=(640, 0))

            # Show cropped frames of detected motion
            elif name == "VIDEO:Cam1 cropped":
                imshow("Cam1 Cropped", data, pos=(1280, 0))  # Show cropped region for Cam1
            elif name == "VIDEO:Cam2 cropped":
                imshow("Cam2 Cropped", data, pos=(1920, 0))  # Show cropped region for Cam2

            # Handle actuator control, name == 'PUSH'
            # TODO: Implement actuator control here

            if name == 'DONE':
                FORCE_STOP = True

            q.task_done()

    cam1_thread.join()
    cam2_thread.join()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit(0)  # 정상 종료 코드 전달
