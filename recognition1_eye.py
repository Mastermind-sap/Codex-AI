import cv2
import pyautogui
import datetime
import time

eyecascade = cv2.CascadeClassifier(
    "C:\\Users\\user\\PycharmProjects\\video\\haarcascades\\haarcascades\\haarcascade_eye.xml")
Video = cv2.VideoCapture(1)
time_min = -1
time_sec = -1

while True:
    hierarchy, frame = Video.read()
    eyes = eyecascade.detectMultiScale(frame, 1.1, 2)
    # print(eyes)
    # print(len(eyes))
    if len(eyes) == 2:
        time1 = time.localtime()
        # print(time.localtime())
        # print(type(time1))
        # print(type(time1[4]))
        if time_min == -1 and time_sec == -1:
            time_min = time1[4]
            time_sec = time1[5]
        else:
            if time1[4] == time_min:
                diff = time1[5] - time_sec
                print(diff)
                time_sec = time1[5]
                # if diff < 5:
                #     print("click")
            else:
                time_min=time1[4]
        for x, y, h, w in eyes:
            cv2.circle(frame, ((x + w // 2), (y + h // 2)), 20, (0, 0, 255), cv2.FILLED)

    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
