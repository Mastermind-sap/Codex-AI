import cv2

cascade1 = cv2.CascadeClassifier(
    "C:\\Users\\user\\PycharmProjects\\video\\haarcascades\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml")
Video = cv2.VideoCapture(1)
while True:
    check, frame = Video.read()
    c1 = cascade1.detectMultiScale(frame, 1.1, 5)
    print (c1)
    for x, y, h, w in c1:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow("Image",frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break