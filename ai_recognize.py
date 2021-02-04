import cv2
import os
import face_recognition as fc

face_cascade = cv2.CascadeClassifier(
    "C:\\Users\\user\\PycharmProjects\\artificial_intelligence\\opencv\\haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(1)
image = fc.load_image_file("Mastermind.JPG")
face_encoding = fc.face_encodings(image)[0]

while True:
    check, frame = video.read()
    # cv2.imwrite("test.jpg", frame)
    grey_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(grey_img, scaleFactor=1.05, minNeighbors=5)
    for x, y, w, h in faces:
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # img = fc.load_image_file("C:\\Users\\user\\PycharmProjects\\video\\test.jpg")
        unknown_face_encoding1 = fc.face_encodings(frame)
        if len(unknown_face_encoding1) > 0:
            unknown_face_encoding = unknown_face_encoding1[0]
            results = fc.compare_faces([face_encoding], unknown_face_encoding)
            if results[0]:
                cv2.putText(img, 'Mastermind', (x, y - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (36, 255, 12), 2)
            else:
                cv2.putText(img, 'NOT Mastermind', (x, y - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow("VIDEO1", frame)
    # cv2.imshow("VIDEO2", frame)
    # cv2.imshow("VIDEO3", frame)
    # cv2.imshow("VIDEO4", frame)
    # cv2.imshow("VIDEO5", frame)
    # cv2.imshow("VIDEO6", frame)
    # os.remove("C:\\Users\\user\\PycharmProjects\\video\\test.jpg")
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
