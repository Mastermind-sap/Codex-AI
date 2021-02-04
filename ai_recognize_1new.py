import cv2
import os
import face_recognition as fc
import numpy as np

face_cascade = cv2.CascadeClassifier(
    "C:\\Users\\user\\PycharmProjects\\artificial_intelligence\\opencv\\haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(1)


def get_encoded_faces():
    encoded = {}

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fc.load_image_file("faces/" + f)
                encoding = fc.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded


def classify_face(im):
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = im

    face_locations = fc.face_locations(img)
    unknown_face_encodings = fc.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = fc.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = fc.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(img, (left - 20, top - 20), (right + 20, bottom + 20), (0, 255, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img, (left - 20, bottom - 15), (right + 20, bottom + 20), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left - 20, bottom + 15), font, 1.0, (0, 0, 0), 2)

    return img


while True:
    check, frame = video.read()

    cv2.imshow("VIDEO", classify_face(frame))
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
