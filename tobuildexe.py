import cv2
import os
import face_recognition as fc
import numpy as np
import pyttsx3
import itertools


def get_encoded_faces():
    encoded = {}

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fc.load_image_file("faces/" + f)
                encoding = fc.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded


class AiRecognize:
    video = cv2.VideoCapture(1)
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')

    def speak(self, audio):
        self.engine.setProperty('voice', self.voices[1].id)
        self.engine.say(audio)
        self.engine.runAndWait()
        self.engine.stop()

    old_names = []
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    def classify_face(self, img, face_locations):

        unknown_face_encodings = fc.face_encodings(img, face_locations)

        face_names = []
        for face_encoding in unknown_face_encodings:
            # See if the face is a match for the known face(s)
            matches = fc.compare_faces(self.faces_encoded, face_encoding)
            name = "Unknown"

            # use the known face with the smallest distance to the new face
            face_distances = fc.face_distance(self.faces_encoded, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            face_names.append(name)

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Draw a box around the face
                cv2.rectangle(img, (left - 20, top - 20), (right + 20, bottom + 20), (0, 255, 0), 2)

                # Draw a label with a name below the face
                cv2.rectangle(img, (left - 20, bottom - 15), (right + 20, bottom + 20), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(img, name, (left - 20, bottom + 15), font, 1.0, (0, 0, 0), 2)
        face_names.sort()
        self.old_names.sort()
        print(self.old_names, face_names)
        if len(face_names) != 0:
            if face_names != self.old_names:
                for i in face_names:
                    f = 0
                    for j in self.old_names:
                        if i == j:
                            f = 1
                            break
                    if f == 0:
                        self.speak("Hello " + i)
                self.old_names = face_names
        return img

    def __init__(self):
        while True:
            check, frame = self.video.read()
            face_locations = fc.face_locations(frame)
            if len(face_locations) != 0:
                cv2.imshow("VIDEO", self.classify_face(frame, face_locations))
            else:
                self.old_names = []
                cv2.imshow("VIDEO", frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break

        self.video.release()
        cv2.destroyAllWindows()


run = AiRecognize()
