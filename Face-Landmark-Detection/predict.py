import cv2
import dlib
import numpy as np
from imutils import face_utils

def run():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("webcam")
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        dets = detector(gray, 0)
        for (i, det) in enumerate(dets):
            shape = predictor(gray, det)
            shape = face_utils.shape_to_np(shape)

            for (x, y) in shape:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        cv2.imshow("webcam", frame)
        key = cv2.waitKey(30)
        if key == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    run()