import cv2
import numpy as np

def run():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("webcam")
    lower = np.array([60, 135, 100], dtype = "uint8")
    upper = np.array([255, 170, 125], dtype = "uint8")
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
            
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(converted, lower, upper)
        idx = (mask == 0)
        frame[idx] = 0
        frame = cv2.medianBlur(frame, 3)
        cv2.imshow("webcam", frame)
        key = cv2.waitKey(30)
        if key == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    run()