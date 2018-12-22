import cv2

def run():
    casc_path = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(casc_path)
    cv2.namedWindow("webcam")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30,30)
        )        

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("webcam", frame)
        key = cv2.waitKey(30)
        if key == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    run()