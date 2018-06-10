import numpy as np 
import cv2
import pickle
# print(cv2.__file__)


face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eyes_cascade= cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

labels = {'person__name': 1}
with open('labels.pickle', 'rb') as f:
    labels = pickle.load(f)
    labels = {v:k for k,v in labels.items()}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        # img_item = 'my-image.png'
        # cv2.imwrite(img_item, roi_gray)
        
        #
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 :# and conf <= 85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

        color = (255, 0, 0)     #BGR 0-255
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

        eyes = eyes_cascade.detectMultiScale(roi_gray)
        for (x, y, w, h) in eyes:
            cv2.rectangle(roi_color, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()