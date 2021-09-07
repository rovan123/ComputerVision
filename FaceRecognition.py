import cv2
import numpy as np
import time
#from Face_Recognition.recognizer.DatasetCollector import collect

pTime = 0
cTime = 0

facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Training.yml")


font = cv2.FONT_HERSHEY_PLAIN

Id = 0
names = ['None', 'Rovan', 'Suresh', 'Latha', 'Sajan', 'Messi']


while True:
    success, img = video.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for x,y,w,h in faces:
        l=30
        t=7
        x1, y1 = x+w, y+h

        #cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 255), 2)
        #top left
        cv2.line(img, (x,y), (x+l, y), (255, 0, 255), t)
        cv2.line(img, (x,y), (x, y+l), (255, 0, 255), t)
        #top right
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        #bottom left
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        #bottom right
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)


        Id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        """
        if Id == 1:
            Id = 'Rovan'
        elif Id == 2:
            Id = 'Suresh'
        elif Id == 3:
            Id = 'Latha'
        elif Id == 4:
            Id = 'Sajan'
"""


        if (confidence < 100):
            Id = names[Id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            Id = "Unrecognized"
            confidence = "  {0}%".format(round(100 - confidence))
            #ans = input("Do you want to train this new image into dataset?")
            #if ans == 'yes':
                #collect()


        cv2.putText(img, str(Id), (x,y-10), font, 3, (255,0,0), 3)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, "FPS: " + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow('Facial Recognition', img)
    k = cv2.waitKey(1)

    if k == 27:
        break