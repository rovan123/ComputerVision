import cv2
import time
from Hand_Tracking import HandDetectionModule as hdm

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = hdm.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    print(lmList)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, "FPS: " + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    k = cv2.waitKey(1)

    if k == 27:
        break