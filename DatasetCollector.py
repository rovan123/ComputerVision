import cv2

def collect():
    path = '/Users/rovansuresh/PycharmProjects/ComputerVision/Face_Recognition/dataset'
    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    id1 = input("Enter you id: ")
    video = cv2.VideoCapture(0)
    count = 0;
    print("Collecting Samples and Storing in Database...")

    while True:
        success, img = video.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for x,y,h,w in faces:
            count = count + 1
            print(count)

            cv2.imwrite(path + "/User." + str(id1) + "."
                        + str(count) + ".jpg", gray[y:y+h, x:x+h])
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 2)

        cv2.imshow("Faces", img)
        cv2.waitKey(1)
        if count > 50:
            break

    video.release()
    cv2.destroyAllWindows()

    print("Complete")

collect()