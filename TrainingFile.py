import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()

""""
Algorithm
1.Eignfaces(1991)
2.LBPH(1996) (Local Binary Pattern Histogram)
3.FisherFace(1997)
4.SIFT(1999)
5.SURF(2006)

"""


path = '/Users/rovansuresh/PycharmProjects/ComputerVision/Face_Recognition/dataset'
TrainedPath = '/Users/rovansuresh/PycharmProjects/ComputerVision/Face_Recognition/recognizer/Training.yml'

def getImageWithId(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    print(imagePaths)
    faces = []
    IDS = []

    for imagePath in imagePaths:
        faceImage = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImage, 'uint8')
        #print(faceNp)
        Id = (os.path.split(imagePath)[-1].split('.')[1])
        #dataset\\User.1.1.jpg
        Id = int(Id)
        print(Id)
        faces.append(faceNp)
        IDS.append(Id)
        cv2.imshow("Training", faceNp)
        cv2.waitKey(1)

    return IDS, faces

IDs, faces = getImageWithId(path)
recognizer.train(faces, np.array(IDs))
recognizer.write(TrainedPath)
print("Training Complete")
cv2.destroyAllWindows()


