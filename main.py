import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier  # classification module
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

folder = "Data/X"
counter = 0

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
          "V", "W", "X", "Y", "Z"]

while True:
    try:
        # code to read the image
        success, img = cap.read()
        if not success:
            continue

        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # white frame to avoid fluctuations
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]  # cropped image to avoid the environment

            aspectRatio = h / w

            if aspectRatio > 1:  # if height is greater than width
                k = imgSize / h
                wCal = math.ceil(k * w)  # calculated width with respect to constant height(300)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize  # cropped image is superimposed on imgWhite
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)

            else:
                k = imgSize / w  # if width is greater than height
                hCal = math.ceil(k * h)  # calculated height with respect to constant width (300)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50), (255, 0, 255),
                          cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

            cv2.imshow("imageCrop", imgCrop)  # command to show the cropped image
            cv2.imshow("imageWhite", imgWhite)  # command to show the image white

        cv2.imshow('Image', imgOutput)  # command to show the full webcam image

        if cv2.waitKey(1) & 0xFF == ord('q'):  # exit the loop when 'q' is pressed
            break

    except Exception as e:
        print(f"An error occurred: {e}")
        continue

cap.release()  # release the webcam
cv2.destroyAllWindows()  # close all OpenCV windows
