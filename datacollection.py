import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np 
import math
import time 

cap = cv2.VideoCapture(0) 
detector = HandDetector(maxHands = 1)

offset = 20
imgSize = 300

folder = "Data/L"
counter = 0 

while True:
    #code to read the image
    success, img = cap.read()
    hands,img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255 #white frame is defined to avoid fluctuations
        imgCrop = img[y-offset:y + h +offset , x-offset :x + w + offset] #cropped image to avoid the environemt

        

        aspectRatio = h / w

        if aspectRatio > 1: # if height is greater than width
            k =  imgSize / h
            wCal= math.ceil(k * w) #calculated width with respect to constant height(300)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize #cropped image is superimposed on imagewhite

        else:
            k =  imgSize / w  # if width is greater than height
            hCal= math.ceil(k * h) #calculated height with respect to constant widht(300)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("imageCrop", imgCrop) #command to show the cropped image
        cv2.imshow("imageWhite", imgWhite) #command to show the image white 


    cv2.imshow('Image',img) # command to show the full webcam image

    key = cv2.waitKey(1)
    if key == ord('a'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)