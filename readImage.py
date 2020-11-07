import cv2
import os
import sys
POS_DIR = "./positive_Covid-19"

print(cv2.getBuildInformation())

for image in os.listdir(POS_DIR):
    path = os.path.join(POS_DIR, image)
    print(path)
    a = cv2.imread(path, 0)
    cv2.imshow('image', a)
    print(a)
    sys.exit(0)