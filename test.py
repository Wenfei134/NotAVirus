import cv2 
  
# path 
path = r'C:\Users\Rajnish\Desktop\geeksforgeeks.png'
  
# Using cv2.imread() method 
img = cv2.imread(path) 
  
# Displaying the image 
cv2.imshow('image', img) 