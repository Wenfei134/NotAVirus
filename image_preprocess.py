import numpy as np
import cv2
import math
from scipy import ndimage
from scipy import ndimage
import matplotlib.pyplot as plt
import os

WIDTH_IN = 432
HEIGHT_IN = 288
  
def rotate_img(img, pts1, pts2):
  rows,cols,ch = img.shape

  # Tranformation matrix
  M = cv2.getPerspectiveTransform(pts1,pts2)

  dst = cv2.warpPerspective(img,M,(432,288))

  plt.subplot(121),plt.imshow(img_before),plt.title('Input')
  plt.subplot(122),plt.imshow(dst),plt.title('Output')
  plt.show()

  print("Original:")
  cv2_imshow(img)
  print("Fixed:")
  cv2_imshow(dst)
  return


def get_four_corners(img):
  # final_corners = np.float32([[37,63],[369,45],[50,243],[383,225]]) # 4 corners in orginal image
  final_corners = np.empty((0,2), np.float32)
  ref_corners = np.float32([[0,0],[WIDTH_IN,0],[0,HEIGHT_IN],[WIDTH_IN,HEIGHT_IN]]) # 4 corners in the new image
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
  # Find corners (may have more than 4)
  found_corners = cv2.goodFeaturesToTrack(gray, 27, 0.01, 10) 
  found_corners = np.int0(found_corners) 
    
  # we iterate through each corner,  
  # making a circle at each point that we think is a corner. 
  for i in found_corners: 
    x, y = i.ravel() 
    cv2.circle(img, (x, y), 3, 255, -1) 
    print("pts: x:", x, "y:", y)
    
  plt.imshow(img), plt.show() 

  # Filter the corners to find those whose coordinates are closest to the actual edges of the frame

  # For each ref_corner P(x_i, y_i) in ref_corners: (These are (0,0), (432,0), (0,288), (432,288))
  for p_ref in ref_corners: # ordering: Top Left, Top Right, Bottom Left, Bottom Right
    x_i, y_i = p_ref.ravel() 

    min_dist = float("inf") # Reset
    min_pt = np.empty((1, 2), np.float32)

    # Compare the absolute distance to every found_corner P(x_f, y_f) in found_corners
    # Record the found_corner with the minimum distance to the ref_corner. Append to the final_corners array.
    for p_f in found_corners:
      x_f, y_f = p_f.ravel()
      abs_dist = math.sqrt((x_f - x_i) ** 2 + (y_f - y_i) ** 2)
      if (abs_dist < min_dist):
        min_dist = abs_dist # Set new closest point
        min_pt[0, 0] = x_f.astype(float)
        min_pt[0, 1] = y_f.astype(float)
    # print("test: ", min_pt)
    final_corners = np.vstack((final_corners, min_pt)) # Found closest point for current corner

  print(np.shape(final_corners))
  # Note: The final_corners array must have have the following ordering: Top Left, Top Right, Bottom Left, Bottom Right
  # in order for rotate_img to work properly. 
  # We make the assumption that any skewing in images is less than 45 degrees. (Else, the absolute distance will not be accurate)
  print(final_corners[:])
 
  return final_corners, ref_corners

def main():
  dir = "/content/drive/My Drive/newhacks2020/"
  
  for img_name in os.listdir(dir):
    img = cv2.imread(dir + img_name)
    pts1, pts2 = get_four_corners(img)
    rotate_img(img, pts1, pts2)


if __name__ == "__main__":
  main()