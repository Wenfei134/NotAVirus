import numpy as np
import cv2
import math
from scipy import ndimage
from scipy import ndimage
import matplotlib.pyplot as plt
import os

#Past issues (now fixed): positive_0_318.jpg - some skewing
#negative_0_992, negative_0_1867, negative_0_2476 is blank, negative_0_2553 very skewed, negative_0_3283 extreme skew, 
#negative_0_3721 some skew, negative_0_4428 a lot, negative_0_6384 wide skew, negative_0_7078 little, negative_0_7944 some, 

# Define size of input images
WIDTH_IN = 432
HEIGHT_IN = 288

#Input directories
POS_DIR_IN = "./positive_Covid-19"
NEG_DIR_IN = "./negative_Covid-19"

#Output directories
POS_DIR_OUT = "./proc_positive_Covid-19"
NEG_DIR_OUT = "./proc_negative_Covid-19"
  
def rotate_and_save_img(img, img_name, pts1, pts2):

    # # Cannot rotate
    # if (np.shape(pts1)[0] < 4 or len(np.unique(pts1)) != len(pts1)): #Checks if <4 pts, or any duplicate pts
    #     # Know we can't rotate.
    #     # don't output
    #     return

    rows,cols,ch = img.shape

    # Tranformation matrix
    M = cv2.getPerspectiveTransform(pts1,pts2)

    # Apply perspective transformation
    dst = cv2.warpPerspective(img, M, (WIDTH_IN, HEIGHT_IN))

    # Save image in output directory
    if (img_name[0:8] == "positive"): #infer output directory
        dir_out = POS_DIR_OUT
    elif (img_name[0:8] == "negative"):
        dir_out = NEG_DIR_OUT
    else:
        dir_out = os.curdir

    cv2.imwrite(os.path.join(dir_out, img_name), dst, [cv2.IMWRITE_JPEG_QUALITY, 100]) # Save the image
    return


def get_four_corners(img):
    final_corners = np.empty((0,2), np.float32) # 4 corners in extracted from list of points
    ref_corners = np.float32([[0,0],[WIDTH_IN,0],[0,HEIGHT_IN],[WIDTH_IN,HEIGHT_IN]]) # 4 corners in the new image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        
    # Find corners (may have more than 4)
    found_corners = cv2.goodFeaturesToTrack(gray, 40, 0.0001, 50) #was 27, 0.01, 10
    found_corners = np.int0(found_corners) 
        
    # Visual: Circle the corners
    # for i in found_corners: 
    #     x, y = i.ravel() 
    #     cv2.circle(img, (x, y), 3, 255, -1) 
        # print("pts: x:", x, "y:", y)
    # plt.imshow(img), plt.show() 

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
    # print(final_corners[:])

    # Note: The final_corners array must have have the following ordering: Top Left, Top Right, Bottom Left, Bottom Right
    # in order for rotate_img to work properly. 
    # We make the assumption that any skewing in images is less than 45 degrees. (Else, the absolute distance will not be accurate)
    return final_corners, ref_corners

def main():
    # Input directories
    dirs = [POS_DIR_IN, NEG_DIR_IN]

    for dir in dirs:
        # Rotate all images
        for img_name in os.listdir(dir):
            path = os.path.join(dir, img_name)
            img = cv2.imread(path)
            pts1, pts2 = get_four_corners(img)
            rotate_and_save_img(img, img_name, pts1, pts2)


if __name__ == "__main__":
  main()