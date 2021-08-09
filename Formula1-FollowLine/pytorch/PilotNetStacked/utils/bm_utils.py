import numpy as np
import time
import cv2

def check_center(position_x):
    if (len(position_x[0]) > 1):
        x_middle = (position_x[0][0] + position_x[0][len(position_x[0]) - 1]) / 2
        not_found = False
    else:
        # The center of the line is in position 326
        x_middle = 326
        not_found = True
    return x_middle, not_found

def get_point(index, img):
    mid = 0
    if np.count_nonzero(img[index]) > 0:
        left = np.min(np.nonzero(img[index]))
        right = np.max(np.nonzero(img[index]))
        mid = np.abs(left - right)/2 + left
    return int(mid)


def calculate_deviation(image):
   
    image_cropped = image
    image_hsv = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,50,50])
    upper_red = np.array([180,255,255])
    image_mask = cv2.inRange(image_hsv, lower_red, upper_red)

    # show image in gui -> frame_0
    
    rows, cols = image_mask.shape
    rows = rows - 1     # para evitar desbordamiento

    alt = 0
    ff = cv2.reduce(image_mask, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    if np.count_nonzero(ff[:, 0]) > 0:
        alt = np.min(np.nonzero(ff[:, 0]))

    points = []
    for i in range(3):
        if i == 0:
            index = alt
        else:
            index = rows//(2*i)
        points.append((get_point(index, image_mask), index))

    points.append((get_point(rows, image_mask), rows))

    # We convert to show it
    # Shape gives us the number of rows and columns of an image
    size = image_mask.shape
    rows = size[0]
    columns = size[1]

    # We look for the position on the x axis of the pixels that have value 1 in different positions and
    position_x_down = np.where(image_mask[points[3][1], :])
    position_x_middle = np.where(image_mask [points[1][1], :])
    position_x_above = np.where(image_mask[points[2][1], :])        

    # We see that white pixels have been located and we look if the center is located
    # In this way we can know if the car has left the circuit
    x_middle_left_down, not_found_down = check_center(position_x_down)
    x_middle_left_middle, not_found_middle = check_center(position_x_middle)

    # We look if white pixels of the row above are located
    if (len(position_x_above[0]) > 1):
        x_middle_left_above = (position_x_above[0][0] + position_x_above[0][len(position_x_above[0]) - 1]) / 2
        # We look at the deviation from the central position. The center of the line is in position cols/2
        deviation = x_middle_left_above - (cols/2)
    else:
        deviation = cols

    return deviation
