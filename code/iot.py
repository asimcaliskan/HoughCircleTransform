from turtle import distance
from cv2 import sqrt
import numpy as np
from scipy.optimize import brentq

def intersection_area(d, R, r):
    if d <= abs(R-r):
        # One circle is entirely enclosed in the other.
        return np.pi * min(R, r)**2
    if d >= r + R:
        # The circles don't overlap at all.
        return 0

    r2, R2, d2 = r**2, R**2, d**2
    alpha = np.arccos((d2 + r2 - R2) / (2*d*r))
    beta = np.arccos((d2 + R2 - r2) / (2*d*R))
    return ( r2 * alpha + R2 * beta -
             0.5 * (r2 * np.sin(2*alpha) + R2 * np.sin(2*beta))
           )

def calculate_iot(file_name, detected_circles):
    ground_truth = []
    with open(file_name) as f:
        for line in f.readlines()[1:]:
            temp = []
            for val in line.strip().split(" "):
                temp.append(float(val))
            ground_truth.append(temp)
    
    len_detected_circles = len(detected_circles)
    len_ground_truth = len(ground_truth)

    one = sorted(detected_circles, key=lambda element: (element[0], element[1]))
    two = sorted(ground_truth, key=lambda element: (element[0], element[1]))

    temp = 0
    for i in range(min(len_ground_truth, len_detected_circles)):
        area_one = np.pi * one[i][2] ** 2 
        area_two = np.pi * two[i][2] ** 2

        dist = sqrt((one[i][0] - two[i][0])**2 + (one[i][1] - two[i][1])**2 )
        int_area = intersection_area(dist[0][0], one[i][2] ** 2, two[i][2] ** 2 )

        temp += int_area / (area_one + area_two - int_area)
    print("IOU:", abs(round(temp / len(one), 2)))