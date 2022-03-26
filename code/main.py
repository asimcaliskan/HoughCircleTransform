import cv2
import sys
from hough_transform import CircleHoughTransform
from iot import calculate_iot

if __name__ == "__main__":
    image_file_name = "dataset/" + input("image filename(it must be in dataset path): ")
    ground_truth_file_name= "ground_truth/" + input("ground truth filename(it must be in ground_truth path): ")
    min_radius = input("min_radius: ")
    max_radius = input("max_radius: ")
    max_candidate_center = input("max_candidate_center: ")
    min_center_distance = input("min_center_distance: ")
    print("PLEASE WAIT...")
    image = cv2.imread(filename=image_file_name)
    circle_hough_transform = CircleHoughTransform(image, int(min_radius), int(max_radius), int(max_candidate_center), int(min_center_distance))
    detected_circles = circle_hough_transform.find_circles()
    calculate_iot(ground_truth_file_name, detected_circles)