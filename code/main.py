from asyncio.windows_events import NULL
from re import A
import cv2
from cv2 import accumulate
import numpy
import sys
from hough_transform import CircleHoughTransform

accumulator_arr = NULL

#loads an jpg image and returns it
def read_jpg(file_name: str) -> numpy.ndarray:
    return cv2.imread(file_name)

def jpg_to_grayscale(image: numpy.ndarray) -> numpy.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def show_ndarray_image(image: numpy.ndarray)->None:
    cv2.imshow("JPG IMAGE", image)
    cv2.waitKey(0)

def get_first_and_last_image_number()->tuple:
    arguments = sys.argv
    if len(arguments) != 3:
        raise Exception("Wrong arguments!\nArgument format must be main.py first_image_number last_image_number")

    first_image_number = int(arguments[1])
    last_image_number = int(arguments[2])

    if first_image_number > last_image_number:
        raise Exception("Wrong arguments!\nfirst_image_number must be smaller or equal than last_image_number")

    return first_image_number, last_image_number

def detect_edge(image: numpy.ndarray)->numpy.ndarray:
    return cv2.Canny(image, 100, 200)

def try_pixel_val_on_accumulator_arr(pixel_row: int, pixel_column: int):
    global accumulator_arr
    height, width, radius = accumulator_arr.shape
    for row in range(height):
        for column in range(width):
            for depth in range(radius):
                equetion_result = (row - pixel_row)**2 + (column - pixel_column)**2 - depth ** 2
                if equetion_result == 0:
                    accumulator_arr[row, column, depth] += 1

def hough_transform(image: numpy.ndarray)->None:
    """
    r^2 = (x - x_0)^2 + (y - y_0)^2
    accumalator matrix shape is (image_height, image_width, radius)
    I chose the radius value as maximum value of image_height image_width to reduce accumulator array

    """
    global accumulator_arr
    image_height, image_width = image.shape
    accumulator_arr = numpy.zeros((image_height, image_width, max(image_width, image_width) // 2))

    for row in range(image_height):
        for column in range(image_width):
            pixel_val = image[row, column]
            if pixel_val != 0:#it is an edge pixel
                try_pixel_val_on_accumulator_arr(pixel_row=row, pixel_column=column)

    print("hough_transform result ", accumulator_arr.argmax)

if __name__ == "__main__":
    first_image_number, last_image_number = get_first_and_last_image_number()
    image_counter = first_image_number
    while image_counter <= last_image_number:
        file_name = ".\\dataset\\" + str(image_counter) + ".jpg"
        image = read_jpg(file_name)
        gray_image = jpg_to_grayscale(image)
        circle_hough_transform = CircleHoughTransform(image, gray_image, 1, 1, 10)
        circle_hough_transform.find_circles()
        image_counter += 1