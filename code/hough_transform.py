from cv2 import Sobel
import numpy
import cv2
from collections import defaultdict
import math

class CircleHoughTransform:
    def __init__(self, image: numpy.ndarray, min_radius, max_radius, max_candidate_center, min_center_distance) -> None:
        #USER PARAMETERS BEGIN
        self.max_candidate_center = max_candidate_center
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_center_distance = min_center_distance
        #USER PARAMETERS END

        self.image = image
        self.canny_image = self.get_canny_image()

        self.sobel_gradient_x :numpy.ndarray = None
        self.sobel_gradient_y : numpy.ndarray = None
        self.sobel_gradient_x, self.sobel_gradient_y = self.get_sobel_gradient()

    def get_canny_image(self, sigma=0.33)->numpy.ndarray:
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.medianBlur(gray_image, 5)
        median_value = numpy.median(blurred_image)
        lower = int(max(0, (1.0 - sigma) * median_value))
        upper = int(min(255, (1.0 + sigma) * median_value))
        return cv2.Canny(blurred_image, lower, upper)
    
    def get_sobel_gradient(self):
        ddepth = cv2.CV_16S
        scale = 1
        delta = 2
        blurred_image = cv2.GaussianBlur(self.image, (3, 3), 0)
        gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray_image, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(gray_image, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        return (grad_x, grad_y)
    
    def get_line_coordinates(self, gradient_x: float, gradient_y: float, x_0: int, y_0: int, image_width: int, image_height: int):
        line_coordinates = []

        #Calculate gradient
        if gradient_x == 0 and gradient_y != 0:
            m = numpy.Inf
        elif gradient_x != 0 and gradient_y == 0:
            m = 0
        else:
            m = gradient_y / gradient_x

        #infinity and zero slope don't take into account
        if m == numpy.Inf or m == -numpy.Inf or m == 0:
            pass
        else:
            """
            x = x_center + r.cos(theta)
            y = y_center + r.sin(theta)

            find gradient lines and returns it
            """
            theta = math.atan(m)
            for r in range(-self.max_radius, self.min_radius):
                x = r * math.cos(theta) + x_0
                y = r * math.sin(theta) + y_0
                if 0 <= x < image_width and 0 <= y < image_height:
                    line_coordinates.append((round(x), round(y), abs(r)))

            for r in range(self.min_radius, self.max_radius):
                x = r * math.cos(theta) + x_0
                y = r * math.sin(theta) + y_0
                if 0 <= x < image_width and 0 <= y < image_height:
                    line_coordinates.append((round(x), round(y), abs(r)))
        return line_coordinates

    def save_canny_and_original_images(self):
        cv2.imwrite("original_image9.jpg", self.image)
        cv2.imwrite("canny_image9.jpg", self.canny_image)

    """
    def show_candidate_circles(self)->None:
        edge_indices = numpy.argwhere(self.canny_image[:,:])
        image_height, image_width = self.canny_image.shape
        candidate_circle_accumulator = defaultdict(int)
       
        for edge_y, edge_x in edge_indices:
            gradient_x = self.sobel_gradient_x[edge_y, edge_x]
            gradient_y = self.sobel_gradient_y[edge_y, edge_x]
            line_coordinates = self.get_line_coordinates(gradient_x, gradient_y, edge_x, edge_y, image_width, image_height)
            for line_x, line_y in line_coordinates:
                candidate_circle_accumulator[(line_y, line_x)] += 1

        high_voted_centers = sorted(candidate_circle_accumulator.items(), key=lambda i: i[1], reverse=True)[0: self.max_candidate_center]
        
        for candidate_center, vote in high_voted_centers:
            candidate_center_y, candidate_center_x = candidate_center
            self.image = cv2.circle(self.image, (candidate_center_x, candidate_center_y), 1, (0,0,255), 1)

        cv2.imshow("Candidate Circles", self.image)
        cv2.waitKey(0)
    """

    """
    def show_gradient_lines(self):
        edge_indices = numpy.argwhere(self.canny_image[:,:])
        image_height, image_width = self.canny_image.shape
        candidate_circle_accumulator = defaultdict(int)

        for edge_y, edge_x in edge_indices:
            gradient_x = self.sobel_gradient_x[edge_y, edge_x]
            gradient_y = self.sobel_gradient_y[edge_y, edge_x]
            line_coordinates = self.get_line_coordinates(gradient_x, gradient_y, edge_x, edge_y, image_width, image_height)
            for line_x, line_y, r in line_coordinates:
                self.image = cv2.circle(self.image, (line_x, line_y), 1, (0,0,255), 1)
                candidate_circle_accumulator[(line_y, line_x)] += 1

        cv2.imshow("Gradient Line", self.image)
        cv2.waitKey(0)     
    """

    def draw_circles(self, circles: list):
        print("Number Of Detected Circles: ", len(circles))
        for y, x, r in circles:
            print( y, " ", x, " ", r)
            self.image = cv2.circle(self.image, (y, x), r, (255,0,255), 2)
                
        cv2.imwrite("output.jpg", self.image)
        cv2.waitKey(0)

    def find_circles(self)->None:
        edge_indices = numpy.argwhere(self.canny_image[:,:])

        image_height, image_width = self.canny_image.shape
        candidate_circle_accumulator = defaultdict(int)

        for edge_y, edge_x in edge_indices:
            gradient_x = self.sobel_gradient_x[edge_y, edge_x]
            gradient_y = self.sobel_gradient_y[edge_y, edge_x]
            
            line_coordinates = self.get_line_coordinates(gradient_x, gradient_y, edge_x, edge_y, image_width, image_height)
            
            for line_x, line_y, line_length in line_coordinates:
                candidate_circle_accumulator[(line_y, line_x, line_length)] += 1

        high_voted_centers = sorted(candidate_circle_accumulator.items(), key=lambda i: i[1], reverse=True)[0: self.max_candidate_center]

        #remove candidate center which are close to each other
        postprocess_circles = []
        detected_circles = []
        for candidate_center, vote in high_voted_centers:
            candidate_center_y, candidate_center_x, r = candidate_center
            if all(abs(candidate_center_x - xc) > self.min_center_distance or abs(candidate_center_y - yc) > self.min_center_distance or abs(r - rc) > self.min_center_distance for xc, yc, rc in postprocess_circles):
                postprocess_circles.append((candidate_center_x, candidate_center_y, r))
                detected_circles.append([candidate_center_y, candidate_center_x, r])

        self.draw_circles(postprocess_circles)
        return detected_circles