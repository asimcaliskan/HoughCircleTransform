from cmath import cos
from re import T
from turtle import left
from cv2 import Sobel
import numpy
import cv2
from collections import defaultdict
import math

class CircleHoughTransform:
    def __init__(self, image: numpy.ndarray) -> None:
        #USER PARAMETERS BEGIN
        self.max_candidate_center = 200
        self.max_radius_range = 40
        #USER PARAMETERS END


        self.image = image
        self.canny_image = self.get_canny_image()

        self.min_radius = 20
        self.max_radius = 80

        self.min_center_distance = 20
        self.radiuses =  [i for i in range(self.min_radius, self.max_radius)]
        self.number_of_theta = 360
        self.theta_step = int(360 / self.number_of_theta)
        self.thetas = numpy.arange(0, 360, step=self.theta_step)
        self.sin_theta =  numpy.sin(numpy.deg2rad(self.thetas))
        self.cos_theta =  numpy.cos(numpy.deg2rad(self.thetas))
        self.candidate_circles = []
        self.create_candidate_circles()

        self.sobel_gradient_x :numpy.ndarray = None
        self.sobel_gradient_y : numpy.ndarray = None
        self.sobel_gradient_x, self.sobel_gradient_y = self.get_sobel_gradient()
        

    def create_candidate_circles(self)->list:
        for radius in self.radiuses:
            for theta in self.thetas:
                self.candidate_circles.append((radius, int(radius * self.cos_theta[theta]), int(radius *self.sin_theta[theta])))

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

        if gradient_x == 0 and gradient_y != 0:
            m = numpy.Inf
        elif gradient_x != 0 and gradient_y == 0:
            m = 0
        else:
            m = gradient_y / gradient_x


        if m == numpy.Inf or m == -numpy.Inf or m == 0:
            pass
        else:
            theta = math.atan(m)
            for r in range(-self.max_radius_range, self.max_radius_range):
                x = r * math.cos(theta) + x_0
                y = r * math.sin(theta) + y_0
                if 0 <= x < image_width and 0 <= y < image_height:
                    line_coordinates.append((round(x), round(y), abs(r)))
        return line_coordinates

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

    def show_gradient_lines(self):
        edge_indices = numpy.argwhere(self.canny_image[:,:])
        image_height, image_width = self.canny_image.shape
        candidate_circle_accumulator = defaultdict(int)

        for edge_y, edge_x in edge_indices:
            gradient_x = self.sobel_gradient_x[edge_y, edge_x]
            gradient_y = self.sobel_gradient_y[edge_y, edge_x]
            line_coordinates = self.get_line_coordinates(gradient_x, gradient_y, edge_x, edge_y, image_width, image_height)
            for line_x, line_y in line_coordinates:
                self.image = cv2.circle(self.image, (line_x, line_y), 1, (0,0,255), 1)
                candidate_circle_accumulator[(line_y, line_x)] += 1

        cv2.imshow("Gradient Line", self.image)
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
        
        """for candidate_center, vote in high_voted_centers:
            candidate_center_y, candidate_center_x, r = candidate_center
            self.image = cv2.circle(self.image, (candidate_center_x, candidate_center_y), r, (0,0,255), 1)"""

        pixel_threshold = 5
        postprocess_circles = []
        for candidate_center, vote in high_voted_centers:
            candidate_center_y, candidate_center_x, r = candidate_center
            if all(abs(candidate_center_x - xc) > pixel_threshold or abs(candidate_center_y - yc) > pixel_threshold or abs(r - rc) > pixel_threshold for xc, yc, rc in postprocess_circles):
                postprocess_circles.append((candidate_center_x, candidate_center_y, r))
        

        for y, x, r in postprocess_circles:
            self.image = cv2.circle(self.image, (y, x), r, (0,0,255), 1)
        
        cv2.imshow("Output", self.image)
        cv2.waitKey(0)    