from re import T
from turtle import left
from cv2 import Sobel
import numpy
import cv2
from collections import defaultdict

class CircleHoughTransform:
    def __init__(self, image: numpy.ndarray) -> None:
        self.image = image
        self.canny_image = self.get_canny_image()

        self.min_radius = 20
        self.max_radius = 40
        self.radius_range = self.max_radius - self.min_radius
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
        self.max_candidate_center = 50
        

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
            b = y_0 - x_0 * m
            min_y = y_0 - self.radius_range if y_0 - self.radius_range >= 0 else 0
            max_y = y_0 + self.radius_range if y_0 + self.radius_range < image_height else image_height

            min_x = x_0 - self.radius_range if x_0 - self.radius_range >= 0 else 0
            max_x = x_0 + self.radius_range if x_0 + self.radius_range < image_width else image_width

            for i in range(-self.radius_range, self.radius_range):
                x = round(((y_0 + i) - b)/m)
                if 0 <= x < image_width and 0 <= y < image_height:
                    line_coordinates.append((x, y))
                
                y = round(m*x + b)
                if 0 <= x < image_width and 0 <= y < image_height:
                    line_coordinates.append((x, round(y)))
            """   
            for y in range(min_y, max_y):
                x = round((y - b)/m)
                if 0 <= x < image_width and 0 <= y < image_height:
                    line_coordinates.append((x, y))
         
            for x in range(min_x, max_x):
                y = round(m*x + b)
                if 0 <= x < image_width and 0 <= y < image_height:
                    line_coordinates.append((x, round(y)))
            """
        return sorted(list(dict.fromkeys(line_coordinates)), key=lambda i:i[0], reverse=True)



    def find_circles(self)->None:
        edge_indices = numpy.argwhere(self.canny_image[:,:])
        image_height, image_width = self.canny_image.shape
        candidate_circle_accumulator = defaultdict(int)
        for edge_y, edge_x in edge_indices:
            gradient_x = self.sobel_gradient_x[edge_y, edge_x]
            gradient_y = self.sobel_gradient_y[edge_y, edge_x]
            line_coordinates = self.get_line_coordinates(gradient_x, gradient_y, edge_x, edge_y, image_width, image_height)
            self.image = cv2.line(self.image, line_coordinates[0], line_coordinates[-1], (0,255,0), 1)
            for line_x, line_y in line_coordinates:
                self.image = cv2.circle(self.image, (line_x, line_y), 1, (0,0,255), 1)
                candidate_circle_accumulator[(line_y, line_x)] += 1
            break

        cv2.imshow("asdasd", self.image)
        cv2.waitKey(0)
"""         high_voted_centers = sorted(candidate_circle_accumulator.items(), key=lambda i: i[1], reverse=True)[0: 500]
        
        for candidate_center, vote in high_voted_centers:
            candidate_center_y, candidate_center_x = candidate_center
            self.image = cv2.circle(self.image, (candidate_center_x, candidate_center_y), 1, (0,0,255), 1)
        cv2.imshow("asdasd", self.image)
        cv2.imshow("asd", self.canny_image)
        cv2.waitKey(0)
 """
   