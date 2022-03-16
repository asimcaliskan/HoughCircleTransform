from turtle import left
from cv2 import Sobel
import numpy
import cv2
from collections import defaultdict

class CircleHoughTransform:
    def __init__(self, image: numpy.ndarray) -> None:
        self.image = image
        self.canny_image = self.get_canny_image()

        self.min_radius = 30
        self.max_radius = 200
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

            for y in range(min_y, max_y):
                x = round((y - b)/m)
                if 0 <= x < image_width and 0 <= y < image_height:
                    line_coordinates.append((x, y))

            for x in range(min_x, max_x):
                y = round(m*x + b)
                if 0 <= x < image_width and 0 <= y < image_height:
                    line_coordinates.append((x, y))
        return list(dict.fromkeys(line_coordinates))



    def find_circles(self)->None:
        edge_indices = numpy.argwhere(self.canny_image[:,:])
        image_height, image_width = self.canny_image.shape
        candidate_circle_accumulator = defaultdict(int)
        for edge_y, edge_x in edge_indices:
            gradient_x = self.sobel_gradient_x[edge_y, edge_x]
            gradient_y = self.sobel_gradient_y[edge_y, edge_x]
            line_coordinates = self.get_line_coordinates(gradient_x, gradient_y, edge_x, edge_y, image_width, image_height)
            for line_x, line_y in line_coordinates:
                candidate_circle_accumulator[(line_y, line_x)] += 1

        high_voted_centers = sorted(candidate_circle_accumulator.items(), key=lambda i: i[1], reverse=True)[0: len(edge_indices)]

        postprocess_centers = []
        counter = 0
        for candidate_center, vote in high_voted_centers:
            candidate_center_y, candidate_center_x = candidate_center
            if all(abs(candidate_center_x - xc) > self.min_center_distance or abs(candidate_center_y - yc) > self.min_center_distance  for yc, xc in postprocess_centers):
                postprocess_centers.append((candidate_center_y, candidate_center_x))
                counter += 1
            if counter == self.max_candidate_center:
                break
                """
        for y, x in postprocess_circles:
            self.image = cv2.circle(self.image, (x, y), 1, (255, 0, 0), 2)
        cv2.imshow("ad", self.image)
        """
        accumulator = defaultdict(int)
        for candidate_center_y, candidate_center_x in postprocess_centers:
            for radius, r_cos_theta, r_sin_theta in self.candidate_circles:
                x = candidate_center_x + r_cos_theta
                y = candidate_center_y + r_sin_theta
                if x < image_width and y < image_height and x >= 0 and y >= 0:
                    if self.canny_image[y, x] != 0:
                        accumulator[(candidate_center_x, candidate_center_y, radius)] += 1

        print("--a--")
        detected_circles = []
        for candidate_circle, votes in sorted(accumulator.items(), key=lambda i: i[1], reverse=True):
            x, y, r = candidate_circle
            if votes >= 100: 
                detected_circles.append((x, y, r, votes))
                print(x, y, r, votes)
        
        print(len(detected_circles))
        
        """
        pixel_threshold = 5
        postprocess_circles = []
        for x, y, r, v in detected_circles:
            if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(r - rc) > pixel_threshold for xc, yc, rc, v in postprocess_circles):
                postprocess_circles.append((x, y, r, v))
            detected_circles = postprocess_circles
        """
        
        print(len(detected_circles))
        for y, x, r, v in detected_circles:
            output_img = cv2.circle(self.image, (y, x), r, (0,0,255), 1)
            cv2.imshow("asdasd", output_img)
        
        cv2.waitKey(0)

            