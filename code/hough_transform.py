import numpy
import cv2
from collections import defaultdict

class CircleHoughTransform:
    def __init__(self,original_image: numpy.ndarray, gray_image: numpy.ndarray, min_canny_threshold: int, min_accumulator_threshold: int, min_center_distance: int) -> None:
        self.original_image = original_image
        self.gray_image = gray_image
        self.image_height = gray_image.shape[0]
        self.image_width = gray_image.shape[1]
        self.max_radius = 73
        self.min_radius = 70
        self.number_of_theta = 360
        self.theta_step = int(360 / self.number_of_theta)
        self.thetas = numpy.arange(0, 360, step=self.theta_step)
        self.radiuses =  [i for i in range(self.min_radius, self.max_radius)]
        self.accumulator_matrix = self.create_accumulator_matrix()
        self.canny_image = self.create_canny_image()
        self.min_vote_threshold = 130
        self.min_canny_threshold = min_canny_threshold
        self.min_center_distance = min_center_distance
        self.sin_theta =  numpy.sin(numpy.deg2rad(self.thetas))
        self.cos_theta =  numpy.cos(numpy.deg2rad(self.thetas))
        self.candidate_circles = []
        self.create_candidate_circles()

    def create_candidate_circles(self)->list:
        for radius in self.radiuses:
            for theta in self.thetas:
                self.candidate_circles.append((radius, int(radius * self.cos_theta[theta]), int(radius *self.sin_theta[theta])))

    def create_accumulator_matrix(self)->numpy.ndarray:
        return numpy.zeros((self.image_height, self.image_width, self.max_radius - self.min_radius), int)

    def create_canny_image(self, sigma=0.33)->numpy.ndarray:
        median_value = numpy.median(self.gray_image)
        lower = int(max(0, (1.0 - sigma) * median_value))
        upper = int(min(255, (1.0 + sigma) * median_value))
        return cv2.Canny(self.gray_image, lower, upper)

    def show_image(self):
        cv2.imshow("Canny Image", self.canny_image)
        cv2.waitKey(0)


    def find_circles(self) -> None:
        print(self.radiuses)
        accumulator = defaultdict(int)
        for image_y in range(self.image_height):
            for image_x in range(self.image_width):
                if self.canny_image[image_y, image_x] != 0:
                    for radius, r_cos_theta, r_sin_theta in self.candidate_circles:
                        x = image_x - r_cos_theta
                        y = image_y - r_sin_theta
                        accumulator[(x, y, radius)] += 1
  
        detected_circles = []
        for candidate_circle, votes in sorted(accumulator.items(), key=lambda i: i[1], reverse=True):
            x, y, r = candidate_circle
            if votes >= self.min_vote_threshold: 
                detected_circles.append((x, y, r, votes))
                print(x, y, r, votes)

        print(len(detected_circles))
        pixel_threshold = 5
        postprocess_circles = []
        for x, y, r, v in detected_circles:
            if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(r - rc) > pixel_threshold for xc, yc, rc, v in postprocess_circles):
                postprocess_circles.append((x, y, r, v))
            detected_circles = postprocess_circles

        print(len(detected_circles))
        for y, x, r, v in detected_circles:
            output_img = cv2.circle(self.original_image.copy(), (y, x), r, (255,0,0), 1)
            cv2.imshow("asdasd", output_img)
            cv2.waitKey(0)

        """
        edges_indices = numpy.argwhere(self.canny_image[:,:])   
        threshold = 200
        for radius in self.radius_arr:
            acc_cells = numpy.full((self.image_height, self.image_width), fill_value=0, dtype=int)
            for edge in edges_indices:
                edge_y = edge[0]
                edge_x = edge[1]
                if self.canny_image[edge_y, edge_x] == 255:
                    for angle in range(0, 360): 
                        b = edge_x - round(radius * self.sin_angles[angle]) 
                        a = edge_y - round(radius * self.cos_angles[angle]) 
                        if a >= 0 and a < self.image_height and b >= 0 and b < self.image_width: 
                            acc_cells[a, b] += 1
            print('For radius: ',radius)
            acc_cell_max = numpy.amax(acc_cells)
            print('max acc value: ',acc_cell_max)
            
            if(acc_cell_max > threshold):  

                print("Detecting the circles for radius: ",radius)       
                
                # Initial threshold
                acc_cells[acc_cells < threshold] = 0  
                
                # find the circles for this radius 
                for i in range(self.image_height): 
                    for j in range(self.image_width): 
                        if(i > 0 and j > 0 and i < self.image_height-1 and j < self.image_width-1 and acc_cells[i][j] >= threshold):
                            avg_sum = numpy.float32((acc_cells[i][j]+acc_cells[i-1][j]+acc_cells[i+1][j]+acc_cells[i][j-1]+acc_cells[i][j+1]+acc_cells[i-1][j-1]+acc_cells[i-1][j+1]+acc_cells[i+1][j-1]+acc_cells[i+1][j+1])/9) 
                            print("Intermediate avg_sum: ",avg_sum)
                            if(avg_sum >= 33):
                                print("For radius: ",radius,"average: ",avg_sum,"\n")
                                im = cv2.circle(self.canny_image, (i, j), radius, (255, 0, 0), 2)
                                cv2.imshow("ads", im)
                                cv2.waitKey(0)
                                acc_cells[i:i+5,j:j+7] = 0
        """
        """
        for radius in self.radius_arr:
            acc_cells = numpy.full((self.image_height, self.image_width), fill_value=0, dtype=int)
            for image_row in range(self.image_height):
                for image_column in range(self.image_width):
                    if self.canny_image[image_row, image_column] == 255:
                        for angle in range(0, 360): 
                            b = image_column - round(radius * self.sin_angles[angle]) 
                            a = image_row - round(radius * self.cos_angles[angle]) 
                            if a >= 0 and a < self.image_height and b >= 0 and b < self.image_width: 
                                acc_cells[a, b] += 1
            print('For radius: ',radius)
            acc_cell_max = numpy.amax(acc_cells)
            print('max acc value: ',acc_cell_max)
            
            if(acc_cell_max > 150):  

                print("Detecting the circles for radius: ",radius)       
                
                # Initial threshold
                acc_cells[acc_cells < 150] = 0  
                
                # find the circles for this radius 
                for i in range(self.image_height): 
                    for j in range(self.image_width): 
                        if(i > 0 and j > 0 and i < self.image_height-1 and j < self.image_width-1 and acc_cells[i][j] >= 150):
                            avg_sum = numpy.float32((acc_cells[i][j]+acc_cells[i-1][j]+acc_cells[i+1][j]+acc_cells[i][j-1]+acc_cells[i][j+1]+acc_cells[i-1][j-1]+acc_cells[i-1][j+1]+acc_cells[i+1][j-1]+acc_cells[i+1][j+1])/9) 
                            print("Intermediate avg_sum: ",avg_sum)
                            if(avg_sum >= 33):
                                print("For radius: ",radius,"average: ",avg_sum,"\n")
                                im = cv2.circle(self.canny_image, (i, j), radius, (255, 0, 0), 2)
                                cv2.imshow("ads", im)
                                cv2.waitKey(0)
                                acc_cells[i:i+5,j:j+7] = 0
        """

        """
        for image_row in range(self.image_height):
            for image_column in range(self.image_width):
                if self.canny_image[image_row, image_column] != 0:
                    for accumulator_r in range(self.min_radius, self.max_radius):
                        for accumulator_y in range(self.image_height):
                            for accumulator_x in range(self.image_width):
                                equetion_result = (image_row - accumulator_y)**2 + (image_column - accumulator_x)**2 - accumulator_r ** 2
                                if(round(equetion_result) == 0):
                                    self.accumulator_matrix[accumulator_y, accumulator_x, accumulator_r - self.min_radius] += 1
        print("---asd---")"""