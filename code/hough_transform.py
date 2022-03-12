import numpy
import cv2

class CircleHoughTransform:
    def __init__(self, gray_image: numpy.ndarray, min_canny_threshold: int, min_accumulator_threshold: int, min_center_distance: int) -> None:
        self.gray_image = gray_image
        self.image_height = gray_image.shape[0]
        self.image_width = gray_image.shape[1]
        self.max_radius = max(self.image_height, self.image_width) // 2
        self.min_radius = 3
        self.accumulator_matrix = self.create_accumulator_matrix()
        self.canny_image = self.create_canny_image()
        self.min_accumulator_threshold = min_accumulator_threshold
        self.min_canny_threshold = min_canny_threshold
        self.min_center_distance = min_center_distance

    def create_accumulator_matrix(self)->numpy.ndarray:
        return numpy.zeros((self.image_height, self.image_width, self.max_radius - self.min_radius), int)

    def create_canny_image(self, sigma=0.33)->numpy.ndarray:
        median_value = numpy.median(self.gray_image)
        lower = int(max(0, (1.0 - sigma) * median_value))
        upper = int(min(255, (1.0 + sigma) * median_value))
        return cv2.Canny(self.gray_image, lower, upper)

    def find_circles(self) -> None:
        for image_row in range(self.image_height):
            for image_column in range(self.image_width):
                if self.canny_image[image_row, image_column] != 0:
                    for accumulator_y in range(self.image_height):
                        for accumulator_x in range(self.image_width):
                            for accumulator_r in range(self.min_radius, self.max_radius):
                                equetion_result = (image_row - accumulator_y)**2 + (image_column - accumulator_x)**2 - accumulator_r ** 2
                                if(round(equetion_result) == 0):
                                    print(accumulator_r)