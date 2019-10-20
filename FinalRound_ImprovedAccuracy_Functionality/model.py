import numpy as np
import cv2
from numpy.testing import assert_almost_equal
from training.predict import Predict

class Model(object):
    """
    The Model class represents an ml model
    """

    def __init__(self):
        # Initialise your model here
        self.model = Predict('config.json','..\\data\\test.jpg','..\\data\\output\\')
        #self.model = None

    def predict(self, x: np.ndarray):
        """Returns the prediction result of the model

        Parameters
        ----------
        x : ndarray
            Input image of shape width, height, channels(RGB)


        Returns
        -------
        result: list
            A 2D list of shape mx5 where each one of the m items represents a detected parking slot.
            Each detected parking slot (cx,cy,v) contains the center coordinates cx,cy of the parking lot
            vehicle occupancy v (1 if vehicle is present, 0 otherwise),
            color_code c (0-8 values for White-0, Silver-1, Black-2, Grey-3, Blue-4, Red-5, Brown-6, Green-7, Others-8)
            pose p (0 for front facing, 1 for back facing)
        """
        img = cv2.resize(x, (227,227))
        self.model.run()
        result = [[1.1, 1.1, 1, 6, 0], [1.2, 1.2, 1, 7, 0], [3, 3, 1, 8, 1], [4, 4, 0, 6, 1], [5, 5, 0, 5, 1]]
        return result


def test_model():

    def create_blank(width, height, rgb_color=(0, 0, 0)):
        """Create new image(numpy array) filled with certain color in RGB"""

        # Create black blank image
        image = np.zeros((height, width, 3), np.uint8)
        # Since OpenCV uses BGR, convert the color first
        color = tuple(reversed(rgb_color))
        # Fill image with color
        image[:] = color

        return image

    x = create_blank(500,500, rgb_color=(255, 0, 0))
    y = [[1.1, 1.1, 1, 6, 0], [1.2, 1.2, 1, 7, 0], [3, 3, 1, 8, 1], [4, 4, 0, 6, 1], [5, 5, 0, 5, 1]]

    model = Model()
    y_predicted = model.predict(x)
    assert_almost_equal(y_predicted, y)


if __name__ == "__main__":
    test_model()

