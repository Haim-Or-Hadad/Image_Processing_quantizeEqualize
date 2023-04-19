
"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from ex1_utils import LOAD_GRAY_SCALE
import cv2 as cv
import numpy as np


# Constants for gamma correction GUI
MAX_GAMMA = 10.0
MAX_TRACKBAR_VALUE = 200
WINDOW_TITLE = 'Gamma Correction'
TRACKBAR_NAME = 'Gamma'

# Define the trackbar function
def on_trackbar(value, img):
    # Calculate gamma based on trackbar value
    gamma = MAX_GAMMA * (value / MAX_TRACKBAR_VALUE)
    
    # Apply gamma correction to the image and display it
    # Convert image to float to allow for gamma correction
    corrected_img = ((img / 255) ** (gamma) * 255).astype(np.uint8)
    cv.imshow(WINDOW_TITLE, corrected_img)

# Define the main function for displaying the gamma correction GUI
def gammaDisplay(image_path: str, representation: int):
    """
    GUI for gamma correction
    :param image_path: Path to the image
    :param representation: grayscale(1) or RGB(2)
    :return: None
    """
    # Load the image
    img = cv.imread(image_path)
    
    # Convert to grayscale if specified
    if representation == LOAD_GRAY_SCALE:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Create the window and trackbar
    cv.namedWindow(WINDOW_TITLE)
    cv.createTrackbar(TRACKBAR_NAME, WINDOW_TITLE, 0, MAX_TRACKBAR_VALUE, 
                      lambda x: on_trackbar(x, img))
    on_trackbar(0, img)

    # Wait for user to close the window
    cv.waitKey()
    cv.destroyAllWindows()

def main():
    gammaDisplay('water_bear.png', 2)

if __name__ == '__main__':
    main()