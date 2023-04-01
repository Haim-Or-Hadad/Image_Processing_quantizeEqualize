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
from typing import List
import cv2
import numpy as np
import matplotlib.pyplot as plt

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 123456789


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    # read the image using cv2 of openCV
    img= cv2.imread(filename)

    # check if representation is 1 to GRAY_SCALE and convert it
    if representation == LOAD_GRAY_SCALE:
        img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    else : 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #converts the pixel values of the input image img to float32 data type and normalize the pixel intensities
    img = img.astype(np.float32) / 255 
    
    return img 


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    # read and convert image using imReadAndConvert function
    img = imReadAndConvert(filename, representation)
    # display image using matplotlib
    plt.imshow(img)
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """

    tranfer_matrix = np.array([[0.299, 0.587, 0.114],
                               [0.596, -0.275, -0.321],
                               [0.212, -0.523, 0.311]])
    # save original shape of image 
    img_shape = imgRGB.shape
    # matrix multiplication 
    return (imgRGB.reshape(-1,3) @ tranfer_matrix.transpose()).reshape(img_shape)

def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    tranfer_matrix = np.array([[0.299, 0.587, 0.114],
                            [0.596, -0.275, -0.321],
                            [0.212, -0.523, 0.311]])
    # Calculating an inverse matrix
    inverse_matrix = np.linalg.inv(tranfer_matrix)
    # save original shape of image 
    img_shape = imgYIQ.shape
    # matrix multiplication 
    return(inverse_matrix @ imgYIQ.transpose()).reshape(img_shape) 
    
def rgb_or_grayscale(img: np,ndarray):
    if len(img.shape) is 2:
        return 'gray'
    elif len(img.shape) is 3:
        return 'rgb'
    
def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    #if imput image is rgb 
    if rgb_or_grayscale(imgOrig) == 'rgb':
        # converts the RGB image to YIQ color space
        YIQ_img = transformRGB2YIQ(imgOrig)
        # multiplies the Y channel by 255 to scale its values to the range [0, 255]
        YIQ_img[: ,: ,0] = YIQ_img[: ,: , 0] * 255
        # calculates the histogram
        original_hist,bins = np.histogram(YIQ_img[:,:,0].flatten(),256,[0,255])
        cdf = original_hist.cumsum()
        # mask all values equal to 0
        cdf_m = np.ma.masked_equal(cdf, 0)
         
    elif rgb_or_grayscale(imgOrig) == 'gray':
        pass

    pass


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass
