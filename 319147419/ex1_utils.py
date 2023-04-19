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
    return 319147419


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
    inverse_matrix = np.linalg.inv(tranfer_matrix.transpose())
    # save original shape of image 
    img_shape = imgYIQ.shape
    # matrix multiplication 
    return np.dot(imgYIQ.reshape(-1, 3), inverse_matrix).reshape(img_shape)
    
def rgb_or_grayscale(img: np.ndarray):
    if len(img.shape) == 2:
        return 'gray'
    elif len(img.shape) == 3:
        return 'rgb'
    
def hsitogramEqualize(imgOrig: np.ndarray) :
    """
        Equalizes the histogram of an image , if input image is rgb image so we transform to YIQ and then take the y channel(0).
        after that we convert the pixels of y_channel values from the range of [0, 1] to [0, 255]
        :param imgOrig: Original Histogram
        :ret
    """
    # Determine if the image is grayscale or RGB
    img_type = rgb_or_grayscale(imgOrig)
    # Convert RGB image to YIQ color space and extract Y channel if necessary
    if img_type == 'rgb':
        yiq_img = transformRGB2YIQ(imgOrig)
        y_channel = yiq_img[:, :, 0] 
    else:
        y_channel = imgOrig

    # Compute the histogram and equalize the Y channel
    y_channel, histo, new_histo = hist_normalize(y_channel)

    # Convert the image back to RGB color space if need
    if img_type == 'rgb':
        yiq_img[:, :, 0] = y_channel
        img_copy = transformYIQ2RGB(yiq_img)
    else:
        img_copy = y_channel

    return img_copy, histo, new_histo

def hist_normalize(y_channel):
    y_channel = (y_channel * 255).astype(np.int)
    y_channel, histo, new_histo = equalize(y_channel)
    y_channel = y_channel/255
    return y_channel,histo,new_histo

def equalize(channel):
    # Flatten the channel to 1D array
    flat_channel = channel.flatten()
    print(channel)
    # Compute the histogram
    old_histogram, _ = np.histogram(flat_channel, bins=256, range=(0, 255))
    # Compute the cumulative sum of the histogram
    cumsum = np.cumsum(old_histogram)
    # Compute the maximum value in the channel
    max_val = np.max(channel)
    # take the total number of pixels in the channel
    num_total_pixels = cumsum[-1]
    # create look up table for mapping the values to the new values
    lu_table = [np.ceil(max_val * m / num_total_pixels) for m in cumsum] # np.ceil function is used to round up the new pixel values to the nearest integer
    # Map the values in the channel using the lookup table
    new_image = np.array([lu_table[pixel] for pixel in flat_channel])
    # Reshape the mapped values to the original channel shape
    new_image = new_image.reshape(channel.shape)
    # Compute the new histogram of the mapped channel
    new_histogram, _ = np.histogram(new_image, bins=256, range=(0, 255))
    return new_image, old_histogram, new_histogram



def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    if len(imOrig.shape) == 2:  
        # If the image is already a gray scale image
        return quantize_channel(imOrig.copy(), nQuant, nIter)
    # Convert the RGB image to YIQ color space
    yiq_img = transformRGB2YIQ(imOrig)
    # Quantize the Y channel
    qImage_, mse = quantize_channel(yiq_img[:, :, 0].copy(), nQuant, nIter)  
    qImage = []
    # Convert each quantized Y channel image back to RGB color space
    for img in qImage_:
        qImage_i = transformYIQ2RGB(np.dstack((img, yiq_img[:, :, 1], yiq_img[:, :, 2])))
        qImage.append(qImage_i)
    # Return the list of quantized RGB images and the list of errors (MSE)
    return qImage, mse


def quantize_channel(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an single channel image (grey channel) in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """

    Images = []
    errors = []

    # Normalize the pixel values of the input image to the range [0,255]
    imOrig = cv2.normalize(imOrig, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # Flatten the image into a 1D array
    imOrig_flat = imOrig.flatten().astype(int)
    # Compute the histogram of the pixel intensities
    histOrg, edges = np.histogram(imOrig_flat, bins=256)
    # Initialize the quantization borders to be equally spaced
    z_board = np.zeros(nQuant + 1, dtype=int) 
    for i in range(nQuant + 1):
        z_board[i] = i * (255.0 / nQuant)

    
    # Perform the optimization loop
    for i in range(nIter):
        # Calculate the average weighted pixel intensity for each quantization region
        average_weighted = []
        average_weighted,z_board = image_quantization(nQuant,histOrg,z_board,average_weighted)

        # Quantize the image using the updated quantization borders
        qImage_i = np.zeros_like(imOrig)
        for k in range(len(average_weighted)):
            true_labels = imOrig > z_board[k]
            qImage_i[imOrig > z_board[k]] = average_weighted[k]

        # Calculate the mean squared error between the original and quantized images
        mse = np.sqrt((imOrig - qImage_i) ** 2).mean()
        errors.append(mse)
        # Append the quantized image to the list of images
        Images.append(qImage_i / 255.0)  
        for k in range(len(average_weighted) - 1):
            z_board[k + 1] = (average_weighted[k] + average_weighted[k + 1]) / 2  

    return Images, errors

def image_quantization(n_levels, histogram, initial_boundaries, final_boundaries):
    # loop over the quantization levels
    for i in range(n_levels):
        # extract the intensities of the pixels in the i-th bin
        bin_intensities = histogram[initial_boundaries[i]:initial_boundaries[i + 1]]
        # create an array of indices corresponding to the intensities
        bin_indices = range(len(bin_intensities))
        # compute the weighted mean intensity of the i-th bin
        bin_weighted_mean = (bin_intensities * bin_indices).sum() / np.sum(bin_intensities)
        # add the weighted mean to the list of final boundaries
        final_boundaries.append(initial_boundaries[i] + bin_weighted_mean)
    
    # return the list of final boundaries and the initial boundary array
    return final_boundaries, initial_boundaries
