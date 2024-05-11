
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv
import time


def detect_background_color(image):
    width, height = image.shape[:2]
    corners = [(0, 0), (0, height-1), (width-1, 0), (width-1, height-1)]
    corners_colors = []
    for corner in corners:
        corners_colors.append(image[corner[0], corner[1]])
    background_color = np.mean(corners_colors, axis=0)
    return background_color


def remove_additive_noise(image, kernal: int):
    image = cv.filter2D(image, -1, np.ones((kernal, kernal)))
    return image


def remove_implusive_noise(image, kernal: int):
    image = cv.medianBlur(image, kernal)
    return image


def resize_image(image, width: int, height: int):
    image = cv.resize(image, (width, height))
    return image


def threshold_image(image, threshold: int):
    _, output = cv.threshold(image, 50, 255, cv.THRESH_BINARY)
    return output


def convert_to_binary(image):
    output = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, output = cv.threshold(image, 50, 255, cv.THRESH_BINARY)
    output = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return output


def rotate_45(image):
    angle = 45
    height, width = image.shape[:2]
    background_color = 0
    background = np.full((height, width, 3), background_color, dtype=np.uint8)
    rotation_matrix = cv.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv.warpAffine(image, rotation_matrix, (width, height))
    result = cv.bitwise_and(rotated_image, rotated_image, mask=image)
    return result


def flip_image(image):
    return cv.flip(cv.flip(image, 0), 1)


def rotate_image(image):
    return cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)


def closing(image, height: int, width: int):
    kernal = np.ones((height, width), np.uint8)
    image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernal)
    return image


def openning(image, height: int, width: int):
    kernal = np.ones((height, width), np.uint8)
    image = cv.morphologyEx(image, cv.MORPH_OPEN, kernal)
    return image


def errode(image, height=1, width=1, iterations=1, kernal=None):
    kernal = np.ones((height, width), np.uint8) if kernal is None else kernal
    image = cv.erode(image, kernal, iterations=iterations)
    return image


def dilate(image, height=1, width=1, iterations=1, kernal=None):
    kernal = np.ones((height, width), np.uint8) if kernal is None else kernal
    image = cv.dilate(image, kernal, iterations=iterations)
    return image


def detect_if_flipped(image):
    output = image.copy()
    _, height = image.shape[:2]

    output = closing(image, 1, 25)

    output = dilate(output, 1, 15, 2)
    output = errode(output, 1, 90, 4)

    contours, hierarchy = cv.findContours(
        output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # for countour in contours:
    #     x, y, w, h = cv.boundingRect(countour)
    #     cv.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)

    # plt.imshow(image, cmap='gray')
    # plt.show()

    rotated = 0
    not_rotated = 0
    padding = 7
    for countour in contours:
        x, y, w, h = cv.boundingRect(countour)
        cropped_image = image[y-padding:y+h+padding, x-padding:x+w+padding]

        # # Calculate the size of each half
        height, width = cropped_image.shape[:2]
        half_height = height // 2

        # # Split the image into two halves
        half1 = cropped_image[:half_height, :]
        half2 = cropped_image[half_height:, :]

        sum1 = np.sum(half1)
        sum2 = np.sum(half2)

        if sum1 < sum2:
            not_rotated += 1
        else:
            rotated += 1

    if rotated > not_rotated:
        return True
    else:
        return False


def detect_if_vertical(image):
    output = image.copy()
    width, height = image.shape[:2]

    output = closing(image, 25, 1)

    output = dilate(output, 15, 1, 1)
    output = errode(output, 15, 1, 2)

    contours, hierarchy = cv.findContours(
        output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours is None or len(contours) == 0:
        return False
    max_contour = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(max_contour)

    aspect_ratio = w/h

    if aspect_ratio > 0.1:
        return False
    else:
        return True


def detect_if_rotated(image):
    output = image.copy()
    width, height = image.shape[:2]

    output = closing(image, 1, 25)

    output = dilate(output, 1, 15, 2)
    output = errode(output, 1, 90, 4)

    plt.imshow(output, cmap='gray')
    plt.show()

    contour, _ = cv.findContours(
        output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contour) == 0:
        plt.imshow(image, cmap='gray')
        plt.show()
        return
    max_contour = max(contour, key=cv.contourArea)

    x, y, w, h = cv.boundingRect(max_contour)
    contour_area = cv.contourArea(max_contour)
    rect_area = w*h


def preprocess_image(image_file, image_size=500):
    image0 = plt.imread(image_file, 0)
    output = image0
    output = convert_to_binary(output)
    output = remove_implusive_noise(output, 5)
    output = resize_image(output, image_size, image_size)
    bg = detect_background_color(output)
    if bg > 127.0:
        output = cv.bitwise_not(output)

    # for preformace reasons
    vertical = False
    if detect_if_vertical(output):
        vertical = True
        output = rotate_image(output)

    if detect_if_flipped(output) and not vertical:
        output = flip_image(output)

    return output


directory = 'C:\\Users\\Mohamad Ameen\\Desktop\\NN-Project\\fonts-dataset\\Scheherazade New'
save_directory = 'C:\\Users\\Mohamad Ameen\\Desktop\\NN-Project\\fonts-dataset\\Scheherazade New preprocessed'
files = os.listdir(directory)
for i in files:
    image = preprocess_image(directory + '\\' + i)
    plt.imsave(save_directory + '\\' + i, image, cmap='gray')
