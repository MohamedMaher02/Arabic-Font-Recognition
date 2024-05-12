from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.metrics import accuracy_score
import skimage.io as io
from predict import predict_text_font
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import cv2 as cv

app = Flask(__name__, template_folder='./')




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

    contours, hierarchy = cv.findContours(output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

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

    contours, hierarchy = cv.findContours(output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
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

    contour, _ = cv.findContours(output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contour) == 0:
        plt.imshow(image, cmap='gray')
        plt.show()
        return
    max_contour = max(contour, key=cv.contourArea)

    x, y, w, h = cv.boundingRect(max_contour)
    contour_area = cv.contourArea(max_contour)
    rect_area = w*h


def preprocess_image(image, image_size=500):
    output = image
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



def precompute_gabor_kernels(ksize, sigma, lambd, gamma, psi):
    gabor_kernels = {}
    for theta in [0 , np.pi / 8.0 , 3.0 * np.pi / 4.0]:
        for freq in lambd:
            gabor_kernel = cv.getGaborKernel((ksize, ksize), sigma, theta, freq, gamma, psi)
            gabor_kernels[(theta, freq)] = gabor_kernel
    return gabor_kernels


def feature_extraction(img):
    ksize = 31
    sigma = 1.5
    lambd = [0.01 ,0.1,0.5, 4, 8, 12]
    gamma = 0.5
    psi = 0

    # Precompute Gabor Kernels
    gabor_kernels = precompute_gabor_kernels(ksize, sigma, lambd, gamma, psi)

    # Read and preprocess image
    image = preprocess_image(img, image_size=256)
    
    
    feature_vecotr = []
    for gabor_kernel in gabor_kernels.values():
        # Apply Gabor filter
        filtered_image = cv.filter2D(image, cv.CV_64F, gabor_kernel)

        # Calculate mean and standard deviation using numpy operations
        mean_val = np.mean(filtered_image)
        std_val = np.std(filtered_image)

        # Append to feature_vecotr
        feature_vecotr.extend([mean_val, std_val])
        
    
    
    dft = cv.dft(np.float32(image), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    passband_lower_cutoff = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225]
    passband_upper_cutoff = [30, 55, 80, 105, 130, 155, 180, 205, 230, 255]

    

    
    # Create a passband filter in the frequency domain
    rows, cols = dft_shift.shape[0], dft_shift.shape[1]
    crow, ccol = rows // 2, cols // 2
    

 
    for i,lower_cutoff in enumerate(passband_lower_cutoff):
        mask = np.zeros((rows, cols, 2), np.uint8)
        mask[crow - int(passband_upper_cutoff[i]):crow + int(passband_upper_cutoff[i]), 
            ccol - int(passband_upper_cutoff[i]):ccol + int(passband_upper_cutoff[i])] = 1
        mask[crow - int(passband_lower_cutoff[i]):crow + int(passband_lower_cutoff[i]), 
            ccol - int(passband_lower_cutoff[i]):ccol + int(passband_lower_cutoff[i])] = 0  
        filtered_dft = dft_shift * mask
        magnitude_spectrum = 20 * np.log(cv.magnitude(filtered_dft[:, :, 0], filtered_dft[:, :, 1]) + 1e-10)
        mean = np.mean(magnitude_spectrum)
        std = np.std(magnitude_spectrum)
        feature_vecotr.extend([mean, std])

    
    return np.array(feature_vecotr)

def predict_text_font(img_path):
    # Extract features from the image
    labels={3: 'IBM Plex Sans Arabic', 4: 'Lemonada', 5: 'Marhey', 11: 'Scheherazade New'}
    
    features = feature_extraction(img_path)

    # Create a DataFrame from the features
    num_features = len(features)
    df = pd.DataFrame(features.reshape(1, num_features), columns=[
                      f"Feature{i+1}" for i in range(num_features)])

    # Load the scaler model
    scaler = joblib.load('scaler_model.pkl')
    # Load the PCA model
    pca = joblib.load('pca_model.pkl')

    # Scale the features using the loaded StandardScaler
    X_test_scaled = scaler.transform(df)
    # Apply PCA transformation to the scaled features
    X_test_pca = pca.transform(X_test_scaled)

    # Load the trained classifier
    clf = joblib.load('trained_model.pkl')

    # Predict the label for the features
    y_pred = clf.predict(X_test_pca)

    # Map the predicted label to the corresponding text font
    predicted_font = labels[y_pred[0]]

    return predicted_font


@app.route('/')
def home():
    return render_template('./index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in request.'}), 400

    image_file = request.files['image']
    image = cv.imdecode(np.frombuffer(image_file.read(), np.uint8), cv.IMREAD_COLOR)

    prediction = predict_text_font(image)
    return jsonify({'prediction': prediction})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
