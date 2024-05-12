import pandas as pd
import numpy as np
import cv2 as cv
from preprocessing import *
import joblib



def precompute_gabor_kernels(ksize, sigma, lambd, gamma, psi):
    gabor_kernels = {}
    for theta in [0 , np.pi / 8.0 , 3.0 * np.pi / 4.0]:
        for freq in lambd:
            gabor_kernel = cv.getGaborKernel((ksize, ksize), sigma, theta, freq, gamma, psi)
            gabor_kernels[(theta, freq)] = gabor_kernel
    return gabor_kernels


def feature_extraction(img_path):
    ksize = 31
    sigma = 1.5
    lambd = [0.01 ,0.1,0.5, 4, 8, 12]
    gamma = 0.5
    psi = 0

    # Precompute Gabor Kernels
    gabor_kernels = precompute_gabor_kernels(ksize, sigma, lambd, gamma, psi)

    # Read and preprocess image
    image = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    
    
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



Labels = {3: 'IBM Plex Sans Arabic', 4: 'Lemonada', 5: 'Marhey', 11: 'Scheherazade New'}
img_path = "C:\\Users\\Mohamad Ameen\\Pictures\\Screenshots\\Screenshot 2024-05-12 043256.png"
features = feature_extraction(img_path)


num_features = len(features) # features is an np.array containing 62 elements
df = pd.DataFrame(features.reshape(1, num_features))

# Assign column names from 'Feature1' to 'FeatureN'
column_names = [f"Feature{i+1}" for i in range(num_features)]
df.columns = column_names

df.head()



scaler = joblib.load('C:\\Users\\Mohamad Ameen\\Desktop\\NN-Project\\Arabic-Font-Recognition\\scaler_model.pkl')
# Load the PCA model
pca = joblib.load('C:\\Users\\Mohamad Ameen\\Desktop\\NN-Project\\Arabic-Font-Recognition\\pca_model.pkl')

# Assuming you have your test data in a DataFrame called 'df_test'
# Scale the test data using the loaded StandardScaler
X_test_scaled = scaler.transform(df)

# Apply PCA transformation to the scaled test data
X_test_pca = pca.transform(X_test_scaled)

clf = joblib.load('C:\\Users\\Mohamad Ameen\\Desktop\\NN-Project\\Arabic-Font-Recognition\\trained_model.pkl')

y=clf.predict(X_test_pca)

print(Labels[y[0]])
