# Arabic Font Recognition Project

This project aims to develop an Arabic font recognition system using a regression forest model. It preprocesses input images, extracts relevant features using techniques such as Gabor filters and spectral analysis, and then classifies the fonts based on the extracted features.

## Table of Contents

- [Project Pipeline](#project-pipeline)
- [Preprocessing](#preprocessing)
- [Feature Extraction](#feature-extraction)
- [Model Selection](#model-selection)
- [Performance Analysis](#performance-analysis)
- [Future Work](#future-work)
- [How to Use](#how-to-use)
- [Installation](#installation)

---

## Project Pipeline

The project takes a dataset of images and preprocesses them to train the model. The preprocessing stage prepares the data for feature extraction. The extracted features are then fed into a regression forest model, which is trained and tested.

### Key Stages:

1. **Preprocessing** - Image binarization, noise removal, resizing, and rotation correction.
2. **Feature Extraction** - Gabor filters for texture analysis and spectral analysis for frequency domain behavior.
3. **Model Selection** - Regression forest model used for final training and testing.
4. **Performance Analysis** - Cross-validation and accuracy testing.

## Preprocessing

The preprocessing pipeline includes several steps:

1. **Binarization**: Converts the image to binary format.
2. **Noise Removal**: A median filter removes impulsive noise from the image.
3. **Resizing**: Each image is resized to 300x300 pixels.
4. **Background Color Detection**: Detects and unifies the background color.
5. **Rotation Correction**: Detects and corrects the orientation (90-degree or 180-degree rotations) of the image.

### Steps for Background and Rotation Correction:

- **Background Color**: The most frequent corner color is assumed to be the background color.
- **Rotation Detection**: Closing operations detect contours, and bounding box aspect ratios help identify incorrect rotations.
- **Upside Down Detection**: Analyzes lines and pixel distribution to ensure correct orientation.

## Feature Extraction

Two primary techniques are used for feature extraction:

1. **Gabor Filter**: Utilizes two rotation angles and two sinusoidal wavelengths to extract texture features by aggregating standard deviation and mean values.
2. **Spectral Analysis**: Converts the image to the frequency domain using a Fourier Transform to analyze the frequency behavior in the text image.

## Model Selection

After testing various models, the **Regression Forest Model** provided the best performance for recognizing Arabic fonts. It was trained on 80% of the dataset and tested on 20% using 5-fold cross-validation.

## Performance Analysis

The model achieved an **accuracy of 96%** after cross-validation. 

## Future Work

- Implement text segmentation to extract features from individual words rather than entire paragraphs for more precise classification.

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/MohamedMaher02/Arabic-Font-Recognition
cd your-repo-name
```
### 2. Run the Flask Server

To start the Flask development server, run:

```bash
python app.py
```

The server will start at http://127.0.0.1:5000.

How to Use
----------

### Access via Web Interface

1.  Open your browser and navigate to http://127.0.0.1:5000.
    
2.  Use the **Choose Image** button to upload an image.
    
3.  Click **Upload Image** to send the image to the server for font prediction.
    
4.  The predicted Arabic font will be displayed below the upload button.
