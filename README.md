# Computer Vision Algorithms

This repository contains implementations of various computer vision algorithms. Each algorithm has its dedicated directory containing input materials, scripts, and output results. Below is a brief description of each implemented algorithm.

## Structure
For each algorithm, navigate to its corresponding directory:
- **`input/`**: Contains input materials required for the algorithm.
- **`output/`**: Contains final and intermediate results produced by the algorithm.

## Implemented Algorithms

### 1. Harris Corner Detection and Matching
- Implements the Harris corner detection algorithm to detect interest points in an image.
- Matches corresponding points between two images.
- Results are stored in the `output/` directory, including an image visualizing matched points.

#### Example Output:
![Harris Corner Detection and Matching Result](/Harris%20Corner%20Detection%20and%20Matching/Output/res11.jpg.jpg)

### 2. Perspective Transform
- Computes the perspective transformation of a given logo.
- Input and output images are stored in the respective directories.

### 3. Scene Matching using Homography & RANSAC
- Implements homography estimation and RANSAC to match scenes between two images.
- Helps in object detection and scene alignment.

### 4. Epipolar Geometry
- Computes epipolar lines between two images.
- Useful in stereo vision and depth estimation.

#### Example Output:
![Epipolar Geometry Result](/Epipolar%20Geometry%20/Output/res08.jpg)



### 5. Background Extraction, Stabilization, and Panorama Generation from Video
- Processes a video recorded by a human to generate different outputs.
- Extracts the background from the video, creating a separate background-only video.
- Generates a foreground video, indicating moving objects in the video.
- Creates a panorama image by stitching frames together.
- Stabilizes the video by removing camera shakes.
- Below are examples of the original and background-extracted videos:
  

  **Original Video:**  
  ![Original Video](/Creating%20Panorama%20from%20Video%20Background%20Extraction%20&%20Stabilization/Input/output_original.gif)


  **Background Video:**  
  ![background Video](/Creating%20Panorama%20from%20Video%20Background%20Extraction%20&%20Stabilization/Output/output_background.gif)


### 6. Scene Recognition Using Bag of Words
- Uses the Bag of Words (BoW) model to classify and recognize different scenes.
- Extracts feature descriptors and builds a vocabulary for classification.

### 7. Vanishing Points and Lines
- Detects vanishing points and lines in images to understand scene geometry.
- Useful in applications like architectural analysis and camera calibration.

### 8. Scene Recognition Using Neural Networks
- Implements a neural network-based approach for scene recognition.
- Trained on different scene categories for classification.

### 9. HOG Face Detection
- Uses Histogram of Oriented Gradients (HOG) to detect faces in images.
- The output includes detected faces with bounding boxes.

#### Example Output for HOG Face Detection:
![HOG Face Detection Result](/%20HOG%20Face%20Detection/Output/res4.jpg)

