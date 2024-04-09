### Constraints
1. Shadow effect: use dark braground
2. Object boundry: use contrasting background

## Getting Started

### Prerequisites
Python 3
Pip
OpenCV
Numpy

### Installing
prerequisites:
- pip install numpy
- pip install opencv-python
- pip install streamlit

## Algorithm
1. Image pre-processing
  - Read an image and convert it it no grayscale
  - Blur the image using Gaussian Kernel to remove un-necessary edges
  - Edge detection using Canny edge detector
  - Perform morphological closing operation to remove noisy contours

2. Object Segmentation
  - Find contours
  - Remove small contours by calculating its area (threshold used here is 100)
  - Sort contours from left to right to find the reference objects
  
3. Reference object 
  - Calculate how many pixels are there per metric (centi meter is used here)

4. Compute results
  - Draw bounding boxes around each object and calculate its height and width

### Testing
- To click each and every frames while live streaming
  Run python stream_video.py
- To directly detect from live straming
  Rn python live.py






 
