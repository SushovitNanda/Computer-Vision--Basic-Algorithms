# Computer Vision - Basic Algorithms

A comprehensive implementation of fundamental computer vision algorithms from scratch using Python and NumPy. This repository contains two assignments covering image processing, edge detection, feature extraction, and texture classification.

## Repository Structure

```
Computer-Vision--Basic-Algorithms-main/
├── Assignment-1/          # Basic Image Processing and Edge Detection
│   ├── CV_1.ipynb        # Main implementation notebook
│   ├── CV_Assignment_1.pdf
│   └── README.md
├── Assignment-2/          # Feature Extraction and Texture Classification
│   ├── cv2.ipynb         # Main implementation notebook
│   ├── CV_Assignment_2.pdf
│   └── README.md
└── README.md             # This file
```

## Assignment 1: Basic Image Processing and Edge Detection

### Topics Covered
1. **RGB to Grayscale Conversion** - Luminance formula implementation and channel analysis
2. **Gaussian Noise Analysis** - Noise addition, SNR calculation, and histogram analysis
3. **Edge Detection Methods** - Sobel and Laplacian operators with multiple filter sizes
4. **Canny Edge Detector** - Complete from-scratch implementation with all steps

### Key Algorithms
- Luminance conversion: `Y = 0.299×R + 0.587×G + 0.114×B`
- Gaussian noise: `f(x) = (1/√(2πσ²)) * e^(-(x-μ)²/(2σ²))`
- Sobel edge detection: Gradient magnitude computation
- Laplacian edge detection: Second derivative-based detection
- Canny algorithm: Gaussian smoothing → Gradient computation → Non-maximum suppression → Hysteresis thresholding

### Results Highlights
- **Filter Size Impact**: 3×3 filters provide optimal edge detection
- **Canny Parameters**: σ=1.4, Low threshold=5%, High threshold=15%
- **Noise Effects**: All methods show varying sensitivity to Gaussian noise

[See Assignment-1 README for details](./Assignment-1/README.md)

## Assignment 2: Feature Extraction and Texture Classification

### Topics Covered
1. **Integral Image** - Efficient region sum computation
2. **Haar-like Features** - Four pattern types for feature extraction
3. **Texture Classification** - Four methods on KTH-TIPS dataset:
   - Raw pixel intensity
   - Local Binary Pattern (LBP)
   - Bag of Words (BoW)
   - Histogram of Oriented Gradients (HOG)

### Key Algorithms
- Integral image: O(1) region sum after O(n²) preprocessing
- LBP: 8-neighbor circular pattern encoding
- BoW: K-means vocabulary building and histogram representation
- HOG: Gradient orientation histograms in cells

### Results Highlights
- **Best Performance**: LBP achieves 93.83% accuracy
- **Dataset**: KTH-TIPS (10 classes, 810 images)
- **Feature Dimensions**: Range from 50 (BoW) to 40,000 (Raw Pixels)

[See Assignment-2 README for details](./Assignment-2/README.md)

## Libraries Used

### Core Libraries
- **NumPy** - Numerical computations and array operations
- **Matplotlib** - Visualization and plotting
- **OpenCV** - Image loading and preprocessing (limited use)
- **scikit-learn** - Machine learning tools (SVM, K-means, preprocessing)
- **PIL (Pillow)** - Image format conversion
- **Math** - Mathematical functions

## Key Features

### From-Scratch Implementation
- All algorithms implemented from first principles
- No reliance on high-level computer vision libraries for core algorithms
- OpenCV used only for basic I/O operations

### Comprehensive Analysis
- Statistical analysis of results
- Performance comparisons
- Visualization of intermediate steps
- Parameter sensitivity analysis

### Educational Value
- Well-documented code with mathematical formulations
- Step-by-step algorithm explanations
- Visual demonstrations of each processing stage

## Results Summary

### Assignment 1: Edge Detection
- **Best Method**: Canny edge detector
- **Optimal Filter Size**: 3×3 for Sobel/Laplacian
- **Noise Impact**: Significant degradation with larger filters

### Assignment 2: Texture Classification
- **Best Method**: Local Binary Pattern (LBP) - 93.83% accuracy
- **Feature Efficiency**: LBP provides best accuracy-to-dimension ratio
- **Dataset**: Successfully classified 10 texture classes from KTH-TIPS

## Visualizations

Both assignments include extensive visualizations:
- Original and processed images
- Histograms and statistical plots
- Step-by-step algorithm visualization
- Performance comparison charts
- Feature visualization

## Getting Started

### Prerequisites
```bash
pip install numpy matplotlib opencv-python scikit-learn pillow
```

### Running the Code
1. Open the Jupyter notebooks in each assignment directory
2. Ensure required images/datasets are in the correct paths
3. Run cells sequentially to reproduce results

### Dataset Requirements
- **Assignment 1**: Requires an RGB image file (`Image.jpg`)
- **Assignment 2**: Requires KTH-TIPS dataset in specified directory

## Key Findings

1. **Filter Size Matters**: Smaller filters (3×3) provide better edge detection quality
2. **Feature Engineering Critical**: LBP outperforms raw pixels by 51% in texture classification
3. **Noise Sensitivity**: All algorithms show varying sensitivity to Gaussian noise
4. **Canny Superiority**: Canny edge detector provides most robust edge detection
5. **LBP Excellence**: Local Binary Pattern is highly effective for texture classification

## Report

Detailed PDF reports are included with each assignment covering:
- Mathematical formulations
- Algorithm explanations
- Results analysis
- Performance comparisons
- Visual documentation

