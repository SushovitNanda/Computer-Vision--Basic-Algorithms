# Assignment 1: Basic Image Processing and Edge Detection

This assignment implements fundamental computer vision algorithms from scratch using Python and NumPy. OpenCV is used only for image loading and preprocessing.

## Overview

This assignment covers four main computer vision tasks:

1. **RGB to Grayscale Conversion & Channel Analysis** - Implement luminance formula and analyze individual color channels
2. **Gaussian Noise Addition & Analysis** - Add noise to images and study its effects on processing
3. **Edge Detection Comparison** - Compare Sobel and Laplacian operators with different filter sizes
4. **Canny Edge Detector** - Complete implementation from scratch with all intermediate steps

## Libraries Used

- **NumPy** - Numerical computations and array operations
- **Matplotlib** - Visualization and plotting
- **OpenCV** - Limited to image loading and color space conversion only
- **Math** - Mathematical functions and kernel generation

## Question 1: RGB to Grayscale Conversion and Channel Analysis

### Objective
Read an RGB image, convert it to grayscale, visualize individual color channels (R, G, B), and compute histograms for each channel.

### Implementation

#### Grayscale Conversion
- **Luminance Formula**: `Y = 0.299×R + 0.587×G + 0.114×B`
- Weights are adjusted for BGR format: `[0.114, 0.587, 0.299]` for B, G, R channels

#### Channel Extraction
- Individual R, G, B channels extracted from BGR image
- Each channel displayed as a grayscale image with appropriate colormap

#### Histogram Computation
- Histograms computed for each color channel (256 bins)
- Statistical analysis: mean and standard deviation for each channel

### Results
- **Image Dimensions**: 1280×914×3 pixels, 3,509,760 total pixels
- **Channel Statistics**:
  - Red: Mean=128.55, Std=57.12
  - Green: Mean=122.03, Std=53.25
  - Blue: Mean=116.81, Std=50.92

## Question 2: Gaussian Noise Addition and Analysis

### Objective
Add Gaussian noise to an RGB image (mean=0, variance=10-20), then:
1. Convert noisy image to grayscale
2. Visualize individual color channels (R, G, B) and grayscale image
3. Compute and display histograms for each channel and grayscale image

### Implementation

#### Gaussian Noise Addition
- **Noise Generation**: `f(x) = (1/√(2πσ²)) * e^(-(x-μ)²/(2σ²))`
- Parameters: μ=0, σ²=20 (using maximum from specified range)
- Noise added to image and values clipped to [0, 255] range

#### Analysis
- Comparison between original and noisy images
- Histogram analysis showing noise effects
- SNR calculation and visualization

### Results
- **Noise Variance**: 20
- **Original Grayscale**: Mean=122.84, Std=53.67
- **Noisy Grayscale**: Mean=122.38, Std=53.75
- Noise effects visible in histogram distributions

## Question 3: Edge Detection with Multiple Filter Sizes

### Objective
Apply three different filters (3×3, 5×5, 7×7) on images with and without Gaussian noise. Perform edge detection using:
1. Sobel edge detector (combined)
2. Sobel operator in horizontal and vertical directions
3. Laplacian edge detector

Compare and describe differences across filter sizes and noise conditions.

### Implementation

#### Filter Application
- **Average (Box) Filter**: Applied before edge detection for noise reduction
- Filter sizes: 3×3, 5×5, 7×7

#### Sobel Edge Detector
- **Horizontal Gradient (Gx)**: Detects vertical edges
- **Vertical Gradient (Gy)**: Detects horizontal edges
- **Gradient Magnitude**: `|G| = √(Gx² + Gy²)`
- Extended kernels for 5×5 and 7×7 sizes

#### Laplacian Edge Detector
- **Second Derivative**: `∇²f = ∂²f/∂x² + ∂²f/∂y²`
- Detects edges by finding zero-crossings
- Extended kernels for larger sizes

### Results
- **3×3 filters**: Optimal edge detection quality with clear, well-defined edges
- **5×5 filters**: Moderate performance with some edge degradation
- **7×7 filters**: Poor edge detection with incomplete and disturbed results
- Noise significantly affects edge detection quality, especially with larger filters

## Question 4: Canny Edge Detector Implementation

### Objective
Implement the Canny edge detector from scratch following these steps:
1. Apply Gaussian smoothing
2. Compute gradient magnitude and orientation using Sobel filters
3. Apply non-maximum suppression
4. Apply hysteresis thresholding (use two thresholds)

Apply to both clean and noisy images and show how noise affects results.

### Implementation

#### Step 1: Gaussian Smoothing
- **Gaussian Kernel**: `G(x,y) = (1/(2πσ²)) * e^(-(x²+y²)/(2σ²))`
- Parameters: kernel_size=5, σ=1.4
- Reduces noise while preserving edge information

#### Step 2: Gradient Computation
- **Sobel Operators**: Compute Gx and Gy gradients
- **Gradient Magnitude**: `|G| = √(Gx² + Gy²)`
- **Gradient Orientation**: `θ = arctan(Gy/Gx)` (converted to degrees, range [0, 180))

#### Step 3: Non-Maximum Suppression
- Thins edges by keeping only local maxima along gradient direction
- Compares each pixel with neighbors in gradient direction
- Preserves only strongest edge responses

#### Step 4: Hysteresis Thresholding
- **Dual Thresholds**:
  - High threshold: 15% of maximum (strong edges)
  - Low threshold: 5% of maximum (weak edges)
- **Edge Connection**: Weak edges kept only if connected to strong edges
- **8-connectivity**: Checks all 8 neighbors for edge connectivity

### Results
- **Parameters**: σ=1.4, Low threshold=5%, High threshold=15%
- **Clean Image**: Clear, well-defined edges with minimal noise
- **Noisy Image**: More edge artifacts, but still maintains edge structure
- Canny detector provides robust edge detection compared to simple Sobel/Laplacian methods

## Key Findings

1. **Filter Size Impact**: Smaller filters (3×3) provide superior edge detection quality
2. **Noise Sensitivity**: All algorithms show varying sensitivity to Gaussian noise
3. **Canny Superiority**: Canny edge detector provides most robust edge detection
4. **Parameter Optimization**: Careful parameter tuning crucial for optimal performance

## Visualizations

The implementation includes comprehensive visualizations for:
- Original image and color channel histograms
- Grayscale conversion results
- Noise addition effects and comparisons
- Edge detection results across different methods and filter sizes
- Complete Canny algorithm step-by-step visualization

