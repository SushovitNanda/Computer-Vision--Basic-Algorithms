# Assignment 2: Feature Extraction and Texture Classification

This assignment implements advanced computer vision algorithms for feature extraction and texture classification from scratch using Python, NumPy, and scikit-learn.

## Overview

This assignment covers two main computer vision tasks:

1. **Integral Image and Haar-like Features** - Implement integral image computation and extract Haar-like features
2. **Texture Classification** - Compare multiple feature extraction methods for texture classification on the KTH-TIPS dataset

## Libraries Used

- **NumPy** - Numerical computations and array operations
- **OpenCV** - Image loading and preprocessing
- **Matplotlib** - Visualization and plotting
- **scikit-learn** - SVM classifier, train-test split, standardization, K-means clustering
- **PIL (Pillow)** - Image loading and format conversion
- **Math** - Mathematical functions

## Question 1: Integral Image and Haar-like Features

### Part 1: Integral Image Computation

#### Objective
Implement integral image computation and verify correctness by comparing region sums computed using integral image vs. direct computation.

#### Implementation

**Integral Image Formula**:
```
I(x,y) = Σ(i=0 to x) Σ(j=0 to y) image(i,j)
```

**Region Sum Computation**:
```
Sum(x1,y1,x2,y2) = I(x2,y2) - I(x1-1,y2) - I(x2,y1-1) + I(x1-1,y1-1)
```

**Features**:
- Efficient O(1) region sum computation after O(n²) preprocessing
- Verification on multiple test regions
- Visualization of original and integral images

#### Results
- Successfully verified integral image correctness on multiple regions
- Test image: 256×256 pixels
- All verification tests passed with negligible error (< 1e-6)

### Part 2: Haar-like Feature Extraction

#### Objective
Extract Haar-like features from a 50×50 center region of an image using four different patterns.

#### Implementation

**Haar Patterns Implemented**:
1. **Horizontal [White | Black]**: Detects vertical edges
2. **Vertical [White / Black]**: Detects horizontal edges
3. **Three-rectangle [White | Black | White]**: Detects more complex patterns
4. **Four-rectangle [White|Black / Black|White]**: Detects diagonal patterns

**Feature Extraction**:
- Filter size: 24×24 pixels
- Center region: 50×50 pixels
- Sliding window with stride of 2 pixels
- Each position extracts 4 features (one per pattern)

**Features**:
- Total features extracted: 784 (196 per pattern)
- Feature range: -5248 to 26368
- Mean feature value: 5171.51

## Question 2: Texture Classification

### Objective
Classify textures from the KTH-TIPS dataset using four different feature extraction methods:
1. Raw pixel intensity values
2. Local Binary Pattern (LBP)
3. Bag of Words (BoW)
4. Histogram of Oriented Gradients (HOG)

### Dataset
- **KTH-TIPS Dataset**: 10 texture classes
  - aluminium_foil, brown_bread, corduroy, cotton, cracker
  - linen, orange_peel, sandpaper, sponge, styrofoam
- **Total Images**: 810 (81 per class)
- **Image Size**: Resized to 200×200 pixels
- **Train-Test Split**: 70% training (567 images), 30% testing (243 images)

### Method 1: Raw Pixel Intensity Classification

#### Implementation
- Flatten images to 1D feature vectors (200×200 = 40,000 features per image)
- Standardize features using StandardScaler
- Train linear SVM classifier

#### Results
- **Feature Dimensions**: 40,000 features per image
- **Accuracy**: 42.39%
- **Analysis**: Poor performance due to high dimensionality and lack of texture-specific features

### Method 2: Local Binary Pattern (LBP) Classification

#### Implementation

**LBP Algorithm**:
- **Parameters**: P=8 neighbors, R=1 radius
- Sample P points in circular neighborhood around each pixel
- Use bilinear interpolation for sub-pixel accuracy
- Encode pattern as 8-bit binary number
- Compute histogram of LBP codes (256 bins)
- Normalize histogram to create probability distribution

**Features**:
- Illumination invariant
- Captures local texture patterns
- Feature dimensions: 256 features per image

#### Results
- **Feature Dimensions**: 256 features per image
- **Accuracy**: 93.83%
- **Analysis**: Excellent performance - LBP effectively captures texture patterns

### Method 3: Bag of Words (BoW) Classification

#### Implementation

**Feature Extraction**:
- Extract dense features from overlapping patches (patch_size=16, stride=8)
- Compute gradients using central difference
- Combine gradient x and y components
- Normalize feature vectors

**Visual Vocabulary**:
- Extract features from all training images
- Build vocabulary using K-means clustering (vocabulary_size=50)
- Assign features to visual words
- Create histogram of visual word occurrences
- Normalize histogram

**Features**:
- Feature dimensions: 50 features per image
- Captures global texture structure

#### Results
- **Feature Dimensions**: 50 features per image
- **Accuracy**: 67.90%
- **Analysis**: Moderate performance - BoW captures spatial structure but may lose fine texture details

### Method 4: Histogram of Oriented Gradients (HOG) Classification

#### Implementation

**HOG Algorithm**:
- **Parameters**: 9 orientations, 16×16 pixels per cell
- Compute gradients using central difference
- Calculate gradient magnitude and orientation
- Divide image into cells (cells_y × cells_x)
- Compute orientation histogram for each cell
- Normalize HOG features

**Features**:
- Captures local gradient information
- Robust to illumination changes
- Feature dimensions: 1296 features per image (for 200×200 image with 16×16 cells)

#### Results
- **Feature Dimensions**: 1296 features per image
- **Accuracy**: 79.42%
- **Analysis**: Good performance - HOG effectively captures texture gradient patterns

## Performance Comparison

### Accuracy Results

| Method | Accuracy | Feature Dimensions |
|--------|----------|-------------------|
| Raw Pixels | 42.39% | 40,000 |
| LBP | **93.83%** | 256 |
| Bag of Words | 67.90% | 50 |
| HOG | 79.42% | 1,296 |

### Key Findings

1. **LBP Best Performance**: Local Binary Pattern achieves highest accuracy (93.83%)
   - Effective at capturing local texture patterns
   - Illumination invariant
   - Compact feature representation (256 dimensions)

2. **Raw Pixels Worst Performance**: Direct pixel intensity values perform poorly (42.39%)
   - High dimensionality (40,000 features)
   - No texture-specific information
   - Sensitive to illumination and noise

3. **HOG Good Performance**: Histogram of Oriented Gradients achieves good accuracy (79.42%)
   - Captures gradient information effectively
   - More features than LBP but still manageable

4. **BoW Moderate Performance**: Bag of Words achieves moderate accuracy (67.90%)
   - Compact representation (50 features)
   - May lose fine texture details due to quantization

### Computational Complexity

- **Raw Pixels**: Highest dimensionality, slowest training
- **LBP**: Moderate dimensionality, fast feature extraction
- **BoW**: Lowest dimensionality, requires vocabulary building
- **HOG**: Moderate dimensionality, efficient computation

## Visualizations

The implementation includes visualizations for:
- Integral image computation and verification
- Haar-like feature patterns
- HOG gradient magnitude and orientation
- Performance comparison charts (accuracy and feature dimensions)

