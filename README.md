# Malaria Detection - Unified Project

This project combines two approaches for malaria parasite detection:

## 1. Deep Learning Approach (VGG19)
- **Technology**: Python, TensorFlow 2.0, VGG19
- **Method**: Transfer learning with pre-trained CNN
- **Accuracy**: 95.7%
- **Location**: `/deep-learning`

## 2. Image Processing Approach
- **Technology**: MATLAB
- **Method**: Traditional image processing (color thresholding, edge detection, Hough transform)
- **Features**: Parasite counting, cell detection
- **Location**: `/image-processing`

## Setup Instructions

### Deep Learning Approach
```bash
cd deep-learning
pip install -r requirements.txt
```

### Image Processing Approach
- Requires MATLAB installation
- Navigate to `/image-processing` and follow the README

## Dataset
NIH Malaria Cell Images Dataset

## Comparison
See `/docs/comparison.md` for detailed analysis of both approaches.