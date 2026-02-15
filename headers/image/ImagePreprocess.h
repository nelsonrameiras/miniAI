#ifndef IMAGE_PREPROCESS_H
#define IMAGE_PREPROCESS_H

#include "ImageLoader.h"

// Config of preprocessing
typedef struct {
    int targetSize;      // Target size (pex, 8 for 8x8, 5 for 5x5)
    float threshold;     // Threshold of binarization (0.0-1.0)
    int invertColors;    // 1 = invert (white->black), 0 = keep
} PreprocessConfig;

// Converts RawImage to array of floats normalized
float* imagePreprocess(RawImage *img, PreprocessConfig cfg);

// Convert RGB pixel to grayscale using luminance formula
uint8_t rgbToGray(uint8_t r, uint8_t g, uint8_t b);

// Convert image to grayscale (caller must free result)
uint8_t* convertToGrayscale(RawImage *img);

// Calculate Otsu's threshold for binarization
uint8_t calculateOtsuThreshold(uint8_t *gray, int totalPixels);

#endif