#ifndef SEGMENTER_H
#define SEGMENTER_H

#include "ImageLoader.h"
#include "ImagePreprocess.h"

// Represents a sequence of segmented characters from a phrase image
typedef struct {
    float **chars;      // Array of character data (each is targetSize x targetSize floats)
    int count;          // Number of characters found
    int capacity;       // Allocated capacity for chars array
    int charSize;       // Size of each character (targetSize x targetSize)
} CharSequence;

// Configuration for phrase segmentation
typedef struct {
    int targetSize;         // Target grid size for each char (pex, 8 for 8x8, 16 for 16x16)
    float binarizeThreshold;// Threshold for binarization (0.0-1.0)
    int minCharWidth;       // Minimum character width in pixels (to filter noise)
    int spaceThreshold;     // Gap width (in pixels) to consider as word space
} SegmenterConfig;

// Default segmenter configuration
static inline SegmenterConfig defaultSegmenterConfig(int targetSize) {
    return (SegmenterConfig){
        .targetSize = targetSize,
        .binarizeThreshold = 0.5f,
        .minCharWidth = 3,
        .spaceThreshold = 0  // 0 = auto-calculate based on average char width * 1.5
    };
}

// Segments a phrase image into individual characters
// Returns NULL on failure
CharSequence* segmentPhrase(RawImage *img, SegmenterConfig cfg);

// Frees a CharSequence and all its character data
void freeCharSequence(CharSequence *seq);

// Adds a space marker to the sequence (represented as NULL pointer in chars array)
// Returns 0 on success, -1 on failure
int charSequenceAddSpace(CharSequence *seq);

// Adds a character to the sequence
// Returns 0 on success, -1 on failure
int charSequenceAddChar(CharSequence *seq, float *charData, int charSize);

#endif // SEGMENTER_H