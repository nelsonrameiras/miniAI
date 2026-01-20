#include "../headers/Segmenter.h"
#include "../headers/ImagePreprocess.h"  // shared utility functions
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// Initial capacity for character sequence
#define INITIAL_CAPACITY 32

// --- Internal helper structures ---

typedef struct {
    int left;
    int right;
} CharBounds;

// --- Internal helper functions (static) ---

// Check if background is light (should invert for dark-on-light text)
static int shouldInvertColors(uint8_t *gray, int width, int height) {
    float borderSum = 0;
    int borderCount = 0;

    // Sample border pixels
    for (int x = 0; x < width; x++) {
        borderSum += gray[x];
        borderSum += gray[(height - 1) * width + x];
        borderCount += 2;
    }
    for (int y = 1; y < height - 1; y++) {
        borderSum += gray[y * width];
        borderSum += gray[y * width + (width - 1)];
        borderCount += 2;
    }

    float avgBorder = borderSum / borderCount;
    return (avgBorder > 128) ? 1 : 0;
}

// Compute vertical projection (sum of foreground pixels per column)
static int* computeVerticalProjection(uint8_t *binary, int width, int height) {
    int *projection = (int*)calloc(width, sizeof(int));
    if (!projection) {
        fprintf(stderr, "Error: Failed to allocate projection array\n");
        return NULL;
    }

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            if (binary[y * width + x] == 1) {
                projection[x]++;
            }
        }
    }

    return projection;
}

// Find character boundaries from vertical projection
// Returns array of CharBounds, sets *numChars
static CharBounds* findCharBoundaries(int *projection, int width, int minCharWidth, 
                                       int *numChars, int **gapWidths) {
    // First pass: count characters and find gaps
    int capacity = INITIAL_CAPACITY;
    CharBounds *bounds = (CharBounds*)malloc(capacity * sizeof(CharBounds));
    int *gaps = (int*)malloc(capacity * sizeof(int));
    if (!bounds || !gaps) {
        free(bounds);
        free(gaps);
        fprintf(stderr, "Error: Failed to allocate bounds array\n");
        return NULL;
    }

    int count = 0;
    int inChar = 0;
    int charStart = 0;
    int gapStart = 0;

    for (int x = 0; x < width; x++) {
        if (projection[x] > 0) {
            // Foreground column
            if (!inChar) {
                // Starting a new character
                if (count > 0) {
                    // Record gap width before this character
                    gaps[count - 1] = x - gapStart;
                }
                charStart = x;
                inChar = 1;
            }
        } else {
            // Background column
            if (inChar) {
                // Ending a character
                int charWidth = x - charStart;
                if (charWidth >= minCharWidth) {
                    // Valid character
                    if (count >= capacity) {
                        capacity *= 2;
                        bounds = realloc(bounds, capacity * sizeof(CharBounds));
                        gaps = realloc(gaps, capacity * sizeof(int));
                        if (!bounds || !gaps) {
                            free(bounds);
                            free(gaps);
                            return NULL;
                        }
                    }
                    bounds[count].left = charStart;
                    bounds[count].right = x - 1;
                    gaps[count] = 0;  // Will be filled when next char starts
                    count++;
                }
                gapStart = x;
                inChar = 0;
            }
        }
    }

    // Handle character that extends to edge
    if (inChar) {
        int charWidth = width - charStart;
        if (charWidth >= minCharWidth) {
            if (count >= capacity) {
                capacity *= 2;
                bounds = realloc(bounds, capacity * sizeof(CharBounds));
                gaps = realloc(gaps, capacity * sizeof(int));
            }
            bounds[count].left = charStart;
            bounds[count].right = width - 1;
            gaps[count] = 0;
            count++;
        }
    }

    *numChars = count;
    *gapWidths = gaps;
    return bounds;
}

// Find vertical bounding box for a character region
static void findVerticalBounds(uint8_t *binary, int width, int height,
                               int left, int right, int *top, int *bottom) {
    *top = height;
    *bottom = 0;

    for (int y = 0; y < height; y++) {
        for (int x = left; x <= right; x++) {
            if (binary[y * width + x] == 1) {
                if (y < *top) *top = y;
                if (y > *bottom) *bottom = y;
            }
        }
    }

    // Fallback if empty
    if (*top > *bottom) {
        *top = 0;
        *bottom = height - 1;
    }
}

// Calculate center of mass of foreground pixels
static void calculateCenterOfMass(uint8_t *binary, int width, int height,
                                   int left, int right, int top, int bottom,
                                   float *comX, float *comY) {
    float sumX = 0, sumY = 0;
    int count = 0;

    for (int y = top; y <= bottom; y++) {
        for (int x = left; x <= right; x++) {
            if (binary[y * width + x] == 1) {
                sumX += (x - left);  // Relative to bounding box
                sumY += (y - top);
                count++;
            }
        }
    }

    if (count > 0) {
        *comX = sumX / count;
        *comY = sumY / count;
    } else {
        // Fallback to geometric center
        *comX = (right - left) / 2.0f;
        *comY = (bottom - top) / 2.0f;
    }
}

// Extract and resize a single character to target size
// Uses center of mass for proper centering (critical for NN recognition)
// Now uses grayscale image (like ImagePreprocess) for better quality
static float* extractCharacter(uint8_t *gray, uint8_t *binary, int imgWidth, int imgHeight,
                               int left, int right, int targetSize, uint8_t threshold, int invert) {
    // 1. Find vertical bounds using binary image
    int top, bottom;
    findVerticalBounds(binary, imgWidth, imgHeight, left, right, &top, &bottom);

    int charW = right - left + 1;
    int charH = bottom - top + 1;

    // 2. Calculate center of mass using binary image
    float comX, comY;
    calculateCenterOfMass(binary, imgWidth, imgHeight, left, right, top, bottom, &comX, &comY);

    // 3. Create square canvas - size based on max dimension + padding
    int maxDim = (charW > charH) ? charW : charH;
    int margin = maxDim / 4;  // 25% margin for proper padding
    if (margin < 2) margin = 2;
    int squareSize = maxDim + 2 * margin;

    uint8_t *square = (uint8_t*)malloc(squareSize * squareSize);
    if (!square) return NULL;
    memset(square, 255, squareSize * squareSize);  // White background (like ImagePreprocess)

    // 4. Center by center of mass (NOT by bounding box center)
    float squareCenter = squareSize / 2.0f;
    int offsetX = (int)(squareCenter - comX);
    int offsetY = (int)(squareCenter - comY);

    // 5. Copy GRAYSCALE pixels to square, centered by COM
    for (int y = top; y <= bottom; y++) {
        for (int x = left; x <= right; x++) {
            int destX = (x - left) + offsetX;
            int destY = (y - top) + offsetY;
            if (destX >= 0 && destX < squareSize && destY >= 0 && destY < squareSize) {
                square[destY * squareSize + destX] = gray[y * imgWidth + x];
            }
        }
    }

    // 6. Resize to target size using bilinear interpolation (grayscale)
    uint8_t *resizedGray = (uint8_t*)malloc(targetSize * targetSize);
    if (!resizedGray) {
        free(square);
        return NULL;
    }

    float scaleX = (float)squareSize / targetSize;
    float scaleY = (float)squareSize / targetSize;

    for (int y = 0; y < targetSize; y++) {
        for (int x = 0; x < targetSize; x++) {
            float srcX = x * scaleX;
            float srcY = y * scaleY;

            int x0 = (int)srcX;
            int y0 = (int)srcY;
            int x1 = (x0 + 1 < squareSize) ? x0 + 1 : x0;
            int y1 = (y0 + 1 < squareSize) ? y0 + 1 : y0;

            float fx = srcX - x0;
            float fy = srcY - y0;

            float p00 = square[y0 * squareSize + x0];
            float p01 = square[y0 * squareSize + x1];
            float p10 = square[y1 * squareSize + x0];
            float p11 = square[y1 * squareSize + x1];

            float value = (1 - fx) * (1 - fy) * p00 +
                          fx * (1 - fy) * p01 +
                          (1 - fx) * fy * p10 +
                          fx * fy * p11;

            resizedGray[y * targetSize + x] = (uint8_t)value;
        }
    }
    free(square);

    // 7. Binarize and normalize (same as ImagePreprocess)
    float *normalized = (float*)malloc(targetSize * targetSize * sizeof(float));
    if (!normalized) {
        free(resizedGray);
        return NULL;
    }

    // Use Otsu on the resized character
    uint8_t charThreshold = calculateOtsuThreshold(resizedGray, targetSize * targetSize);

    for (int i = 0; i < targetSize * targetSize; i++) {
        // Dark pixels = foreground (1.0f), light = background (0.0f)
        float value = (resizedGray[i] < charThreshold) ? 1.0f : 0.0f;
        // Note: NOT inverting here because training uses invertColors=-1 (auto-detect)
        // and the phrase images have light backgrounds too, so same detection applies
        (void)invert;  // suppress warning
        normalized[i] = value;
    }

    free(resizedGray);
    return normalized;
}

// --- Public API implementation ---

int charSequenceAddSpace(CharSequence *seq) {
    if (seq->count >= seq->capacity) {
        int newCapacity = seq->capacity * 2;
        float **newChars = realloc(seq->chars, newCapacity * sizeof(float*));
        if (!newChars) {
            fprintf(stderr, "Error: Failed to expand CharSequence\n");
            return -1;
        }
        seq->chars = newChars;
        seq->capacity = newCapacity;
    }

    seq->chars[seq->count] = NULL;  // NULL indicates space
    seq->count++;
    return 0;
}

int charSequenceAddChar(CharSequence *seq, float *charData, int charSize) {
    if (!charData) return -1;

    if (seq->count >= seq->capacity) {
        int newCapacity = seq->capacity * 2;
        float **newChars = realloc(seq->chars, newCapacity * sizeof(float*));
        if (!newChars) {
            fprintf(stderr, "Error: Failed to expand CharSequence\n");
            return -1;
        }
        seq->chars = newChars;
        seq->capacity = newCapacity;
    }

    seq->chars[seq->count] = charData;
    seq->count++;
    seq->charSize = charSize;
    return 0;
}

void freeCharSequence(CharSequence *seq) {
    if (!seq) return;

    if (seq->chars) {
        for (int i = 0; i < seq->count; i++) {
            if (seq->chars[i]) {
                free(seq->chars[i]);
            }
        }
        free(seq->chars);
    }

    free(seq);
}

CharSequence* segmentPhrase(RawImage *img, SegmenterConfig cfg) {
    if (!img || !img->data) {
        fprintf(stderr, "Error: Invalid image passed to segmentPhrase\n");
        return NULL;
    }

    // 1. Convert to grayscale
    uint8_t *gray = convertToGrayscale(img);
    if (!gray) return NULL;

    // 2. Determine if we need to invert colors
    int invert = shouldInvertColors(gray, img->width, img->height);

    // 3. Binarize the image (for segmentation only)
    uint8_t threshold = calculateOtsuThreshold(gray, img->width * img->height);
    uint8_t *binary = (uint8_t*)malloc(img->width * img->height);
    if (!binary) {
        free(gray);
        return NULL;
    }

    for (int i = 0; i < img->width * img->height; i++) {
        int isForeground;
        if (invert) {
            isForeground = (gray[i] < threshold);
        } else {
            isForeground = (gray[i] >= threshold);
        }
        binary[i] = isForeground ? 1 : 0;
    }
    // NOTE: Keep gray for character extraction!

    // 4. Compute vertical projection
    int *projection = computeVerticalProjection(binary, img->width, img->height);
    if (!projection) {
        free(binary);
        free(gray);
        return NULL;
    }

    // 5. Find character boundaries
    int numChars = 0;
    int *gapWidths = NULL;
    CharBounds *bounds = findCharBoundaries(projection, img->width, cfg.minCharWidth,
                                            &numChars, &gapWidths);
    free(projection);

    if (!bounds || numChars == 0) {
        free(binary);
        free(gray);
        free(bounds);
        free(gapWidths);
        fprintf(stderr, "Warning: No characters found in image\n");
        return NULL;
    }

    // 6. Calculate gaps between characters for space detection
    int maxGap = 0;
    int minGap = 9999;
    for (int i = 0; i < numChars - 1; i++) {
        int gap = bounds[i+1].left - bounds[i].right - 1;
        if (gap > maxGap) maxGap = gap;
        if (gap < minGap && gap > 0) minGap = gap;
    }
    
    // Space threshold: adaptive based on actual gaps
    int spaceThreshold;
    if (cfg.spaceThreshold > 0) {
        spaceThreshold = cfg.spaceThreshold;
    } else if (maxGap > minGap * 2) {
        spaceThreshold = (minGap + maxGap) / 2;
    } else {
        spaceThreshold = maxGap + 1;
    }

    // 7. Create CharSequence
    CharSequence *seq = (CharSequence*)malloc(sizeof(CharSequence));
    if (!seq) {
        free(binary);
        free(gray);
        free(bounds);
        free(gapWidths);
        return NULL;
    }

    seq->chars = (float**)malloc(INITIAL_CAPACITY * sizeof(float*));
    if (!seq->chars) {
        free(seq);
        free(binary);
        free(gray);
        free(bounds);
        free(gapWidths);
        return NULL;
    }
    seq->count = 0;
    seq->capacity = INITIAL_CAPACITY;
    seq->charSize = cfg.targetSize * cfg.targetSize;

    // 8. Extract each character (using GRAY for quality, BINARY for bounds)
    for (int i = 0; i < numChars; i++) {
        // Check if there's a space before this character (except first)
        if (i > 0 && gapWidths[i - 1] > spaceThreshold) {
            charSequenceAddSpace(seq);
        }

        // Extract character using both gray (for pixels) and binary (for bounds)
        float *charData = extractCharacter(gray, binary, img->width, img->height,
                                           bounds[i].left, bounds[i].right,
                                           cfg.targetSize, threshold, invert);
        if (charData) {
            charSequenceAddChar(seq, charData, cfg.targetSize * cfg.targetSize);
        }
    }

    free(binary);
    free(gray);
    free(bounds);
    free(gapWidths);

    printf("Segmented %d elements (chars + spaces)\n", seq->count);
    return seq;
}