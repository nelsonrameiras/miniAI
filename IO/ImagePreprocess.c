#include "../headers/ImagePreprocess.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

typedef struct {
    int top, bottom, left, right;
} BoundingBox;

// convert RGB pixel to grayscale (luminance)
static inline uint8_t rgbToGray(uint8_t r, uint8_t g, uint8_t b) {
    return (uint8_t)(0.299f * r + 0.587f * g + 0.114f * b);
}

// convert image to grayscale (if needed)
static uint8_t* convertToGrayscale(RawImage *img) {
    int totalPixels = img->width * img->height;
    uint8_t *gray = (uint8_t*)malloc(totalPixels);
    if (!gray) { fprintf(stderr, "Error: failed to allocate grayscale buffer\n"); return NULL; }

    if (img->channels == 1) {
        // already grayscale
        memcpy(gray, img->data, totalPixels);
    } else if (img->channels >= 3) {
        // convert RGB(A) to grayscale
        for (int i = 0; i < totalPixels; i++) {
            int idx = i * img->channels;
            gray[i] = rgbToGray( img->data[idx], img->data[idx + 1], img->data[idx + 2]);
        }
    }

    return gray;
}

static uint8_t calculateOtsuThreshold(uint8_t *gray, int totalPixels) {
    int histogram[256] = {0};

    // build histogram
    for (int i = 0; i < totalPixels; i++) histogram[gray[i]]++;

    // total weighted sum
    float sum = 0;
    for (int i = 0; i < 256; i++) sum += i * histogram[i];

    float sumB = 0;
    int wB = 0;
    float maxVariance = 0;
    uint8_t threshold = 128;  // default fallback

    // find threshold that maximizes between-class variance
    for (int i = 0; i < 256; i++) {
        wB += histogram[i];
        if (wB == 0) continue;

        int wF = totalPixels - wB;
        if (wF == 0) break;

        sumB += i * histogram[i];
        float mB = sumB / wB;
        float mF = (sum - sumB) / wF;

        float variance = wB * wF * (mB - mF) * (mB - mF);

        if (variance > maxVariance) {
            maxVariance = variance;
            threshold = i;
        }
    }

    return threshold;
}

static BoundingBox findBoundingBox(uint8_t *gray, int width, int height, uint8_t threshold) {
    BoundingBox bbox = {height, 0, width, 0};
    int foundPixel = 0;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (gray[y * width + x] < threshold) {  // foreground pixel
                foundPixel = 1;
                if (y < bbox.top) bbox.top = y;
                if (y > bbox.bottom) bbox.bottom = y;
                if (x < bbox.left) bbox.left = x;
                if (x > bbox.right) bbox.right = x;
            }
        }
    }

    // if nothing detected, return full image
    if (!foundPixel) {
        bbox.top = 0;
        bbox.bottom = height - 1;
        bbox.left = 0;
        bbox.right = width - 1;
    }

    return bbox;
}

static int shouldInvertColors(uint8_t *gray, int width, int height) {
    // estimate bgr color from border pixels
    float borderSum = 0;
    int borderCount = 0;

    // top and bottom borders
    for (int x = 0; x < width; x++) {
        borderSum += gray[x];
        borderSum += gray[(height - 1) * width + x];
        borderCount += 2;
    }

    // left and right borders (excluding corners)
    for (int y = 1; y < height - 1; y++) {
        borderSum += gray[y * width];
        borderSum += gray[y * width + (width - 1)];
        borderCount += 2;
    }

    float avgBorder = borderSum / borderCount;

    // if bgr is light, invert foreground
    return (avgBorder > 128) ? 1 : 0;
}

static uint8_t* extractAndCenter(uint8_t *gray, int width, int height, uint8_t threshold, int *outSize) {
    BoundingBox bbox = findBoundingBox(gray, width, height, threshold);

    int letterW = bbox.right - bbox.left + 1;
    int letterH = bbox.bottom - bbox.top + 1;

    int maxDim = (letterW > letterH) ? letterW : letterH;
    int margin = maxDim / 6;
    int squareSize = maxDim + 2 * margin;

    uint8_t *square = malloc(squareSize * squareSize);
    if (!square) return NULL;
    memset(square, 255, squareSize * squareSize);

    int offsetX = (squareSize - letterW) / 2;
    int offsetY = (squareSize - letterH) / 2;

    for (int y = 0; y < letterH; y++)
        for (int x = 0; x < letterW; x++)
            square[(offsetY + y) * squareSize + (offsetX + x)] = gray[(bbox.top + y) * width + (bbox.left + x)];

    *outSize = squareSize;
    return square;
}

static uint8_t* resizeImage(uint8_t *src, int srcW, int srcH, int targetSize) {
    int totalPixels = targetSize * targetSize;
    uint8_t *resized = (uint8_t*)malloc(totalPixels);
    if (!resized) { fprintf(stderr, "Error: failed to allocate resize buffer\n"); return NULL;}

    float scaleX = (float)srcW / targetSize;
    float scaleY = (float)srcH / targetSize;

    for (int y = 0; y < targetSize; y++) {
        for (int x = 0; x < targetSize; x++) {
            float srcX = x * scaleX;
            float srcY = y * scaleY;

            int x0 = (int)srcX;
            int y0 = (int)srcY;
            int x1 = (x0 + 1 < srcW) ? x0 + 1 : x0;
            int y1 = (y0 + 1 < srcH) ? y0 + 1 : y0;

            float fx = srcX - x0;
            float fy = srcY - y0;

            uint8_t p00 = src[y0 * srcW + x0];
            uint8_t p01 = src[y0 * srcW + x1];
            uint8_t p10 = src[y1 * srcW + x0];
            uint8_t p11 = src[y1 * srcW + x1];

            float value =
                (1 - fx) * (1 - fy) * p00 +
                fx * (1 - fy) * p01 +
                (1 - fx) * fy * p10 +
                fx * fy * p11;

            resized[y * targetSize + x] = (uint8_t)value;
        }
    }

    return resized;
}

static float* binarizeAndNormalize(uint8_t *gray, int size, int invertColors) {
    int totalPixels = size * size;
    float *normalized = (float*)malloc(totalPixels * sizeof(float));

    if (!normalized) { fprintf(stderr, "Error: failed to allocate normalized buffer\n"); return NULL; }

    uint8_t threshold = calculateOtsuThreshold(gray, totalPixels);

    for (int i = 0; i < totalPixels; i++) {
        float value = (gray[i] < threshold) ? 1.0f : 0.0f;
        if (invertColors) value = 1.0f - value;
        normalized[i] = value;
    }

    return normalized;
}

float* imagePreprocess(RawImage *img, PreprocessConfig cfg) {
    // convert to grayscale
    uint8_t *gray = convertToGrayscale(img);
    if (!gray) return NULL;

    // auto-detect color inversion
    int invertColors = shouldInvertColors(gray, img->width, img->height);
    if (cfg.invertColors >= 0) invertColors = cfg.invertColors;

    // compute threshold for bounding box
    uint8_t threshold = calculateOtsuThreshold(gray, img->width * img->height);

    // extract and center glyph
    int centeredSize;
    uint8_t *centered = extractAndCenter(gray, img->width, img->height, threshold, &centeredSize);
    free(gray);
    if (!centered) return NULL;

    // resize to target size
    uint8_t *resized = resizeImage(centered, centeredSize, centeredSize, cfg.targetSize);
    free(centered);
    if (!resized) return NULL;

    // binarize and normalize
    float *normalized = binarizeAndNormalize(resized, cfg.targetSize, invertColors);
    free(resized);

    return normalized;
}