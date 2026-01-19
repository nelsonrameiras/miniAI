#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <stdint.h>

typedef struct {
    uint8_t *data;    // Raw pixel data
    int width;
    int height;
    int channels;     // 1=grayscale, 3=RGB, 4=RGBA
} RawImage;

// Loads PNG
RawImage* imageLoad(const char *filename);

// Frees image memory
void imageFree(RawImage *img);

#endif