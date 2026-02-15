#include "../headers/image/ImageLoader.h"
#include <stdlib.h>
#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../../IO/external/stb_image.h"

RawImage* imageLoad(const char *filename) {
    RawImage *img = (RawImage*)malloc(sizeof(RawImage));
    if (!img) { fprintf(stderr, "Error: could not alloc memory for a RawImage\n"); return NULL; }
    
    // stb_image loads autom. any format
    // 0 = detecting channels automatically
    img->data = stbi_load(filename, &img->width, &img->height, &img->channels, 0);
    
    if (!img->data) { fprintf(stderr, "Error loading image: %s\n", filename); free(img); return NULL; }
    
    return img;
}

void imageFree(RawImage *img) {
    if (img) {
        if (img->data) stbi_image_free(img->data);
        free(img);
    }
}