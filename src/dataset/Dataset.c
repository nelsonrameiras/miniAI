#include "../../headers/dataset/Dataset.h"
#include "../../headers/image/ImageLoader.h"
#include "../../headers/image/ImagePreprocess.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <dirent.h>

// Character map constants
const char *CHARMAP_DIGITS = "0123456789";
const char *CHARMAP_ALPHA = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

// ===== MEMORY DATASET =====

Dataset* datasetCreateMemory(int inputSize, int outputSize, int gridSize,
                            float *data, const char *name, 
                            const char *saveFile, const char *charMap) {
    Dataset *ds = (Dataset*)malloc(sizeof(Dataset));
    if (!ds) return NULL;
    
    ds->type = DATASET_MEMORY;
    ds->inputSize = inputSize;
    ds->outputSize = outputSize;
    ds->gridSize = gridSize;
    ds->name = name;
    ds->saveFile = saveFile;
    ds->charMap = charMap;
    ds->isStatic = 1;  // Static in-memory dataset
    ds->memory.data = data;
    
    return ds;
}

// ===== PNG DATASET =====

Dataset* datasetCreatePNG(const char *dir, int gridSize, 
                         int outputSize, const char *name,
                         const char *saveFile, const char *charMap) {
    Dataset *ds = (Dataset*)malloc(sizeof(Dataset));
    if (!ds) return NULL;
    
    ds->type = DATASET_PNG;
    ds->inputSize = gridSize * gridSize;
    ds->outputSize = outputSize;
    ds->gridSize = gridSize;
    ds->name = name;
    ds->saveFile = saveFile;
    ds->charMap = charMap;
    ds->isStatic = 0;  // PNG dataset, not static
    
    int numChars = strlen(charMap);
    ds->png.samples = (float**)malloc(numChars * sizeof(float*));
    if (!ds->png.samples) {
        free(ds);
        return NULL;
    }
    
    ds->png.numSamples = 0;
    ds->png.totalSlots = numChars;
    
    PreprocessConfig cfg = {
        .targetSize = gridSize,
        .threshold = 0.5f,
        .invertColors = 0
    };
    
    for (int i = 0; i < numChars; i++) {
        char filename[256];
        char c = charMap[i];
        
        // Construct filename based on grid size
        if (gridSize == 16) {
            snprintf(filename, sizeof(filename), "%s/%03d_%c.png", dir, (int)c, c);
        } else {
            snprintf(filename, sizeof(filename), "%s/%c.png", dir, c);
        }
        
        // Check if file exists
        struct stat st;
        if (stat(filename, &st) != 0) {
            ds->png.samples[i] = NULL;
            continue;
        }
        
        // Load and preprocess image
        RawImage *img = imageLoad(filename);
        if (!img) {
            ds->png.samples[i] = NULL;
            continue;
        }
        
        float *processed = imagePreprocess(img, cfg);
        imageFree(img);
        
        if (!processed) {
            ds->png.samples[i] = NULL;
            continue;
        }
        
        ds->png.samples[i] = processed;
        ds->png.numSamples++;
    }
    
    printf("Loaded %d/%d PNG samples from %s\n", ds->png.numSamples, numChars, dir);
    
    return ds;
}

// ===== PHRASE DATASET =====

Dataset* datasetCreatePhrase(const char *imagePath, int gridSize,
                            const char *modelFile, const char *charMap) {
    Dataset *ds = (Dataset*)malloc(sizeof(Dataset));
    if (!ds) return NULL;
    
    ds->type = DATASET_PHRASE;
    ds->inputSize = gridSize * gridSize;
    ds->outputSize = strlen(charMap);
    ds->gridSize = gridSize;
    ds->name = "PHRASE";
    ds->saveFile = modelFile;
    ds->charMap = charMap;
    ds->isStatic = 0;  // Phrase uses PNG processing, not static
    
    // Load image
    RawImage *img = imageLoad(imagePath);
    if (!img) {
        fprintf(stderr, "Error: Could not load image %s\n", imagePath);
        free(ds);
        return NULL;
    }
    
    // Segment phrase
    SegmenterConfig cfg = defaultSegmenterConfig(gridSize);
    CharSequence *seq = segmentPhrase(img, cfg);
    imageFree(img);
    
    if (!seq) {
        fprintf(stderr, "Error: Could not segment phrase from %s\n", imagePath);
        free(ds);
        return NULL;
    }
    
    ds->phrase.sequence = seq;
    ds->phrase.imagePath = strdup(imagePath);
    
    return ds;
}

// ===== DATASET ACCESS =====

float* datasetGetSample(Dataset *ds, int idx) {
    if (!ds) return NULL;
    
    switch(ds->type) {
        case DATASET_MEMORY:
            if (idx < 0 || idx >= ds->outputSize) return NULL;
            return ds->memory.data + (idx * ds->inputSize);
            
        case DATASET_PNG:
            if (idx < 0 || idx >= ds->png.totalSlots) return NULL;
            return ds->png.samples[idx];
            
        case DATASET_PHRASE:
            if (idx < 0 || idx >= ds->phrase.sequence->count) return NULL;
            return ds->phrase.sequence->chars[idx];
            
        default:
            return NULL;
    }
}

int datasetGetNumSamples(Dataset *ds) {
    if (!ds) return 0;
    
    switch(ds->type) {
        case DATASET_MEMORY:
            return ds->outputSize;
            
        case DATASET_PNG:
            return ds->png.numSamples;
            
        case DATASET_PHRASE:
            return ds->phrase.sequence ? ds->phrase.sequence->count : 0;
            
        default:
            return 0;
    }
}

// ===== CLEANUP =====

void datasetFree(Dataset *ds) {
    if (!ds) return;
    
    switch(ds->type) {
        case DATASET_MEMORY:
            // Memory dataset doesn't own the data pointer
            break;
            
        case DATASET_PNG:
            if (ds->png.samples) {
                for (int i = 0; i < ds->png.totalSlots; i++) {
                    if (ds->png.samples[i]) {
                        free(ds->png.samples[i]);
                    }
                }
                free(ds->png.samples);
            }
            break;
            
        case DATASET_PHRASE:
            if (ds->phrase.sequence) {
                freeCharSequence(ds->phrase.sequence);
            }
            if (ds->phrase.imagePath) {
                free(ds->phrase.imagePath);
            }
            break;
    }
    
    free(ds);
}