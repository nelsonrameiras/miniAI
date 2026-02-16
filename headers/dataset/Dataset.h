#ifndef _DATASET_H
#define _DATASET_H

#include "../core/Arena.h"
#include "../image/Segmenter.h"

// Dataset type enumeration
typedef enum {
    DATASET_MEMORY,    // In-memory array (like digits or alphabet)
    DATASET_PNG,       // PNG files from directory
    DATASET_PHRASE     // Single phrase image (for recognition)
} DatasetType;

// Unified dataset structure
typedef struct {
    DatasetType type;
    
    // Common fields
    int inputSize;      // Size of each sample (25, 64, 256, etc.)
    int outputSize;     // Number of classes (10, 62, etc.)
    int gridSize;       // Grid dimension (5, 8, 16, etc.)
    const char *name;   // Dataset name (e.g., "DIGITS", "ALPHANUMERIC")
    const char *saveFile; // Model save path
    const char *charMap;  // Character mapping string
    int isStatic;       // 1 if static in-memory, 0 if PNG
    
    // Type-specific data
    union {
        struct {
            float *data;              // For DATASET_MEMORY (flat array)
        } memory;
        
        struct {
            float **samples;          // For DATASET_PNG (array of sample pointers)
            int numSamples;           // Number of loaded samples
            int totalSlots;           // Total allocated slots
        } png;
        
        struct {
            CharSequence *sequence;   // For DATASET_PHRASE
            char *imagePath;          // Path to phrase image
        } phrase;
    };
} Dataset;

// Dataset creation functions
Dataset* datasetCreateMemory(int inputSize, int outputSize, int gridSize,
                            float *data, const char *name, 
                            const char *saveFile, const char *charMap);

Dataset* datasetCreatePNG(const char *dir, int gridSize, 
                         int outputSize, const char *name,
                         const char *saveFile, const char *charMap);

Dataset* datasetCreatePhrase(const char *imagePath, int gridSize,
                            const char *modelFile, const char *charMap);

// Dataset access functions
float* datasetGetSample(Dataset *ds, int idx);
int datasetGetNumSamples(Dataset *ds);

// Dataset cleanup
void datasetFree(Dataset *ds);

// Character maps (shared constants)
extern const char *CHARMAP_DIGITS;
extern const char *CHARMAP_ALPHA;

#endif // _DATASET_H