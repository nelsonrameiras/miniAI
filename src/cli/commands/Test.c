#include "../../../headers/cli/Commands.h"
#include "../../../headers/dataset/Dataset.h"
#include "../../../headers/dataset/TestUtils.h"
#include "../../../headers/core/Model.h"
#include "../../../headers/core/Glue.h"
#include "../../../headers/Utils.h"
#include "../../../headers/image/ImageLoader.h"
#include "../../../headers/image/ImagePreprocess.h"
#include "../../../AIHeader.h"
#include "../../../IO/MemoryDatasets.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern TrainingConfig g_trainConfig;

// Forward declarations
static int testSingleImage(CommandArgs args);
static int testDataset(CommandArgs args);

int cmdTest(CommandArgs args) {
    printf("=== TESTING MODE ===\n\n");
    
    if (!args.modelFile) {
        fprintf(stderr, "Error: test command requires --model option\n");
        return 1;
    }
    
    // If image provided, test single image
    if (args.imageFile) {
        return testSingleImage(args);
    }
    
    // Otherwise, test on dataset
    return testDataset(args);
}

static int testSingleImage(CommandArgs args) {
    printf("Testing single image: %s\n\n", args.imageFile);

    // Check if model name matches expected type (PNG)
    if (strstr(args.modelFile, "_png") == NULL && strstr(args.modelFile, "PNG") == NULL) {
        printf("WARNING: Testing image with model '%s'\n", args.modelFile);
        printf("         Model name suggests static dataset, but testing with image (PNG mode).\n");
        printf("         If you get dimension mismatch errors, use the correct PNG model.\n");
        printf("         Expected model: models/digit_brain_png.bin OR models/alpha_brain_png.bin\n\n");
    }
    
    // Create arenas
    Arena *perm = arenaInit(8 * MB);
    Arena *scratch = arenaInit(2 * MB);
    
    if (!perm || !scratch) {
        fprintf(stderr, "Error: Could not allocate memory\n");
        return 1;
    }
    
    // Determine output size and character map
    const char *charMap;
    int outputSize;
    
    if (args.dataset == DATASET_SPEC_DIGITS) {
        charMap = CHARMAP_DIGITS;
        outputSize = 10;
    } else {
        charMap = CHARMAP_ALPHA;
        outputSize = 62;
    }
    
    int inputSize = args.gridSize * args.gridSize;
    
    // Create model
    int dims[] = {inputSize, g_trainConfig.hiddenSize, outputSize};
    Model *model = modelCreate(perm, dims, NUM_DIMS);
    
    if (!model) {
        fprintf(stderr, "Error: Could not create model\n");
        arenaFree(perm);
        arenaFree(scratch);
        return 1;
    }
    
    // Load model
    printf("Loading model from %s\n", args.modelFile);
    if (modelLoad(model, args.modelFile) != 0) {
        fprintf(stderr, "Error: Could not load model\n");
        arenaFree(perm);
        arenaFree(scratch);
        return 1;
    }
    
    // Load and preprocess image
    printf("Loading image: %s\n", args.imageFile);
    RawImage *img = imageLoad(args.imageFile);
    
    if (!img) {
        fprintf(stderr, "Error: Could not load image\n");
        arenaFree(perm);
        arenaFree(scratch);
        return 1;
    }
    
    printf("Image size: %dx%d pixels, %d channels\n", img->width, img->height, img->channels);
    printf("Preprocessing to %dx%d grid...\n\n", args.gridSize, args.gridSize);
    
    PreprocessConfig cfg = {
        .targetSize = args.gridSize,
        .threshold = 0.5f,
        .invertColors = 0
    };
    
    float *processed = imagePreprocess(img, cfg);
    imageFree(img);
    
    if (!processed) {
        fprintf(stderr, "Error: Could not preprocess image\n");
        arenaFree(perm);
        arenaFree(scratch);
        return 1;
    }
    
    // Display preprocessed image
    printDigit(processed, args.gridSize);
    printf("\n");
    
    // Run prediction
    arenaReset(scratch);
    Tensor *input = tensorAlloc(scratch, inputSize, 1);
    memcpy(input->data, processed, inputSize * sizeof(float));
    
    float confidence;
    int prediction = gluePredict(model, input, scratch, &confidence);
    
    printf("=== PREDICTION ===\n");
    printf("Character: '%c'\n", charMap[prediction]);
    printf("Confidence: %.2f%%\n", confidence * 100);
    
    // Show alternative predictions
    Tensor *output = glueForward(model, input, scratch);
    Tensor *probs = tensorAlloc(scratch, outputSize, 1);
    tensorSoftmax(probs, output);
    
    // Find top 5 predictions
    printf("\nTop 5 predictions:\n");
    for (int rank = 0; rank < 5 && rank < outputSize; rank++) {
        int maxIdx = 0;
        for (int i = 1; i < outputSize; i++) {
            if (probs->data[i] > probs->data[maxIdx]) {
                maxIdx = i;
            }
        }
        
        printf("  %d. '%c' - %.2f%%\n", rank + 1, charMap[maxIdx], probs->data[maxIdx] * 100);
        probs->data[maxIdx] = -1; // Remove from consideration
    }
    
    // Cleanup
    free(processed);
    arenaFree(perm);
    arenaFree(scratch);
    
    return 0;
}

static int testDataset(CommandArgs args) {
    printf("Testing on dataset\n\n");
    
    // Create arenas
    Arena *perm = arenaInit(16 * MB);
    Arena *scratch = arenaInit(4 * MB);
    
    if (!perm || !scratch) {
        fprintf(stderr, "Error: Could not allocate memory\n");
        return 1;
    }
    
    // Determine dataset parameters
    const char *charMap;
    int outputSize;
    
    if (args.dataset == DATASET_SPEC_DIGITS) {
        charMap = CHARMAP_DIGITS;
        outputSize = 10;
    } else {
        charMap = CHARMAP_ALPHA;
        outputSize = 62;
    }
    
    // Create dataset
    Dataset *ds = NULL;
    
    if (args.useStatic) {
        // Static in-memory dataset
        printf("Dataset: Static in-memory\n");
        printf("Grid: %dx%d (%d inputs)\n", args.gridSize, args.gridSize, args.gridSize * args.gridSize);
        printf("Classes: %d\n\n", outputSize);
        
        float *data = (args.dataset == DATASET_SPEC_DIGITS) ? (float*)digits : (float*)dataset;
        ds = datasetCreateMemory(args.gridSize * args.gridSize, outputSize, args.gridSize,
                                data, args.dataset == DATASET_SPEC_DIGITS ? "DIGITS" : "ALPHANUMERIC",
                                args.modelFile, charMap);
    } else if (args.dataPath) {
        // PNG directory
        printf("Dataset: PNG from %s\n", args.dataPath);
        printf("Grid: %dx%d (%d inputs)\n", args.gridSize, args.gridSize, args.gridSize * args.gridSize);
        printf("Classes: %d\n\n", outputSize);
        
        ds = datasetCreatePNG(args.dataPath, args.gridSize, outputSize,
                             args.dataset == DATASET_SPEC_DIGITS ? "DIGITS" : "ALPHANUMERIC",
                             args.modelFile, charMap);
    } else {
        fprintf(stderr, "Error: No dataset specified (use --static or --data)\n");
        arenaFree(perm);
        arenaFree(scratch);
        return 1;
    }
    
    if (!ds) {
        fprintf(stderr, "Error: Could not create dataset\n");
        arenaFree(perm);
        arenaFree(scratch);
        return 1;
    }
    
    // Create model
    int dims[] = {ds->inputSize, g_trainConfig.hiddenSize, ds->outputSize};
    Model *model = modelCreate(perm, dims, NUM_DIMS);
    
    if (!model) {
        fprintf(stderr, "Error: Could not create model\n");
        datasetFree(ds);
        arenaFree(perm);
        arenaFree(scratch);
        return 1;
    }
    
    printf("Model: %d -> %d -> %d\n", dims[0], dims[1], dims[2]);
    printf("Loading model from %s\n\n", args.modelFile);
    
    // Load model
    if (modelLoad(model, args.modelFile) != 0) {
        fprintf(stderr, "Error: Could not load model\n");
        datasetFree(ds);
        arenaFree(perm);
        arenaFree(scratch);
        return 1;
    }
    
    // Run full test suite
    testPerfect(model, ds, scratch);
    testRobustness(model, ds, scratch);
    visualDemo(model, ds, scratch);
    displayConfusionMatrix(model, ds, scratch);
    
    // Cleanup
    datasetFree(ds);
    arenaFree(perm);
    arenaFree(scratch);
    
    printf("\nTesting complete!\n");
    return 0;
}