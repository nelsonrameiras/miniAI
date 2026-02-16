#include "../../../headers/cli/Commands.h"
#include "../../../headers/dataset/Dataset.h"
#include "../../../headers/dataset/TestUtils.h"
#include "../../../headers/core/Model.h"
#include "../../../AIHeader.h"
#include "../../../IO/MemoryDatasets.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern TrainingConfig g_trainConfig;

int cmdTrain(CommandArgs args) {
    printf("=== TRAINING MODE ===\n\n");
    
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
    
    // Load best configuration if available
    const char *configFileToLoad = args.configFile;
    char generatedConfig[256];

    // If the user did not pass the flag --config, we gen the default best configuration path
    if (!configFileToLoad) {
        const char *datasetName = args.dataset == DATASET_SPEC_DIGITS ? "digits" : "alpha";
        const char *datasetType = args.useStatic ? "static" : "png";
        snprintf(generatedConfig, sizeof(generatedConfig), "IO/configs/best_config_%s_%s.txt",
                 datasetName, datasetType);
        configFileToLoad = generatedConfig;
    }
    
    loadBestConfig(configFileToLoad);
    
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
    printf("Learning rate: %.4f\n", g_trainConfig.learningRate);
    printf("Regularization: %.6f\n\n", LAMBDA);
    
    // Train or load
    if (args.loadModel) {
        printf("Loading existing model from %s\n\n", args.modelFile);
        if (modelLoad(model, args.modelFile) != 0) {
            fprintf(stderr, "Error: Could not load model\n");
            datasetFree(ds);
            arenaFree(perm);
            arenaFree(scratch);
            return 1;
        }
    } else {
        trainModel(model, ds, scratch);
    }
    
    // Run tests
    testPerfect(model, ds, scratch);
    testRobustness(model, ds, scratch);
    visualDemo(model, ds, scratch);
    displayConfusionMatrix(model, ds, scratch);
    
    // Cleanup
    datasetFree(ds);
    arenaFree(perm);
    arenaFree(scratch);
    
    printf("\nTraining complete!\n");
    return 0;
}